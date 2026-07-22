import os
import math
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

from naxi.v_0d1.gridman.config import RUNNING_CONFIG
from naxi.v_0d1.gridman.core import Gridman
from naxi.v_0d1.gridman.dataloader import StreamLoader
from naxi.v_0d1.gridman.tools import save_checkpoint, load_checkpoint, print_model_parameters

from torch.utils.tensorboard import SummaryWriter


def get_gaussian_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    k: float = 17.0, 
    last_epoch: int = -1
):
    """
    高斯退火公式: e^(-kx^2) + e^(-k)*((2k+2)x^3 - (2k+3)x^2)
    """
    def lr_lambda(current_step: int):
        # 1. Warmup 阶段：线性增加到 1.0
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. 衰减阶段：计算相对进度 x (0.0 到 1.0)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        # 限制在 [0, 1] 之间，防止越界
        x = min(max(progress, 0.0), 1.0)
        
        # 如果已经结束训练，学习率归零
        if x >= 1.0:
            return 0.0
        
        # 3. 高斯退火公式计算衰减系数
        # term1 = 1/(k*x+1) - x/(k+1)
        # term2 = (2*k+1)/(k+1)**2 * (x**2-x)
        term1 = math.exp(-k * (x ** 2))
        term2 = math.exp(-k) * ((2 * k + 2) * (x ** 3) - (2 * k + 3) * (x ** 2))
        
        return term1 + term2

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def reduce_value(value: torch.Tensor):
    if not dist.is_initialized():
        return value
    val = value.data.clone()
    # 求和
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    # 真实平均值
    return val / dist.get_world_size()


def train_model(is_sft: bool = False, grad_accum_steps: int = 1):
    config = RUNNING_CONFIG
    dtype = config.dtype

    # 初始化多卡环境
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    chunk_size = config.chunk_size
    bptt_size = config.bptt_size
    
    if is_sft:
        mode_name = 'SFT'
        model_name = f'{config.name}_sft'
        dataset_file = config.sft_train_file
        lr = config.sft_lr
        steps = config.sft_steps
    else:
        mode_name = 'PRE-TRAIN'
        model_name = f'{config.name}_pretrain'
        dataset_file = config.pretrain_train_file
        lr = config.pretrain_lr
        steps = config.pretrain_steps

    dataloader = StreamLoader(
        patch_size=config.patch_size, 
        chunk_size=chunk_size, 
        datasets=dataset_file,
        is_sft=is_sft,
        rank=local_rank,
        world_size=world_size
    )

    grid_man = Gridman(config).to(device)
    writer = None

    if local_rank == 0:
        print(f'🚀 正在初始化 {mode_name}...')
        print_model_parameters(grid_man)
        print('\n' + '>'*25 + f' 开始极速流式 {mode_name} ' + '<'*25)

        log_dir = os.path.join('log', model_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f'📊 TensorBoard 日志将保存至: {log_dir}')

    # load_checkpoint(grid_man, False, need_print=(local_rank==0))
    # grid_man.core_ouro.mem_clear()

    if is_sft:
        # 加载预训练模型
        load_checkpoint(grid_man, True, need_print=(local_rank==0))
        grid_man.core_ouro.mem_clear()

    grid_man = torch.compile(grid_man)
    # 此处为强制类型标记
    grid_man: Gridman = DDP(grid_man, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=(bptt_size==1))
    grid_man_module: Gridman = grid_man.module

    optimizer = torch.optim.AdamW(grid_man.parameters(), lr=lr)

    total_update_steps = steps // (bptt_size * grad_accum_steps)

    num_warmup_steps = int(total_update_steps * 0.05) 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_update_steps
    )

    loss_acc = torch.tensor(0.0, device=device)
    loss_acc_log = torch.tensor(0.0, device=device)

    optimizer.zero_grad()

    for step in range(steps): 
        grid_man.train()
        step_true = step + 1
        
        # 接收 Token 和 Mask
        input_patches, mask_patches = dataloader.get_batch()
        input_patches = input_patches.to(device)
        mask_patches = mask_patches.to(device)
        
        # 构造 Input 和 Target
        inputs = input_patches[:, :-1]   
        targets = input_patches[:, 1:].clone()  # 防止修改原 tensor
        target_masks = mask_patches[:, 1:]      # 与 targets 对应
        
        targets[target_masks == 0] = -100

        # 最后一次 forward 触发同步
        is_bptt_step = (step_true % bptt_size == 0)
        is_update_step = (step_true % (bptt_size * grad_accum_steps) == 0)

        sync_context = grid_man.no_sync() if not is_update_step  else nullcontext()
        
        with sync_context:
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = grid_man(inputs)
                grid_man_module.core_ouro.mem_sync()

                if (targets != -100).any():
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), 
                        targets.reshape(-1),
                        ignore_index=-100 
                    )
                else: 
                    loss = logits.sum() * 0.0
            
            loss_acc = loss_acc + loss

            with torch.no_grad():
                dist_loss = reduce_value(loss)
                loss_acc_log = loss_acc_log + dist_loss

            if is_bptt_step:
                loss_for_backward = loss_acc / (bptt_size * grad_accum_steps)
                loss_for_backward.backward()

                grid_man_module.core_ouro.mem_detach()
                loss_acc = torch.tensor(0.0, device=device)

        if is_update_step:
            total_norm = torch.nn.utils.clip_grad_norm_(grid_man.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if local_rank == 0 and step_true % (bptt_size * grad_accum_steps * 10) == 0:
                avg_loss = loss_acc_log.item() / (bptt_size * grad_accum_steps)
                writer.add_scalar('Train/Loss', avg_loss, step)
                writer.add_scalar('Train/Grad_Norm', total_norm, step)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], step)
                print(f'\n📌 Step {step_true} | Total Loss = {avg_loss:.4f}')

            loss_acc_log = torch.tensor(0.0, device=device)

        if step_true % 3600 == 0 and local_rank == 0:
            save_checkpoint(grid_man_module, is_sft)


def generate_test(is_sft: False):
    """仅进行生成测试"""
    config = RUNNING_CONFIG
    device = config.device
    tokenizer = config.tokenizer
    dtype = config.dtype

    grid_man = Gridman(config).to(device)
    load_checkpoint(grid_man, is_sft)
    grid_man.eval()
    if is_sft:
        prompt_ids_list = (
            [tokenizer.eos_token_id, tokenizer.user_token_id] + 
            tokenizer.encode('你是谁') + 
            [tokenizer.eos_token_id, tokenizer.assistant_token_id]
        )
    else:
        prompt_ids_list = tokenizer.encode('世界上最高的山是')

    prompt_ids = torch.tensor([[tokenizer.eos_token_id] + prompt_ids_list], dtype=torch.long, device=device)
    
    with torch.amp.autocast(config.device_type, dtype=dtype):
        generated_ids = grid_man.generate(prompt_ids)
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f'Gridman 🤖: {generated_text}')
    print('-' * 65)
