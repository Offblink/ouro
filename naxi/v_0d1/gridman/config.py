from dataclasses import dataclass
import os

import torch
from naxi.v_0d1.gridman.lm_kernel import ByteTokenizer
    
    
@dataclass
class Config:
    # 通用配置
    name: str = 'gridman_mini'

    embed_dim: int = 512
    block_layers: int = 4
    blocks: int = 2

    patch_size: int = 64
    
    # 分词器
    tokenizer: ByteTokenizer = ByteTokenizer()

    chunk_size: int = 128
    bptt_size: int = 8

    # L2 状态约束损失（等效原理）权重，0.0 = 关闭
    l2_weight: float = 0.0

    # 预训练配置
    pretrain_train_file: str = f'/root/autodl-tmp/pretrain.jsonl'
    pretrain_lr: float = 3e-4
    pretrain_steps: int = 1056000*2

    # SFT 配置
    sft_train_file: str = f'/root/autodl-tmp/sft.jsonl'
    sft_lr: float = 6e-5
    sft_steps: int = 1056000

    # 版本号
    version: str = 'v_0d1'

    # 运行信息
    device_type: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.bfloat16

    # 关键路径
    checkpoint_dir: str = os.path.join(os.getcwd(), 'checkpoints')
    log_dir: str = os.path.join(os.getcwd(), 'log')


GRIDMAN_MINI = Config()

GRIDMAN_SMALL = Config(
    'gridman_small',
    768,
    chunk_size=128,
    bptt_size=7,

    pretrain_lr=2.11e-4,
    pretrain_steps=1056000*2,

    sft_lr=4.2e-5,
    sft_steps=1056000*2
)

GRIDMAN_MEDIUM = Config(
    'gridman_medium',
    1280,
    chunk_size=64,
    bptt_size=7,

    pretrain_lr=1.5e-4,
    pretrain_steps=1056000*2,

    sft_lr = 1e-4,
    sft_steps = 4200000
)

GRIDMAN_LARGE = Config(
    'gridman_large',
    1856,
    chunk_size=64,
    bptt_size=6,
   
    pretrain_lr=1e-4,
    pretrain_steps=350000
)

GRIDMAN_XL = Config(
    'gridman_xl',
    2624,
    7,
    7
)


RUNNING_CONFIG = GRIDMAN_MEDIUM
