import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


HEAD_DIM = 64


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 64, base: float = 128.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self._set_cos_sin_cache(self.max_position_embeddings, self.inv_freq.device, self.inv_freq.dtype)

    def __call__(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x, seq_len)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_position_embeddings = seq_len
        t = torch.arange(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    

class OuroNorm(nn.Module):
    def __init__(self, embed_dim: int, init_bias: float = 0):
        super().__init__()
        self.embed_dim = embed_dim
    
        self.k_proj = nn.Linear(embed_dim, 1)
        self.act = nn.Sigmoid()

        nn.init.zeros_(self.k_proj.weight)
        nn.init.constant_(self.k_proj.bias, init_bias)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.act(self.k_proj(x))
        x_normed = F.normalize(x, p=2.0, dim=-1) * (self.embed_dim ** 0.5)
        return k * x_normed


class Attention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim
        self.dropout = dropout
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 初始化旋转位置编码模块
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv: torch.Tensor = self.qkv_proj(x)
        
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k  = self.q_norm(q), self.k_norm(k)
        
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output


class GateAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):  
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim

        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q: torch.Tensor = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if state.dim() == 2:
            state = state.unsqueeze(1) 
            
        kv: torch.Tensor = self.kv_proj(state)
        kv = kv.view(batch_size, 1, 2, self.num_heads, self.head_dim)
    
        kv = kv.permute(2, 0, 3, 1, 4) 
        k, v = kv[0], kv[1]

        q, k = self.q_norm(q), self.k_norm(k)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
    
        gate = torch.sigmoid(scores)
        
        if self.training and self.dropout > 0.0:
            gate = F.dropout(gate, p=self.dropout)
            
        attn_output = gate * v  
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return attn_output


class OuroStateAttention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // self.head_dim

        self.dropout = dropout

        self.state_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.gate_attn = GateAttention(self.embed_dim, self.dropout)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(self.embed_dim, self.dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        state = state.unsqueeze(1)
        state = self.state_proj(state)

        state_injection: torch.Tensor = self.gate_attn(x, state) 
        x = x + state_injection
        return state_injection + self.attn(self.norm(x))


class OuroDepthAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = embed_dim // self.head_dim

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.o_proj.weight)

    def forward(self, active_c: torch.Tensor, history_states: list[torch.Tensor]) -> torch.Tensor:
        batch_size = active_c.shape[0]
        seq_len = history_states[0].shape[1]
        num_layers = len(history_states)

        H = torch.stack(history_states, dim=2)
        
        q_norm = self.norm_q(active_c) 
        q: torch.Tensor = self.q_proj(q_norm)
        
        q = q.view(batch_size, 1, self.num_heads, 1, self.head_dim)

        H_norm = self.norm_kv(H)
        kv: torch.Tensor = self.kv_proj(H_norm)
        
        kv = kv.view(batch_size, seq_len, num_layers, 2, self.num_heads, self.head_dim)
        kv = kv.permute(3, 0, 1, 4, 2, 5)
        k, v = kv[0], kv[1]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.squeeze(-2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.o_proj(out)
    

class OuroTemporalAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = embed_dim // self.head_dim

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        with torch.no_grad():
            nn.init.zeros_(self.o_proj.weight)

    def forward(self, current_c: torch.Tensor, state_queue: torch.Tensor) -> torch.Tensor:
        batch_size = current_c.shape[0]
        
        q_norm = self.norm_q(current_c)

        q: torch.Tensor = self.q_proj(q_norm)
        q = q.view(batch_size, self.num_heads, 1, self.head_dim)
        
        kv_norm = self.norm_kv(state_queue)
        kv: torch.Tensor = self.kv_proj(kv_norm)
        kv = kv.view(batch_size, state_queue.shape[1], 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.squeeze(2).contiguous().view(batch_size, self.embed_dim)
        return self.o_proj(out)
    

class FFN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim

        self.multiple_of = 256
        hidden_dim = int(2 * (4 * self.embed_dim) / 3)
        self.hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
        
        self.w12 = nn.Linear(self.embed_dim, 2 * self.hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.w3 = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined_projected = self.w12(x)
        x_w1, x_v = torch.chunk(combined_projected, chunks=2, dim=-1)
        
        swiglu_out = self.act(x_w1) * x_v
        return self.w3(swiglu_out)
    

class OuroSTM(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch

        self.state_attn = OuroStateAttention(self.embed_dim)
        self.act = nn.SiLU()

        self.norm = nn.LayerNorm(self.embed_dim)
        self.w = nn.Linear(self.embed_dim, self.embed_dim * 4)

        self.c_state: torch.Tensor
        self._pending_c_state: torch.Tensor
        self._runtime_c_state: torch.Tensor | None = None

        self.register_buffer("c_state", torch.zeros(self.max_batch, self.embed_dim))
        self.state_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.ouro_norm = OuroNorm(self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def __call__(self, x: torch.Tensor, lock_mem: bool = False):
        return self.forward(x, lock_mem)

    def mem_detach(self):
        self._runtime_c_state = self._runtime_c_state.detach()
            
        with torch.no_grad():
            batch_size = self._runtime_c_state.shape[0]
            self.c_state[:batch_size].copy_(self._runtime_c_state)

    def mem_clear(self):
        self._runtime_c_state = None
        self.c_state.zero_()

    def mem_sync(self):
        self._runtime_c_state = self._pending_c_state    
        del self._pending_c_state

    def active_c(self, batch_size: int):
        if self._runtime_c_state is None:
            active_c = self.c_state[:batch_size].detach().clone()
        else:
            active_c = self._runtime_c_state
        return active_c

    def forward(self, x: torch.Tensor, lock_mem: bool = False):
        batch_size, seq_len, _ = x.shape

        active_c = self.active_c(batch_size)

        state_attn = self.state_attn(x, active_c)
        x = x + state_attn

        # 计算门控信号
        gates = self.w(x) 
        i, f, g, o = torch.chunk(gates, 4, dim=-1)

        i = torch.sigmoid(i)
        g: torch.Tensor = self.act(g)
        o = torch.sigmoid(o)

        v = i * g  
        f_gate = torch.sigmoid(f) 

        c_states = []
        curr_c = active_c 

        # 线性时序扫描
        for t in range(seq_len):
            curr_c = f_gate[:, t, :] * curr_c + v[:, t, :]
            c_states.append(curr_c)

        c = torch.stack(c_states, dim=1)
        h: torch.Tensor = o * self.act(self.norm(c))

        # 更新 Buffer
        if not lock_mem:
            c_last = c[:, -1, :] 
            c_last = self.state_proj(c_last)
            self._pending_c_state = self.ouro_norm(c_last) 

        return self.out_proj(h)


class OuroLayer(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int, need_mem: bool = False, need_stm: bool = False):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = HEAD_DIM
        self.num_heads = self.embed_dim // HEAD_DIM

        self.max_batch = max_batch
     
        self.need_mem = need_mem
        self.need_stm = need_stm

        self.act = nn.SiLU()
        self.state_attn = OuroStateAttention(self.embed_dim)

        if self.need_stm:
            self.ouro_stm = OuroSTM(self.embed_dim, self.max_batch)

        # 开启标准的 Delte Rule 实现
        if self.need_mem:
            self._pending_mem = None
            self._last_delta_mem = None  # L2 状态约束损失用：本层记忆更新量 dW

            # 全局记忆矩阵 
            self.mem: torch.Tensor
            self.register_buffer('mem', torch.eye(self.embed_dim).unsqueeze(0))

            self._causal_mask: torch.Tensor
            self.register_buffer('_causal_mask', torch.ones(self.embed_dim, self.embed_dim, dtype=torch.bool).tril_(), persistent=False)

            self.rotary_emb = RotaryEmbedding(dim=self.embed_dim)

            self.mem_norm = nn.LayerNorm(embed_dim)

            self.w_qkvgd = nn.Linear(embed_dim, embed_dim * 5)

            self.w_o = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.w_qkvgd.weight[3 * embed_dim : 4 * embed_dim, :], 
                    mean=0.0, std=0.02
                )
                torch.nn.init.constant_(
                    self.w_qkvgd.bias[3 * embed_dim : 4 * embed_dim], 
                    -6.0
                )

    def __call__(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False):
        return self.forward(x, last_state, lock_mem)
    
    def mem_detach(self):
        if self.need_stm:
            self.ouro_stm.mem_detach()
        if self.need_mem:
            self.mem = self.mem.detach()

    def mem_clear(self):
        if self.need_stm:
            self.ouro_stm.mem_clear()
        if self.need_mem:
            self.mem = torch.zeros(1, self.embed_dim, self.embed_dim)

    def mem_sync(self):
        if self.need_stm:
            self.ouro_stm.mem_sync()

    def forward(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len, _ = x.shape
    
        x = x + self.state_attn(x, last_state)

        if self.need_mem:
            mem_context: torch.Tensor = self.mem_norm(x)
            qkvgd: torch.Tensor= self.w_qkvgd(mem_context)
            context_q, context_k, context_v, context_g, context_d = qkvgd.chunk(5, dim=-1)

            cos, sin = self.rotary_emb(context_v, seq_len=seq_len)
            cos = cos.squeeze(1)
            sin = sin.squeeze(1)

            context_q, context_k = apply_rotary_pos_emb(context_q, context_k, cos, sin)

            context_q = self.act(context_q)
            context_q = F.normalize(context_q, p=2, dim=-1, eps=1e-5)

            context_k = self.act(context_k)
            context_k = F.normalize(context_k, p=2, dim=-1, eps=1e-5) 

            context_v = self.act(context_v)

            mem_g = torch.sigmoid(context_g) * (1.0 / seq_len)

            # 预测的 V
            v_retrieved: torch.Tensor = context_k @ self.mem

            # 计算真实 V 与预测 V 的 Delta
            delta_v = context_v - v_retrieved
            v_dyn = mem_g * delta_v
            
            # 外积更新
            delta_mem: torch.Tensor = torch.bmm(context_k.transpose(-1, -2), v_dyn)

            # L2 状态约束：捕获本层 dW（依赖当前状态的图节点）
            self._last_delta_mem = delta_mem if not lock_mem else None

            # 记忆更新
            next_mem: torch.Tensor = self.mem + delta_mem

            if not lock_mem:
                self._pending_mem = next_mem.mean(0, keepdim=True)
            
            # 历史记忆 
            mem_out_prev = context_q @ self.mem
            
            # QK 的标准注意力打分矩阵
            scores = torch.bmm(context_q, context_k.transpose(-1, -2)) 
          
            # 标准注意力
            mask = self._causal_mask[:seq_len, :seq_len]
            scores.masked_fill_(~mask, 0.0)
            mem_out_delta: torch.Tensor = torch.bmm(scores, v_dyn)
            
            # 合并输出
            mem_out = mem_out_prev
            mem_out += mem_out_delta

            # 动态门控
            mem_out = mem_out * self.act(context_d)
            x = x + self.w_o(mem_out)

            return x, scores, x

        if self.need_stm:
            stm = self.ouro_stm(x)
            return x + stm, None, x + stm

    
class OuroBlock(nn.Module):
    def __init__(self, embed_dim: int, max_batch: int, block_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch
        self.block_layers = block_layers

        self.act = nn.SiLU()

        self.ouro_self_attn_proj = nn.Parameter(torch.zeros(embed_dim, embed_dim))
        self.ouro_self_attn_norm = nn.LayerNorm(self.embed_dim)

        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_norm = nn.LayerNorm(embed_dim)

        self.ouro_self_attn_output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ouro_self_attn_gate = nn.Linear(embed_dim, embed_dim, bias=False)

        self.ouro_layers: nn.ModuleList[OuroLayer] = nn.ModuleList([
            OuroLayer(self.embed_dim, self.max_batch, (_!=0), _==0) for _ in range(self.block_layers)
        ])

        self.ffn = FFN(self.embed_dim)

        self.norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False):
        return self.forward(x, last_state, lock_mem)

    def mem_detach(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_detach()

    def mem_clear(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_clear()

    def mem_sync(self):
        for layer in self.ouro_layers:
            layer: OuroLayer
            layer.mem_sync()
    
    def forward(self, x: torch.Tensor, last_state: torch.Tensor | None = None, lock_mem: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len, _ = x.shape
        residual = x

        ouro_self_attn = torch.tensor(0.0)

        out_list = []
        for layer in self.ouro_layers:
            layer: OuroLayer
            if layer.need_mem:
                x, attn, out = layer(x, last_state, lock_mem)
                ouro_self_attn = ouro_self_attn + attn
            if layer.need_stm:
                x, _, out = layer(x, last_state, lock_mem)

            out_list.append(out)
            inner_residual = x

        # 涌现注意力 (Emergent Attention)
        scale_factor: torch.Tensor = self.embed_dim**(-0.5)

        ouro_self_attn_residual = ouro_self_attn

        x_proj: torch.Tensor = torch.matmul(x, self.ouro_self_attn_proj) * scale_factor
        ouro_self_attn = torch.bmm(ouro_self_attn_residual, x_proj)
        
        ouro_self_attn: torch.Tensor = self.act(ouro_self_attn)
        ouro_self_attn_normed: torch.Tensor = self.ouro_self_attn_norm(ouro_self_attn)

        v_states: torch.Tensor = self.w_v(self.v_norm(residual))

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        attn_bias: torch.Tensor = (ouro_self_attn_residual * scale_factor).masked_fill(~causal_mask, float('-inf'))

        ouro_self_attn_output = F.scaled_dot_product_attention(
            ouro_self_attn_normed.unsqueeze(1),
            ouro_self_attn_normed.unsqueeze(1),
            v_states.unsqueeze(1),
            attn_mask=attn_bias.unsqueeze(1),
            scale=scale_factor
        ).squeeze(1) 

        ouro_self_attn_output: torch.Tensor = self.ouro_self_attn_output_proj(ouro_self_attn_output)

        # 注意力门控
        gate = torch.sigmoid(self.act(self.ouro_self_attn_gate(residual)))
        residual = residual + gate * ouro_self_attn_output

        # 标准输出
        x = self.ffn(self.norm(residual))
        x = x + residual + inner_residual
        out_list.append(x)

        return x, out_list
    

class Ouro(nn.Module):
    """
    Ouro 标准模型
    """
    def __init__(self, embed_dim: int, max_batch: int, blocks: int, block_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_batch = max_batch
        self.blocks = blocks

        self.in_norm = nn.LayerNorm(embed_dim)
        self.in_attn = Attention(self.embed_dim)
        self.in_ffn = FFN(self.embed_dim)

        self.temporal_queue_len = 65
        self.c_state_queue: torch.Tensor
        self.register_buffer("c_state_queue", torch.zeros(max_batch, self.temporal_queue_len, embed_dim))
        self.temporal_attn = OuroTemporalAttention(embed_dim)
        self.state_ffn = FFN(self.embed_dim)
    
        self.ouro_blocks = nn.ModuleList([
            OuroBlock(embed_dim, self.max_batch, block_layers) for _ in range(blocks)
        ])

        self.attnres_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(embed_dim)) for _ in range(self.blocks - 1)
        ])
        self.attnres_k_norm = nn.LayerNorm(embed_dim)
        self.final_depth_attn = OuroDepthAttention(self.embed_dim)
        self.m_ffn = FFN(self.embed_dim)

        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_attn = Attention(self.embed_dim)
        self.out_ffn = FFN(self.embed_dim)

        self.stm = OuroSTM(self.embed_dim, self.max_batch)

        # L2 状态约束损失（等效原理）：训练时由 train.py 开启
        self.l2_enabled = False
        self._l2_state_t: torch.Tensor | None = None   # 前向前的状态 s_t（可微叶）
        self._l2_state_next: torch.Tensor | None = None  # 前向后的状态 s_{t+1}

    def __call__(self, x: torch.Tensor, lock_mem: bool = False):
        return self.forward(x, lock_mem)

    def mem_detach(self):
        self.stm.mem_detach()
        self.c_state_queue = self.c_state_queue.detach()
        for blocks in self.ouro_blocks:
            blocks: OuroBlock
            blocks.mem_detach()

    def mem_clear(self):
        self.stm.mem_clear()
        self.c_state_queue.zero_()
        for blocks in self.ouro_blocks:
            blocks: OuroBlock
            blocks.mem_clear()

    def mem_sync(self):
        self.stm.mem_sync()

        for block in self.ouro_blocks:
            block: OuroBlock
            block.mem_sync()

        pending_mems = []
        mem_layers = []

        for block in self.ouro_blocks:
            block: OuroBlock
            for layer in block.ouro_layers:
                layer: OuroLayer
                if layer.need_mem and layer._pending_mem is not None:
                    pending_mems.append(layer._pending_mem)
                    mem_layers.append(layer)

        if dist.is_initialized():
            stacked_mems = torch.stack(pending_mems, dim=0)
            
            with torch.no_grad():
                dist.all_reduce(stacked_mems, op=dist.ReduceOp.SUM)
                stacked_mems = stacked_mems / dist.get_world_size()

            for i, layer in enumerate(mem_layers):
                local_mem: torch.Tensor = pending_mems[i]
                synced_mem = stacked_mems[i]
                layer.mem = local_mem + (synced_mem - local_mem).detach()
                layer._pending_mem = None
        else:
            for layer in mem_layers:
                layer.mem = layer._pending_mem
                layer._pending_mem = None           

    def forward(self, x: torch.Tensor, lock_mem: bool = False) -> torch.Tensor:
        batch_size, _, _ = x.shape

        x0 = x

        # 标准输入
        x = x + self.in_attn(self.in_norm(x))
        x = x + self.in_ffn(x)

        # 状态获取
        base_active_c = self.stm.active_c(batch_size).to(torch.bfloat16)

        # L2 状态约束：把 s_t 变成可微叶，供 VJP 计算 J^T v
        if self.l2_enabled and not lock_mem:
            base_active_c = base_active_c.detach().clone().requires_grad_(True)
            self._l2_state_t = base_active_c
       
        queue = self.c_state_queue[:batch_size]
        queue_history = queue[:, :-1, :]
        temporal_context = self.temporal_attn(base_active_c, queue_history)
        active_c = base_active_c + temporal_context
        active_c = active_c + self.state_ffn(active_c)

        # 计算核心
        history_states = [x0, x]

        for i, block in enumerate(self.ouro_blocks):
            block: OuroBlock
            if i > 0:
                stacked_history = torch.stack(history_states, dim=2)
                
                keys = self.attnres_k_norm(stacked_history) 
                values = stacked_history 
                q = self.attnres_queries[i - 1] 

                scores = torch.matmul(keys, q) / (self.embed_dim ** 0.5)
                alpha = F.softmax(scores, dim=-1)
                x = torch.sum(alpha.unsqueeze(-1) * values, dim=2)
                                                  
            x, out_list = block(x, active_c, lock_mem)
            history_states.extend(out_list)
            residual = x

        depth_attn_out = self.final_depth_attn(active_c, history_states)
        x = residual + depth_attn_out
        x = x + self.m_ffn(x)

        # 标准输出
        x = x + self.out_attn(self.out_norm(x))
        x = x + self.out_ffn(x)
        out = self.stm(x)

        # 状态更新
        if not lock_mem:
            new_c = self.stm._pending_c_state
            if self.l2_enabled:
                self._l2_state_next = new_c.detach()
            next_queue = torch.roll(self.c_state_queue.clone(), shifts=-1, dims=1)
            next_queue[:batch_size, -1, :] = new_c
            self.c_state_queue = next_queue
        
        return out

    def state_constraint_loss(self, n_probes: int = 8) -> torch.Tensor:
        """
        L2 状态约束损失（等效原理的随机化实现）。

        理论：L2 = E[ || dW - J ds ||^2 ]，其中 dW = Σ delta_mem，J = ∂W/∂s。
        随机化：对高斯探针 v，E[<x, v>^2] = ||x||^2，故

            L2 = E_v[ (<dW, v> - <J ds, v>)^2 ]
               = E_v[ (<dW, v> - ds · J^T v)^2 ]

        J^T v 用 torch.autograd.grad (VJP) 计算；n_probes 个探针通过
        is_grads_batched=True 在一次反向传播内完成，降方差不增成本。

        注意：梯度只流过 <dW, v> 项，<J ds, v> 作为冻结目标——
        这是稳定的一阶近似（避免对 Jacobian 的二阶求导）。
        """
        if not self.l2_enabled:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        s_t = self._l2_state_t
        s_next = self._l2_state_next
        if s_t is None or s_next is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        ds = (s_next - s_t.detach()).detach()  # [batch, embed_dim] 常数方向

        # 收集所有 mem 层的 dW
        dWs = []
        for block in self.ouro_blocks:
            for layer in block.ouro_layers:
                if layer.need_mem and layer._last_delta_mem is not None:
                    dWs.append(layer._last_delta_mem)
        if not dWs:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # n 个随机探针，[n, b, d, d]（is_grads_batched 要求 grad_outputs 前导为探针维）
        probes = [torch.randn(n_probes, *dw.shape, device=dw.device, dtype=dw.dtype) for dw in dWs]

        # 一次反向传播算出所有层的 VJP 批次: J_i^T v_i
        vjps = torch.autograd.grad(
            dWs, s_t, grad_outputs=probes,
            retain_graph=True, allow_unused=True, is_grads_batched=True,
        )

        loss = torch.tensor(0.0, device=s_t.device)
        n = 0
        for dw, probe, vjp in zip(dWs, probes, vjps):
            if vjp is None:
                continue
            # <dW, v> - ds · (J^T v)，每个探针一个标量
            residual = (dw.unsqueeze(0) * probe).sum(dim=(-3, -2, -1)) - \
                       (ds * vjp).sum(dim=(-2, -1))
            loss = loss + residual.pow(2).mean()
            n += 1

        if n == 0:
            return torch.tensor(0.0, device=s_t.device)
        return loss / n

        





