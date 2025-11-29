import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int
    num_dims: int                       
    num_heads: int                      
    num_kv_heads: int                   
    num_layers: int                     
    ffn_hidden_dims: int                
    context_len: int                    
    
    # Architecture flags
    use_cache: bool = True
    use_flash: bool = False             
    attention_bias: bool = False        # Bias for Q, K, V
    attention_out_bias: bool = False    # Bias for Output Projection (False for Qwen)
    mlp_bias: bool = False              # Bias for MLP
    tie_weights: bool = False           

    rmsnorm_eps: float = 1e-6
    rope_theta: float = 1e6

    mask_token_id: int = 0            

def repeat_kv(vct: torch.Tensor, n_times: int):
    c_batch_size, c_context_len, num_kv_heads, c_dim = vct.shape
    if n_times == 1:
        return vct
    else:
        return (
            vct[:, :, :, None, :]
            .expand(c_batch_size, c_context_len, num_kv_heads, n_times, c_dim)
            .reshape(c_batch_size, c_context_len, num_kv_heads * n_times, c_dim)
        )

class Rotary(nn.Module):
    def __init__(self, config):
        super(Rotary, self).__init__()
        head_dim = config.num_dims // config.num_heads
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.seq_len_saved = None
        self.cos_saved = None
        self.sin_saved = None

    def forward(self, x, seq_dim=1):
        seq_len = x.size(seq_dim)
        # Only recompute the cosine and sine matrices if the sequence length has changed.
        if seq_len != self.seq_len_saved:
            self.seq_len_saved = seq_len
            pos = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # Compute the outer product between positions and inverse frequencies.
            freqs = torch.einsum("i,j->ij", pos, self.inv_freq) # (seq_len, inv_freq.shape[0])
            # Duplicate the freqs along the last dimension to create pairs.
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_saved = emb.cos()
            self.sin_saved = emb.sin()

        return self.cos_saved, self.sin_saved

class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.num_dims))
        self.eps = config.rmsnorm_eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_flash = config.use_flash

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads

        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.num_dims // self.num_heads

        self.wq = nn.Linear(config.num_dims, config.num_dims, bias=config.attention_bias)
        self.wk = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.wv = nn.Linear(config.num_dims, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        
        # Only init weight, bias is init by default if it exists
        nn.init.normal_(self.wq.weight, mean=0, std=1/math.sqrt(config.num_dims))
        nn.init.normal_(self.wk.weight, mean=0, std=1/math.sqrt(config.num_dims))
        nn.init.normal_(self.wv.weight, mean=0, std=1/math.sqrt(config.num_dims))
        
        self.wo = nn.Linear(config.num_dims, config.num_dims, bias=config.attention_out_bias)

    def rotate_half(self, x):
        half = x.shape[-1] // 2
        first_half, second_half  = x[..., :half], x[..., half:]
        return torch.cat([-second_half, first_half], dim=-1)

    def apply_rotary_pos(self, q, k, cos, sin):
        # q shape: [batch, heads, seq_len, head_dim]
        seq_len = q.shape[2]
        
        # Take the last 'seq_len' positions from cos/sin
        # Note: This logic assumes 'cos' contains [0...MaxSeq] and we slice [0...CurrentSeq]
        # or that we pass exactly what is needed.
        # Given Rotary implementation, it returns exactly `seq_len` size.
        curr_cos = cos[:seq_len, :]
        curr_sin = sin[:seq_len, :]
        
        # Reshape for broadcast: [batch, heads, seq, dim] vs [1, 1, seq, dim]
        curr_cos = curr_cos.unsqueeze(0).unsqueeze(0) 
        curr_sin = curr_sin.unsqueeze(0).unsqueeze(0)

        q_rot = q * curr_cos + self.rotate_half(q) * curr_sin
        k_rot = k * curr_cos + self.rotate_half(k) * curr_sin
        return q_rot, k_rot

    def forward(self, x, cos, sin, attn_mask=None):
        c_batch_size, c_context_len, c_dim = x.shape 

        # 1. Project
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 2. Reshape to [B, Heads, Seq, HeadDim]
        q = q.view(c_batch_size, c_context_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(c_batch_size, c_context_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE
        q, k = self.apply_rotary_pos(q, k, cos, sin)

        

        # 4. Expand GQA to match MHA for Attention
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_rep, dim=1)
            v = v.repeat_interleave(self.num_rep, dim=1)


        # 5. Normalize attn_mask
        # Expect attn_mask as bool [T, T] or [B, H, T, T], True=ALLOWED
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[None, None, :, :]   # [1, 1, T, T] -> broadcast

        # 6. Attention
        if self.use_flash:
             output = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=attn_mask)
        else:
            # Manual attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))     
            attention = F.softmax(scores, dim=-1)
            output = torch.matmul(attention, v)

        # 7. Output Projection
        output = output.transpose(1, 2).contiguous().view(c_batch_size, c_context_len, c_dim)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.ffn_hidden_dims
        use_bias = config.mlp_bias

        self.w1 = nn.Linear(config.num_dims, self.hidden_dim, bias=use_bias)
        self.w2 = nn.Linear(self.hidden_dim, config.num_dims, bias=use_bias)
        self.w3 = nn.Linear(config.num_dims, self.hidden_dim, bias=use_bias)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x)), None


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)

        self.ffn = FeedForward(config)

        self.norm_attention = RMSNorm(config)
        self.norm_ffn = RMSNorm(config)

    def forward(self, x, cos, sin, attn_mask):
        h = x + self.attention(self.norm_attention(x), cos, sin, attn_mask=attn_mask)
        out, _ = self.ffn(self.norm_ffn(h))
        return h + out, 0

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.rotary_emb = Rotary(config)
        self.tokens_embedding = nn.Embedding(self.vocab_size, config.num_dims)
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.num_layers)])
        self.norm = RMSNorm(config)
        self.ll_head = nn.Linear(config.num_dims, self.vocab_size, bias=False)
        
        if config.tie_weights:
            self.tokens_embedding.weight = self.ll_head.weight

    import torch

    def _right_shift(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Right-shift logits by 1 along the sequence dimension.
        """
        assert logits.dim() == 3, "logits must be (batch, seq_len, vocab_size)"
        bsz, seq_len, vocab_size = logits.shape

        # Dummy first position (won't be used in the loss if you never mask the BOS token)
        first = torch.zeros(bsz, 1, vocab_size,
                            device=logits.device,
                            dtype=logits.dtype)

        shifted = torch.cat([first, logits[:, :-1, :]], dim=1)
        return shifted


    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None,
                attn_mask: torch.Tensor = None, 
                masked_positions: torch.Tensor = None):

        _, seq_len = x.shape
        h = self.tokens_embedding(x)
        cos, sin = self.rotary_emb(h, seq_dim=1)

        for block in self.blocks:
            h, _ = block(h, cos, sin, attn_mask=attn_mask)
        
        h = self.norm(h)
        logits = self.ll_head(h) #[B, T, V]
        logits = self._right_shift(logits)
        
        loss = None
        if x0 is not None:
            if masked_positions is not None:
                loss = F.cross_entropy(logits[masked_positions].view(-1, logits.size(-1)),
                                                    x0[masked_positions].view(-1),
                                                    reduction="mean")
            else:
                raise ValueError("masked_positions in forward of Transformer is None")

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, steps: int = 50, temperature: float = 1.0):
        """
        generation for the Diffusion Language Model.
        """
        self.eval()
        B, T_prompt = idx.shape
        T_total = T_prompt + max_new_tokens

        x = torch.full((B, T_total), self.config.mask_token_id, device=idx.device, dtype=torch.long)
        x[:, :T_prompt] = idx
        

        for step in range(steps):
            progress = (step + 1) / steps

            mask_ratio = np.cos(progress * np.pi / 2)
            num_to_mask = int(max_new_tokens * mask_ratio)

            logits, _ = self.forward(x)
            
  
            gen_logits = logits[:, T_prompt:, :] 
            
            if temperature > 0:
                probs = torch.softmax(gen_logits / temperature, dim=-1)
                pred_ids = torch.multinomial(probs.view(-1, self.config.vocab_size), 1).view(B, max_new_tokens)
                pred_probs = torch.gather(probs, -1, pred_ids.unsqueeze(-1)).squeeze(-1)
            else:
                probs = torch.softmax(gen_logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)
                pred_probs = torch.max(probs, dim=-1).values


            x[:, T_prompt:] = pred_ids
            
            if step == steps - 1:
                break

            if num_to_mask > 0:

                _, mask_indices = torch.topk(pred_probs, k=num_to_mask, dim=1, largest=False)
 
                global_mask_indices = mask_indices + T_prompt
                
                x.scatter_(1, global_mask_indices, self.config.mask_token_id)

        return x
