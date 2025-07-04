from dataclasses import dataclass
import json
import math
import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WednesdayConfig:
    """
    Configuration for the Wednesday Mark2.
    """
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layer: int = 6
    n_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 2048
    d_ff: Optional[int] = None
    dropout: float = 0.1
    model_name: str = "WednesdayMark2"
    use_rope: bool = True
    pad_token_id: int = 50256


class Expert(nn.Module):
    """Single Expert network in the Mixture of Experts"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # SwiGLU gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(self.dropout(gate * up))


class MixtureOfExperts(nn.Module):
    """ Mixture of Experts (MoE) layer with top-k routing"""

    def __init__(self,
                 d_model: int,
                 n_experts: int,
                 top_k: int = 2,
                 d_ff: Optional[int] = None,
                 dropout: float = 0.1,
                 load_balance_loss_coef: float = 0.01):
        super().__init__()

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.load_balance_loss_coef = load_balance_loss_coef
        self.dropout = dropout

        if d_ff is None:
            d_ff = 4 * d_model

        # Router/gating network
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, dropout) for _ in range(n_experts)])

        # For load balancing
        self.register_buffer('expert_usage', torch.zeros(n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE layer.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Output tensor and load balance loss.
        """
        B, T, C = x.shape

        x_flat = x.view(-1, C)  #(batch_size * seq_len, d_model)

        # Compute router scores
        router_logits = self.router(x_flat)  #(batch_size * seq_len, n_experts)

        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits,
                                                 self.top_k,
                                                 dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)  # Normalize scores

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Route to experts
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)

            if expert_mask.any():
                expert_tokens = x_flat[expert_mask]
                expert_output = expert(expert_tokens)

                # Get the weights for this expert
                expert_weights = top_k_probs[expert_mask]
                expert_indices_local = top_k_indices[expert_mask]

                # Find which position in top_k this expert is for each token
                expert_position = (expert_indices_local == i).float()
                expert_weight = (expert_weights * expert_position).sum(
                    dim=-1, keepdim=True)

                # Add weighted expert output
                output[expert_mask] += expert_output * expert_weight

        # Calculate load balancing loss
        router_probs = F.softmax(router_logits, dim=-1)
        load_balance_loss = self._calculate_load_balance_loss(router_probs)

        return output.view(B, T, C), load_balance_loss

    def _calculate_load_balance_loss(
            self, router_probs: torch.Tensor) -> torch.Tensor:
        """Calculate load balancing loss to encourage equal expert usage"""
        # Average probability of routing to each expert
        expert_probs = router_probs.mean(dim=0)

        # Auxiliary loss to balance load
        aux_loss = self.n_experts * torch.sum(expert_probs * expert_probs)
        return self.load_balance_loss_coef * aux_loss


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better positional encoding"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        self.inv_freq: torch.Tensor
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self._cached_freqs = None
        self._cached_seq_len = 0

    def _get_freqs(self, seq_len: int, device: torch.device):
        if self._cached_freqs is None or seq_len > self._cached_seq_len:
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            self._cached_freqs = freqs
            self._cached_seq_len = seq_len
        return self._cached_freqs[:seq_len]

    def forward(self, x: torch.Tensor, seq_len: int):
        freqs = self._get_freqs(seq_len, x.device)
        return self.apply_rotary_pos_emb(x, freqs)

    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, freqs: torch.Tensor):
        # x: (B, n_heads, T, d_k)
        # freqs: (T, d_k//2)
        x1, x2 = x[..., 0::2], x[..., 1::2]  # both (B, n_heads, T, d_k//2)
        cos_freqs = freqs.cos()[None, None, :, :]  # (1, 1, T, d_k//2)
        sin_freqs = freqs.sin()[None, None, :, :]  # (1, 1, T, d_k//2)

        x_out = torch.zeros_like(x)
        x_out[..., 0::2] = x1 * cos_freqs - x2 * sin_freqs
        x_out[..., 1::2] = x1 * sin_freqs + x2 * cos_freqs
        return x_out


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE"""

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.use_rope = use_rope

        self.attn = nn.Linear(d_model, 3 * d_model, bias=False)
        # Alternative projection method (commented out)
        # self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.d_k)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        # Linear projections
        qkv = self.attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Alternative projection method (commented out)
        # q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope:
            q = self.rope(q, T)
            k = self.rope(k, T)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class WM2TransformerBlock(nn.Module):
    """Transformer block with MoE FFN"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_experts: int,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_rope: bool = True,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout, use_rope)
        self.moe = MixtureOfExperts(d_model, n_experts, top_k, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Multi-head attention
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Mixture of Experts
        moe_output, balance_loss = self.moe(self.norm2(x))
        x = x + self.dropout(moe_output)

        return x, balance_loss


class Wednesday(nn.Module):
    """Wednesday Mark 2: GPT model with Mixture of Experts"""

    def __init__(self, config: WednesdayConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layer = config.n_layer
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.d_ff = config.d_ff if config.d_ff is not None else 4 * config.d_model
        self.dropout = config.dropout
        self.pad_token_id = config.pad_token_id
        self.max_seq_len = config.max_seq_len

        # Initialize embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.max_seq_len,
                                                   config.d_model)
        self.use_rope = config.use_rope

        self.blocks = nn.ModuleList([
            WM2TransformerBlock(self.d_model, self.n_heads, self.n_experts,
                                self.top_k, self.d_ff, self.dropout,
                                self.use_rope) for _ in range(self.n_layer)
        ])

        # Output
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def create_causal_mask(self, seq_len: int,
                           device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        B, T = x.shape
        device = x.device

        # Embeddings
        tok_emb = self.token_embedding(x)

        # Position embeddings (if not using RoPE)
        if not self.use_rope:
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.position_embedding(pos_ids)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        # Create causal mask
        causal_mask = self.create_causal_mask(T, device)

        # Combine with attention mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask * mask

        # Pass through transformer blocks
        total_load_balance_loss = 0.0
        for block in self.blocks:
            x, balance_loss = block(x, causal_mask)
            total_load_balance_loss += balance_loss

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        # Calculate loss if labels provided
        loss: Optional[float] = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            # Add load balancing loss only if loss is not None
            if loss is not None:
                loss = loss + total_load_balance_loss / self.n_layer

        return {
            'logits': logits,
            'loss': loss,
            'load_balance_loss': total_load_balance_loss / self.n_layer
        }

    def generate(self,
                 input_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 pad_token_id: Optional[int] = None) -> torch.Tensor:
        """Generate text using the model"""
        self.eval()

        if pad_token_id is None:
            pad_token_id = self.pad_token_id

        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self(input_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits, top_k)
                    next_token_logits = torch.full_like(
                        next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits,
                                                              dim=-1),
                                                    dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[
                        ..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if we hit max length
                if input_ids.shape[1] >= self.max_seq_len:
                    break

        return input_ids

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving/loading"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layer': self.n_layer,
            'n_experts': self.n_experts,
            'top_k': self.top_k,
            'max_seq_len': self.max_seq_len,
            'use_rope': self.use_rope,
            'pad_token_id': self.pad_token_id
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create model from configuration"""
        return cls(**config)

    def save_pretrained(self, save_directory: str):
        """Save model weights and config"""
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f, indent=2)

        # Save model weights
        model_path = os.path.join(save_directory, f'wednesdaym2.bin')
        torch.save(self.state_dict(), model_path)

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_directory: str, device: str = 'cpu'):
        """Load model from saved weights and config"""
        # Load config
        config_path = os.path.join(model_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create model
        model = cls.from_config(config)

        # Load weights
        model_path = os.path.join(model_directory, f'wednesdaym2.bin')
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        print(f"Model loaded from {model_directory}")
        return model

    def count_parameters(self, non_embedding=True) -> int:
        """Count trainable parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            n_params -= sum(p.numel()
                            for p in self.token_embedding.parameters()
                            if p.requires_grad)
        return n_params
