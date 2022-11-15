import copy
from typing import Optional, Any
import math
import torch as t
from torch import nn,Tensor
import torch.nn.functional as F
from torch.nn.functional import *
from torch.nn.init import constant_
from torch.nn import Module
from torch.nn.init import xavier_normal_
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.parameter import Parameter

from typing import List, Optional, Tuple

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)




class gap(nn.Module):

    def __init__(self):
        super(gap, self).__init__()
        self.chanel_in = 256
        self.conv1 = nn.Sequential(
               nn.ConvTranspose2d(256*2, 256,  kernel_size=1, stride=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
                )
        in_dim=256
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1=nn.Conv2d(in_dim, in_dim // 8, 1, bias=False)
        self.linear2=nn.Conv2d(in_dim // 8, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(t.zeros(1))
        self.activation=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout()
        for modules in [self.conv1]:
            for l in modules.modules():
               if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)
    def forward(self,x,y):
        s,b,c=y.size()
       
        w=int(pow(s,0.5))
        y=y.permute(1,2,0).view(b,c,w,w)
        ww=self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y)))))
        x=x.permute(1,2,0).view(b,c,22,22)
        m=x+self.gamma*ww*x
        m=m.view(b,c,-1).permute(2, 0, 1)
        return m


# class InnerAttention(Module):
#     __constants__ = ['batch_first']
#
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(InnerAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#
#         self.q_proj_weight = Parameter(t.empty((embed_dim, embed_dim), **factory_kwargs))
#         self.k_proj_weight = Parameter(t.empty((embed_dim, self.kdim), **factory_kwargs))
#
#         if bias:
#             self.q_proj_bias = Parameter(t.empty(embed_dim, **factory_kwargs))
#             self.k_proj_bias = Parameter(t.empty(embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('q_proj_bias', None)
#             self.register_parameter('k_proj_bias', None)
#         self.out_proj = Linear(self.vdim, self.vdim, bias=bias)
#
#         self.add_zero_attn = add_zero_attn
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         xavier_uniform_(self.q_proj_weight)
#         xavier_uniform_(self.k_proj_weight)
#
#         if self.out_proj.bias is not None:
#             constant_(self.q_proj_bias, 0.)
#             constant_(self.k_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#
#     def _in_projection(
#             self,
#             q: Tensor,
#             k: Tensor,
#             w_q: Tensor,
#             w_k: Tensor,
#             b_q: Optional[Tensor] = None,
#             b_k: Optional[Tensor] = None
#     ) -> Tuple[Tensor, Tensor]:
#         return linear(q, w_q, b_q), linear(k, w_k, b_k)
#
#     def inner_scaled_dot_product_attention(
#             self,
#             q: Tensor,
#             k: Tensor,
#             v: Tensor,
#             attn_mask: Optional[Tensor] = None,
#             dropout_p: float = 0.0
#     ) -> Tuple[Tensor, Tensor]:
#         B, Nt, E = q.shape
#         q = q / math.sqrt(E)
#         # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
#         attn = t.bmm(q, k.transpose(-2, -1))
#         if attn_mask is not None:
#             attn += attn_mask
#         attn = softmax(attn, dim=-1)
#         if dropout_p > 0.0:
#             attn = dropout(attn, p=dropout_p)
#         # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
#         output = t.bmm(attn, v)
#         return output, attn
#
#     def inner_attention_forward(
#             self,
#             query: Tensor,
#             key: Tensor,
#             value: Tensor,
#             embed_dim_to_check: int,
#             num_heads: int,
#             q_proj_bias: Optional[Tensor],
#             k_proj_bias: Optional[Tensor],
#             add_zero_attn: bool,
#             dropout_p: float,
#             out_proj_weight: Tensor,
#             out_proj_bias: Optional[Tensor],
#             training: bool = True,
#             key_padding_mask: Optional[Tensor] = None,
#             need_weights: bool = True,
#             attn_mask: Optional[Tensor] = None,
#             q_proj_weight: Optional[Tensor] = None,
#             k_proj_weight: Optional[Tensor] = None,
#             static_k: Optional[Tensor] = None,
#             static_v: Optional[Tensor] = None
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         # Set up shape vars
#         tgt_len, bsz, embed_dim = query.shape
#         src_len, _, _ = key.shape
#         if isinstance(embed_dim, t.Tensor):
#             # Embed_dim can be a tensor when JIT tracing
#             head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
#         else:
#             head_dim = embed_dim // num_heads
#
#         # Compute in-projection
#         if q_proj_bias is None:
#             b_q = b_k = None
#         else:
#             b_q, b_k = q_proj_bias, k_proj_bias
#         q, k = self._in_projection(query, key, q_proj_weight, k_proj_weight, b_q, b_k)
#         v = value
#
#         # Prep attention mask
#         if attn_mask is not None:
#             if attn_mask.dtype == t.uint8:
#                 attn_mask = attn_mask.to(t.bool)
#             # Ensure attn_mask's dim is 3
#             if attn_mask.dim() == 2:
#                 attn_mask = attn_mask.unsqueeze(0)
#
#         # Prep key padding mask
#         if key_padding_mask is not None and key_padding_mask.dtype == t.uint8:
#             key_padding_mask = key_padding_mask.to(t.bool)
#
#         # Reshape q, k, v for multihead attention and make em batch first
#         q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#         if static_k is None:
#             k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#         else:
#             k = static_k
#         if static_v is None:
#             v = v.contiguous().view(-1, bsz * num_heads, self.vdim // self.num_heads).transpose(0, 1)
#         else:
#             v = static_v
#
#         # Add zero attention along batch dimension
#         if add_zero_attn:
#             zero_attn_shape = (bsz * num_heads, 1, head_dim)
#             k = t.cat([k, t.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
#             v = t.cat([v, t.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
#             if attn_mask is not None:
#                 attn_mask = pad(attn_mask, (0, 1))
#             if key_padding_mask is not None:
#                 key_padding_mask = pad(key_padding_mask, (0, 1))
#
#         # Update source sequence length after adjustments
#         src_len = k.size(1)
#
#         # Merge key padding and attention masks
#         if key_padding_mask is not None:
#             key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
#                 expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
#             if attn_mask is None:
#                 attn_mask = key_padding_mask
#             elif attn_mask.dtype == t.bool:
#                 attn_mask = attn_mask.logical_or(key_padding_mask)
#             else:
#                 attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
#
#         # Convert mask to float
#         if attn_mask is not None and attn_mask.dtype == t.bool:
#             new_attn_mask = t.zeros_like(attn_mask, dtype=t.float)
#             new_attn_mask.masked_fill_(attn_mask, float('-inf'))
#             attn_mask = new_attn_mask
#
#         # Adjust dropout probability
#         if not training:
#             dropout_p = 0.0
#
#         # Calculate attention and out projection
#         attn_output, attn_output_weights = self.inner_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
#         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.vdim)
#         attn_output = attn_output + linear(attn_output, out_proj_weight, out_proj_bias)
#
#         if need_weights:
#             # Average attention weights over heads
#             attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#             return attn_output, attn_output_weights.sum(dim=1) / num_heads
#         else:
#             return attn_output, None
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
#         if self.batch_first:
#             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
#
#         attn_output, attn_output_weights = self.inner_attention_forward(
#             query, key, value, self.embed_dim, self.num_heads, self.q_proj_bias, self.k_proj_bias,
#             self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias,
#             training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights,
#             attn_mask=attn_mask, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight)
#
#         if self.batch_first:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights


# class CorrAttention(Module):
#
#     def __init__(self, num_heads, dropout, match_dim, feat_size):
#         super(CorrAttention, self).__init__()
#         self.match_dim = match_dim
#         self.feat_size = feat_size
#         self.corr_proj = nn.Linear(self.feat_size, self.match_dim)
#         self.corr_attn = InnerAttention(self.match_dim, 1, dropout=dropout, vdim=self.feat_size)
#         self.feat_norm1 = nn.LayerNorm(self.match_dim)
#         self.feat_norm2 = nn.LayerNorm(self.feat_size)
#         self.dropout = nn.Dropout(dropout)
#         self.num_heads = num_heads
#
#     def forward(self, corr_map, pos_emb):
#         batch_size = pos_emb.shape[1]
#         pos_emb = t.repeat_interleave(pos_emb, self.num_heads, dim=1).transpose(0, -1).reshape(self.match_dim, -1,
#                                                                                                    self.feat_size).transpose(
#             0, -1)
#         corr_map = corr_map.transpose(0, 1).reshape(self.feat_size, -1, self.feat_size)
#         corr_map = corr_map.transpose(0, -1)  # From the perspective of keys
#         q = k = self.feat_norm1(self.corr_proj(corr_map)) + pos_emb
#         corr_map1 = self.corr_attn(q, k, value=self.feat_norm2(corr_map))[0]
#         corr_map = self.dropout(corr_map1)
#         corr_map = corr_map.transpose(0, -1)
#         corr_map = corr_map.reshape(self.feat_size, self.num_heads * batch_size, -1).transpose(0, 1)
#         return corr_map
#



# class AiAModule(Module):
#     __constants__ = ['batch_first']
#     bias_k: Optional[t.Tensor]
#     bias_v: Optional[t.Tensor]
#
#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None, use_AiA=True, match_dim=64,
#                  feat_size=400) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(AiAModule, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
#
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#
#         self.use_AiA = use_AiA
#         if self.use_AiA:
#             self.inner_attn = CorrAttention(num_heads, dropout, match_dim, feat_size)
#
#         if self._qkv_same_embed_dim is False:
#             self.q_proj_weight = Parameter(t.empty((embed_dim, embed_dim), **factory_kwargs))
#             self.k_proj_weight = Parameter(t.empty((embed_dim, self.kdim), **factory_kwargs))
#             self.v_proj_weight = Parameter(t.empty((embed_dim, self.vdim), **factory_kwargs))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = Parameter(t.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)
#
#         if bias:
#             self.in_proj_bias = Parameter(t.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
#
#         if add_bias_kv:
#             self.bias_k = Parameter(t.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(t.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None
#
#         self.add_zero_attn = add_zero_attn
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)
#
#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)
#
#     def __setstate__(self, state):
#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if '_qkv_same_embed_dim' not in state:
#             state['_qkv_same_embed_dim'] = True
#
#         super(AiAModule, self).__setstate__(state)
#
#     def _in_projection_packed(
#             self,
#             q: Tensor,
#             k: Tensor,
#             v: Tensor,
#             w: Tensor,
#             b: Optional[Tensor] = None,
#     ) -> List[Tensor]:
#         E = q.size(-1)
#         if k is v:
#             if q is k:
#                 # Self-attention
#                 return linear(q, w, b).chunk(3, dim=-1)
#             else:
#                 # Encoder-decoder attention
#                 w_q, w_kv = w.split([E, E * 2])
#                 if b is None:
#                     b_q = b_kv = None
#                 else:
#                     b_q, b_kv = b.split([E, E * 2])
#                 return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
#         else:
#             w_q, w_k, w_v = w.chunk(3)
#             if b is None:
#                 b_q = b_k = b_v = None
#             else:
#                 b_q, b_k, b_v = b.chunk(3)
#             return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
#
#     def _in_projection(
#             self,
#             q: Tensor,
#             k: Tensor,
#             v: Tensor,
#             w_q: Tensor,
#             w_k: Tensor,
#             w_v: Tensor,
#             b_q: Optional[Tensor] = None,
#             b_k: Optional[Tensor] = None,
#             b_v: Optional[Tensor] = None,
#     ) -> Tuple[Tensor, Tensor, Tensor]:
#         return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
#
#     def aia_scaled_dot_product_attention(
#             self,
#             q: Tensor,
#             k: Tensor,
#             v: Tensor,
#             attn_mask: Optional[Tensor] = None,
#             dropout_p: float = 0.0,
#             pos_emb=None
#     ) -> Tuple[Tensor, Tensor]:
#         B, Nt, E = q.shape
#         q = q / math.sqrt(E)
#         # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
#         attn = t.bmm(q, k.transpose(-2, -1))
#
#         if self.use_AiA:
#             corr_map = attn
#             corr_map = self.inner_attn(corr_map, pos_emb)
#             attn = attn + corr_map
#
#         # We comment out the following two lines since applying mask to the padding regions doesn't have obvious influence on the performance
#         # You can use it if you like (by removing the comment), but the model should be retrained with the padding mask
#         # if attn_mask is not None:
#         #     attn += attn_mask
#
#         attn = softmax(attn, dim=-1)
#         if dropout_p > 0.0:
#             attn = dropout(attn, p=dropout_p)
#         # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
#         output = t.bmm(attn, v)
#         return output, attn
#
#     def aia_attention_forward(
#             self,
#             query: Tensor,
#             key: Tensor,
#             value: Tensor,
#             embed_dim_to_check: int,
#             num_heads: int,
#             in_proj_weight: Tensor,
#             in_proj_bias: Optional[Tensor],
#             bias_k: Optional[Tensor],
#             bias_v: Optional[Tensor],
#             add_zero_attn: bool,
#             dropout_p: float,
#             out_proj_weight: Tensor,
#             out_proj_bias: Optional[Tensor],
#             training: bool = True,
#             key_padding_mask: Optional[Tensor] = None,
#             need_weights: bool = True,
#             attn_mask: Optional[Tensor] = None,
#             use_separate_proj_weight: bool = False,
#             q_proj_weight: Optional[Tensor] = None,
#             k_proj_weight: Optional[Tensor] = None,
#             v_proj_weight: Optional[Tensor] = None,
#             static_k: Optional[Tensor] = None,
#             static_v: Optional[Tensor] = None,
#             pos_emb=None
#     ) -> Tuple[Tensor, Optional[Tensor]]:
#         # Set up shape vars
#         tgt_len, bsz, embed_dim = query.shape
#         src_len, _, _ = key.shape
#         if isinstance(embed_dim, t.Tensor):
#             # Embed_dim can be a tensor when JIT tracing
#             head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
#         else:
#             head_dim = embed_dim // num_heads
#
#         # Compute in-projection
#         if not use_separate_proj_weight:
#             q, k, v = self._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
#         else:
#             if in_proj_bias is None:
#                 b_q = b_k = b_v = None
#             else:
#                 b_q, b_k, b_v = in_proj_bias.chunk(3)
#             q, k, v = self._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)
#
#         # Prep attention mask
#         if attn_mask is not None:
#             if attn_mask.dtype == t.uint8:
#                 attn_mask = attn_mask.to(t.bool)
#             # Ensure attn_mask's dim is 3
#             if attn_mask.dim() == 2:
#                 attn_mask = attn_mask.unsqueeze(0)
#
#         # Prep key padding mask
#         if key_padding_mask is not None and key_padding_mask.dtype == t.uint8:
#             key_padding_mask = key_padding_mask.to(t.bool)
#
#         # Add bias along batch dimension
#         if bias_k is not None and bias_v is not None:
#             k = t.cat([k, bias_k.repeat(1, bsz, 1)])
#             v = t.cat([v, bias_v.repeat(1, bsz, 1)])
#             if attn_mask is not None:
#                 attn_mask = pad(attn_mask, (0, 1))
#             if key_padding_mask is not None:
#                 key_padding_mask = pad(key_padding_mask, (0, 1))
#         else:
#             assert bias_k is None
#             assert bias_v is None
#
#         # Reshape q, k, v for multihead attention and make em batch first
#         q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#         if static_k is None:
#             k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#         else:
#             k = static_k
#         if static_v is None:
#             v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#         else:
#             v = static_v
#
#         # Add zero attention along batch dimension
#         if add_zero_attn:
#             zero_attn_shape = (bsz * num_heads, 1, head_dim)
#             k = t.cat([k, t.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
#             v = t.cat([v, t.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
#             if attn_mask is not None:
#                 attn_mask = pad(attn_mask, (0, 1))
#             if key_padding_mask is not None:
#                 key_padding_mask = pad(key_padding_mask, (0, 1))
#
#         # Update source sequence length after adjustments
#         src_len = k.size(1)
#
#         # Merge key padding and attention masks
#         if key_padding_mask is not None:
#             key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
#                 expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
#             if attn_mask is None:
#                 attn_mask = key_padding_mask
#             elif attn_mask.dtype == t.bool:
#                 attn_mask = attn_mask.logical_or(key_padding_mask)
#             else:
#                 attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
#
#         # Convert mask to float
#         if attn_mask is not None and attn_mask.dtype == t.bool:
#             new_attn_mask = t.zeros_like(attn_mask, dtype=t.float)
#             new_attn_mask.masked_fill_(attn_mask, float('-inf'))
#             attn_mask = new_attn_mask
#
#         # Adjust dropout probability
#         if not training:
#             dropout_p = 0.0
#
#         # Calculate attention and out projection
#         attn_output, attn_output_weights = self.aia_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p,
#                                                                                  pos_emb)
#         attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#         attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
#
#         if need_weights:
#             # Average attention weights over heads
#             attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#             return attn_output, attn_output_weights.sum(dim=1) / num_heads
#         else:
#             return attn_output, None
#
#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[Tensor] = None, pos_emb=None) -> Tuple[
#         Tensor, Optional[Tensor]]:
#         if self.batch_first:
#             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
#
#         if not self._qkv_same_embed_dim:
#             attn_output, attn_output_weights = self.aia_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, use_separate_proj_weight=True,
#                 q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
#                 v_proj_weight=self.v_proj_weight, pos_emb=pos_emb)
#         else:
#             attn_output, attn_output_weights = self.aia_attention_forward(
#                 query, key, value, self.embed_dim, self.num_heads,
#                 self.in_proj_weight, self.in_proj_bias,
#                 self.bias_k, self.bias_v, self.add_zero_attn,
#                 self.dropout, self.out_proj.weight, self.out_proj.bias,
#                 training=self.training,
#                 key_padding_mask=key_padding_mask, need_weights=need_weights,
#                 attn_mask=attn_mask, pos_emb=pos_emb)
#         if self.batch_first:
#             return attn_output.transpose(1, 0), attn_output_weights
#         else:
#             return attn_output, attn_output_weights
#
#
#
#

















class Transformer(Module):


    def __init__(self, d_model: int = 512, nhead: int = 4, num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1, dim_feedforward: int = 1024, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation) #d_model = 256 | 8 2048 0.1 relu
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, srcT: Tensor , srcS: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
       
       
        # print("srcT.shape:",srcT.shape)
        # print("srcS.shape:", srcS.shape)
        memoryT = self.encoder(srcT)
        # print("memoryT.shape:", memoryT.shape)
        memoryS = self.encoder(srcS)
        # print("memoryS.shape:", memoryS.shape)
        memory = self.decoder(memoryT, memoryS)
        # print("memory.shape:", memory.shape)

        
        return memory

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (t.triu(t.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)








class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, srcc: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = srcc

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.scr_attn = RMA(d_model, nhead)
        #self.scr_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.scr_attn = AiAModule(d_model, nhead, dropout=dropout, use_AiA=True, match_dim=64, feat_size=400)
        #self.high_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.gap=gap()
        # Implementation of Feedforward model
    
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        #self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:


        src2 = self.scr_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        srcc2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        srcc = src + self.dropout2(srcc2)
        srcc = self.norm2(srcc)

        return srcc

class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        #self.scr_attn = AiAModule(d_model, nhead, dropout=dropout, use_AiA=True, match_dim=64, feat_size=400)

        #self.scr_attention = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.scr_attn = RMA(d_model, nhead)
        #self.mask = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        #self.gap=gap()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        #self.linear3 = nn.Linear(d_model, dim_feedforward)
        #self.linear4 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        #self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        #self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        #self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        #self.dropout4 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        #self.activation2 = _get_activation_fn(activation)
       

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)
   
    def forward(self, srcc: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        srcc2 = self.scr_attn(memory, srcc, srcc)[0]
        srcc = memory + self.dropout1(srcc2)
        srcc = self.norm1(srcc)
        srcc2 = self.linear2(self.dropout(self.activation(self.linear1(srcc))))
        srcc = srcc + self.dropout3(srcc2)
        srcc = self.norm3(srcc)

        return srcc


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransNonlinear(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class RMA(nn.Module):
    def __init__(self, feature_dim=256, n_head=8, key_feature_dim=1024,
                 extra_nonlinear=True):
        super(RMA, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.extra_nonlinear = nn.ModuleList()
        self.extra_nonlinear_final = TransNonlinear(key_feature_dim, feature_dim)
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
            if extra_nonlinear:
                self.extra_nonlinear.append(TransNonlinear(feature_dim, key_feature_dim))
            else:
                self.extra_nonlinear = None
        self.trans_conv_out = nn.Linear(key_feature_dim, feature_dim, bias=False)
    def forward(self, query=None, key=None, value=None,  #query/key/value: b c w h
                ):
        """
        query : #pixel x batch x dim
        """
        isFirst = True
        for N in range(self.Nh):
            #print("N:",N)
            if(isFirst):
                concat = self.head[N](query, key, value)
                if self.extra_nonlinear:
                    concat = self.extra_nonlinear[N](concat)
                    #print("concat.shape",concat.shape)
                isFirst = False

            elif N == self.Nh-1:
                #print("N:", N)
                #print("tmp.shape:", tmp.shape)
                tmp = self.head[N](query, key, value)
                #print("tmp.shape:",tmp.shape)
                if self.extra_nonlinear:
                    tmp = self.extra_nonlinear[N](tmp)
                concat = t.cat((concat, tmp), -1)
                #print("concat.shape", concat.shape)

            else:
                tmp = self.head[N](query, key, value)
                #print("tmp_middle:",tmp.shape)
                if self.extra_nonlinear:
                    tmp = self.extra_nonlinear[N](tmp)
                concat = t.cat((concat, tmp), -1)
                #print("concat.shape", concat.shape)

        output = concat
        output = self.trans_conv_out(output)
        #print("output.shape:",output.shape)
        return output


class RelationUnit(nn.Module):
    def __init__(self, feature_dim=256, key_feature_dim=1024):
        super(RelationUnit, self).__init__()
        self.temp = 1
        self.WK = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WQ = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WV = nn.Linear(feature_dim, feature_dim, bias=False)
        self.after_norm = nn.BatchNorm2d(feature_dim)
        self.trans_conv = nn.Linear(feature_dim, feature_dim, bias=False)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None, mask=None):
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0) # Batch, Dim, Len_1

        w_q = self.WQ(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2) # Batch, Len_2, Dim

        dot_prod = t.bmm(w_q, w_k) # Batch, Len_2, Len_1
        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == 0, -1e9)
        affinity = F.softmax(dot_prod * self.temp, dim=-1)
        affinity = affinity / (1e-9 + affinity.sum(dim=1, keepdim=True))

        w_v = self.WV(value)
        w_v = w_v.permute(1,0,2) # Batch, Len_1, Dim
        output = t.bmm(affinity, w_v) # Batch, Len_2, Dim
        output = output.permute(1,0,2)

        output = self.trans_conv(query - output)

        return F.relu(output)