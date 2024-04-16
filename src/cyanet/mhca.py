import torch
from torch import nn
from torch.nn import Module


class MultiHeadLinear(Module):
    def __init__(self, d_in: int, heads: int, d_k: int, bias: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, heads * d_k, bias=bias)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x).view(*x.shape[:-1], self.heads, self.d_k)
        return x


class MultiHeadChannelAttention(Module):
    """Multi-Head Attention Module on Channels
    `(B, C, N) -> (B, N, heads, d_k) -> (B, C, N)`

    Args:
        d_in (int): input channels
        d_out (int): output channels
        num_heads (int): number of heads
        attn_drop (float): attention dropout rate
        bias (bool): whether to use bias in Q/K computation
    """

    def __init__(self, d_in: int, d_out: int, num_heads: int = 8,
                 bias: bool = True):
        super().__init__()
        assert d_out % num_heads == 0, \
            f'multi-head division err: d_out {d_out} % num_heads {num_heads} != 0'

        self.d_in = d_in
        self.d_out = d_out
        self.d_k = d_out // num_heads
        self.num_heads = num_heads

        self.q = MultiHeadLinear(d_in, num_heads, self.d_k, bias=bias)
        self.k = MultiHeadLinear(d_in, num_heads, self.d_k, bias=bias)
        self.v = MultiHeadLinear(d_in, num_heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor):
        B, C, *dims = x.shape
        x = x.view(B, -1, C)

        # Q, K are (B, H*W, heads, d_k)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attention = self.softmax(self.d_k ** -.5 *
                                 torch.einsum('bihd,bjhd->bhij', Q, K))

        out = torch.einsum('bhij,bjhd->bihd', attention, V)
        out = out.reshape(B, -1, C)
        out = self.fc(out)
        return out.reshape(B, C, *dims)
