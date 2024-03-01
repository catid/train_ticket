import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

DISABLE_TICKET_LINEAR = False

def generate_weights(in_features, out_features, device='cuda', dtype=torch.float32, seed=42):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if out_features == in_features:
        dim = in_features
        v = torch.randn(dim, 1, generator=generator, device=device, dtype=dtype)
        alpha = -2.0 / dim
        return torch.eye(dim, device=device, dtype=dtype) + alpha * torch.mm(v, v.t())

    d_long = max(out_features, in_features)
    d_tiny = min(out_features, in_features)

    u = torch.randn(d_tiny, 1, generator=generator, device=device, dtype=dtype)
    v = torch.randn(d_long - d_tiny, 1, generator=generator, device=device, dtype=dtype)

    # from "Automatic Gradient Descent: Deep Learning without Hyperparameters"
    # https://arxiv.org/pdf/2304.05187.pdf
    scale = math.sqrt(out_features / in_features)

    I_n = torch.eye(d_tiny, device=device, dtype=dtype) * scale
    alpha = -2.0 * scale / d_long
    upper_block = I_n + alpha * u.mm(u.t())
    lower_block = alpha * v.mm(u.t())

    M = torch.cat((upper_block, lower_block), dim=0)

    if out_features < in_features:
        return torch.transpose(M, 0, 1)

    return M

class STEBinaryQuantizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input > 0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MaskedPrngMatrix(nn.Module):
    def __init__(self, in_features, out_features, seed=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        self.mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=True)

    def forward(self, x):
        M = generate_weights(self.in_features, self.out_features, x.device, x.dtype, self.seed)
        mask = STEBinaryQuantizeFunction.apply(self.mask)
        return F.linear(x, mask * M)

class TicketLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=1):
        super().__init__()

        if DISABLE_TICKET_LINEAR:
            self.proj = nn.Linear(in_features, out_features, bias=bias)
            return

        self.masked_prng = MaskedPrngMatrix(in_features, out_features, seed=seed)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        if DISABLE_TICKET_LINEAR:
            return self.proj(x)

        y = self.masked_prng(x)
        if self.bias is not None:
            y = y + self.bias
        return y
