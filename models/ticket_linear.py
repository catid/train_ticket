import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class CandyCaneDiagonal(nn.Module):
    def __init__(self, rows, cols, shift=0):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.shift = shift  # Shift amount for the diagonal
        self.values = nn.Parameter(torch.zeros(max(rows, cols)))

    @staticmethod
    def calculate_candy_cane_indices(rows, cols, shift, device):
        max_dim = max(rows, cols)
        stride = cols + 1 if rows >= cols else cols * (rows - 1) + 1

        # Adjust starting index based on shift
        start_index = max(0, shift * (cols + 1 if shift > 0 else 1))
        
        diag_indices = torch.arange(start_index, min(rows, cols) * stride + start_index, stride, device=device) % (rows * cols)

        if rows < cols:
            extra_stride = (rows * cols) // min(rows, cols) + 1
            extra_indices_start = diag_indices[-1] + extra_stride if diag_indices.numel() > 0 else start_index
            extra_indices = torch.arange(extra_indices_start, max_dim * stride + start_index, stride, device=device) % (rows * cols)
            indices = torch.cat((diag_indices, extra_indices), dim=0)
        else:
            indices = diag_indices

        # Ensure indices are within bounds
        indices = indices[indices < rows * cols]

        return indices.long()

    def forward(self, x):
        indices = self.calculate_candy_cane_indices(self.rows, self.cols, self.shift, device=x.device)

        values = self.values.clone()
        if len(indices) > len(self.values):
            values = torch.nn.functional.pad(values, (0, len(indices) - len(self.values)))

        matrix = torch.sparse_coo_tensor(indices=indices.unsqueeze(0),
                                         values=values,
                                         size=(self.rows * self.cols,)).to_dense()
        return x + matrix.view(self.rows, self.cols)

class TicketLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        init.kaiming_uniform_(self.weight0, a=math.sqrt(5.0))

        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5.0))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=False)
            alpha = 1.0 / math.sqrt(in_features)
            init.uniform_(self.bias, -alpha, alpha)
        else:
            self.bias = None

        self.gate0 = nn.Parameter(torch.ones(out_features, in_features), requires_grad=True)
        self.gate1 = nn.Parameter(torch.ones(out_features, in_features), requires_grad=True)

        self.diag0 = CandyCaneDiagonal(out_features, in_features, shift=0)
        self.diag1 = CandyCaneDiagonal(out_features, in_features, shift=1)

    def forward(self, x):
        g = torch.sigmoid(self.gate0 * self.gate0) * self.weight0 + torch.sigmoid(self.gate1 * self.gate1) * self.weight1

        g = self.diag0(g)
        g = self.diag1(g)

        return F.linear(x, g, self.bias)
