import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.init as init
import math

dim = 2048

def generate_weights(in_features, out_features, device='cuda', seed=42):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if out_features == in_features:
        dim = in_features
        v = torch.randn(dim, 1, generator=generator, device=device)
        alpha = -2.0 / dim
        return torch.eye(dim, device=device) + alpha * torch.mm(v, v.t())

    d_long = max(out_features, in_features)
    d_tiny = min(out_features, in_features)

    u = torch.randn(d_tiny, 1, generator=generator, device=device)
    v = torch.randn(d_long - d_tiny, 1, generator=generator, device=device)

    # from "Automatic Gradient Descent: Deep Learning without Hyperparameters"
    # https://arxiv.org/pdf/2304.05187.pdf
    scale = math.sqrt(out_features / in_features)

    I_n = torch.eye(d_tiny, device=device) * scale
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
        M = generate_weights(self.in_features, self.out_features, x.device, self.seed)
        mask = STEBinaryQuantizeFunction.apply(self.mask)
        return F.linear(x, mask * M)

class TicketLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=1):
        super().__init__()
        self.masked_prng = MaskedPrngMatrix(in_features, out_features, seed=seed)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        y = self.masked_prng(x)
        if self.bias is not None:
            y = y + self.bias
        return y


# Step 1: Model Definition
class SimpleBinaryNet(nn.Module):
    def __init__(self):
        super(SimpleBinaryNet, self).__init__()
        self.fc1 = nn.Linear(10, dim)
        self.fc2 = TicketLinear(dim, dim)
        self.fc3 = nn.Linear(dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary output
        return x

# Step 2: Create a synthetic dataset
inputs = torch.rand(1000, 10, device='cuda')  # 100 samples, 10 features each
targets = torch.randint(0, 2, (1000, 1), device='cuda').float()  # Binary targets

# Create a DataLoader
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Step 3: Initialize model, optimizer, and loss function
model = SimpleBinaryNet().cuda()
loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification

# Training Loop
def train(model, optimizer, loss_function, dataloader):
    model.train()
    for epoch in range(500):
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass to calculate gradients

            optimizer.step()
    model.eval()
    avg_loss = 0.0
    count = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)  # Forward pass
        loss = loss_function(outputs, targets)  # Compute loss
        avg_loss += loss.item()
        count += 1
    print(f'avg_loss: {avg_loss / count}')

optimizer = optim.AdamW(model.parameters(), lr=0.01)

# Call the training function
train(model, optimizer, loss_function, dataloader)

def print_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}, shape: {param.data.shape}\nWeights:\n{param.data}\n")

print_model_weights(model)
