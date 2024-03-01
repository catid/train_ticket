import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.init as init
import math

enable_bop = True

# Assuming BayesBiNN class is already defined as provided above
from optimizers.BayesBiNN import BayesBiNN

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

class MaskedPrngMatrix(nn.Module):
    def __init__(self, in_features, out_features, seed=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        #self.weight = nn.Parameter(torch.ones(out_features, in_features), requires_grad=True)
        #torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        self.mask = nn.Parameter(torch.ones(out_features, in_features), requires_grad=True)

    def forward(self, x):
        M = generate_weights(self.in_features, self.out_features, x.device, self.seed)
        #M = self.weight
        gate = (self.mask + 1.0) * 0.5 # TODO: Learn 0..1 directly
        return F.linear(x, gate * M)

class TicketLinear(nn.Module):
    def __init__(self, in_features, out_features, seed=1):
        super().__init__()
        self.masked_prng = MaskedPrngMatrix(in_features, out_features, seed=seed)

    def forward(self, x):
        return self.masked_prng(x)

# Step 1: Model Definition
class SimpleBinaryNet(nn.Module):
    def __init__(self):
        super(SimpleBinaryNet, self).__init__()
        if enable_bop:
            self.fc1 = TicketLinear(10, 1024)  # Example layer
        else:
            self.fc1 = nn.Linear(10, 1024)  # Example layer
        self.fc2 = nn.Linear(1024, 1)   # Output layer for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary output
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

def setup_optimizers(model):
    # Lists to hold parameters of TicketLinear layers and all other layers
    binary_params = []
    other_params = []
    
    # Scan the model for parameters
    for name, module in model.named_modules():
        if isinstance(module, MaskedPrngMatrix):
            print(f"XXX Found {len(list(module.children()))} parameters in MaskedPrngMatrix")
            binary_params.extend(list(module.parameters()))
        else:
            print(f"Found {len(list(module.parameters()))} parameters in {name}")
            # Assuming you want to exclude top-level nn.Module parameters that are not layers (e.g., nn.Linear, nn.Conv2d)
            if len(list(module.children())) == 0:  # This checks if the module is a leaf node (no children)
                other_params.extend(list(module.parameters()))

    # Define separate optimizers for each parameter group
    optimizer = BayesBiNN(binary_params, train_set_size=len(dataset), lr=3e-4, betas=0.9)
    optimizer_others = optim.AdamW(other_params, lr=0.01)

    return optimizer, optimizer_others

# Training Loop
def train(model, bop, optimizer, loss_function, dataloader):
    model.train()
    for epoch in range(200):  # Train for 5 epochs
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()

            if bop:
                bop.zero_grad()
            optimizer.zero_grad()
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass to calculate gradients

            # Closure function that returns the loss
            def closure():
                outputs = model(inputs)  # Forward pass
                loss = loss_function(outputs, targets)  # Compute loss
                return loss, outputs

            if bop:
                loss, _ = bop.step(closure)
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

if enable_bop:
    bop, optimizer = setup_optimizers(model)
else:
    bop, optimizer = None, optim.AdamW(model.parameters(), lr=0.01)

# Call the training function
train(model, bop, optimizer, loss_function, dataloader)

def print_model_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}, shape: {param.data.shape}\nWeights:\n{param.data}\n")

#print_model_weights(model)
