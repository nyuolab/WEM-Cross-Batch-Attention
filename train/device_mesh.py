from torch.distributed.device_mesh import init_device_mesh
import torch.nn as nn
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from torch.distributed.nn.functional import all_gather, broadcast, all_to_all, scatter

hidden_dimension = 768
width = 2048
length = 4096

world_size = int(os.environ["WORLD_SIZE"])
tp_size = 4
dp_size = world_size//tp_size
# dist.init_process_group('nccl')
device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("replicate", "tp"))


tp_mesh = device_mesh["tp"]
dp_mesh = device_mesh["replicate"]

tp_group = device_mesh.get_group(mesh_dim="tp")
dp_group = device_mesh.get_group(mesh_dim="replicate")

dp_rank = dp_mesh.get_local_rank()
tp_rank = tp_mesh.get_local_rank()




def scatter_tensor(x, world_size, dim, group, src):
    x_list = list(torch.chunk(x, world_size, dim=dim))
    output = torch.empty_like(x_list[0])
    if src != 0:
        x_list = None
    dist.scatter(tensor=output, scatter_list=x_list, group_src=0, group=group)
    return output

class StructuredGridDataset(torch.utils.data.Dataset):
    def __init__(self, height=256, width=256, hidden_dim=64, noise_std=0.1):
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

    def __len__(self):
        return 10_000_000  # arbitrarily large

    def __getitem__(self, idx):
        X = torch.randn(self.height, self.width, self.hidden_dim)
        y = (X.sum(dim=-1) > 0).float()  # [H, W]
        if self.noise_std > 0:
            X += torch.randn_like(X) * self.noise_std
        return X, y

def transpose_shard_2d_ddp(input_tensor, group, cut_dim=1, cat_dim=0):
    """
    Args:
        input_tensor: [X//n, Y] local shard on each device
        mesh: DeviceMesh over which to communicate (e.g. 1D row mesh)
        mesh_dim: the mesh dimension to operate over (usually 0)
    Returns:
        output_tensor: [X, Y//n] shard on each device
    """
    # Number of devices
    world_size = 4

    # Split input_tensor into equal Y-axis chunks
    input_chunks = torch.chunk(input_tensor, world_size, dim=cut_dim)  # [X//n, Y//n] each

    # Prepare output buffers: [X//n, Y//n] for each incoming chunk
    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(world_size)]

    # All-to-all exchange
    output_chunks = all_to_all(
        output_chunks,
        input_chunks,
        group=group,
    )

    # Step 3: concat along dim=0 (X axis)
    output_tensor = torch.cat(output_chunks, dim=cat_dim)
    return output_tensor





def split_sequences(x, rank, dimension) -> torch.tensor:
    if dimension != 0:
        x = torch.transpose(x, 0, dimension)
    B = x.shape[0]
    slice = B//tp_size
    x_local = x[(slice)*rank:(slice)*(rank+1)]
    if dimension != 0:
        x = torch.transpose(x, 0, dimension)
    return x_local

def sequence_gather(x, group, dimension):
    x_list = all_gather(x, group)
    x = torch.cat(x_list, dimension)
    return x

# ---------- Model ----------
class ToyGridModelDistributed(nn.Module):
    def __init__(self, hidden_dim, rank, group):
        super(ToyGridModelDistributed, self).__init__()
        self.net1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(hidden_dim, hidden_dim)
        self.net3 = nn.Linear(hidden_dim, hidden_dim)
        self.net4 = nn.Linear(hidden_dim, 1)
        self.rank = rank
        self.group = group

    def forward(self, x):
        # x: [B, H, W, D]
        B, H, W, D = x.shape
        if dist.get_rank():
            print(x.shape)
        x = self.net1(x)
        x = self.relu(x)
        if dist.get_rank():
            print(x.shape)
        x = transpose_shard_2d_ddp(x, self.group, 1, 2)
        x = self.relu(self.net2(x)) 
        if dist.get_rank():
            print(x.shape)
        x = transpose_shard_2d_ddp(x, self.group, 2, 1)
        x = self.relu(self.net3(x))  
        if dist.get_rank():
            print(x.shape)
        logits = self.net4(x).squeeze(-1)                # [B, H//tp_size, W]
        return logits



def train(model, dataloader, tp_group, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X, y in dataloader:
            
            X, y = X.to(device), y.to(device)
            X = scatter_tensor(X, world_size=4, dim=1, group=tp_group, src=tp_rank)
            y = scatter_tensor(y, world_size=4, dim=1, group=tp_group, src=tp_rank)
            logits = model(X) 
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_step = loss.item() * X.size(0)
            total_loss += loss_step

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1:02d}, Loss: {avg_loss:.4f}")

try:
    model = ToyGridModelDistributed(hidden_dimension, tp_rank, tp_group).cuda()
    dataset = StructuredGridDataset(height=length, width=width, hidden_dim=hidden_dimension)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    train(model, dataloader, tp_group)
finally:
    dist.destroy_process_group()



class ToyGridModel(nn.Module):
    def __init__(self, hidden_dim):
        super(ToyGridModel, self).__init__()
        self.net1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(hidden_dim, hidden_dim)
        self.net3 = nn.Linear(hidden_dim, hidden_dim)
        self.net4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, H, W, D]
        B, H, W, D = x.shape
        
        x = self.relu(self.net1(x))                      # [B, H//tp_size, W, D]
        x_t = x.transpose(1, 2)                          # [B, W//tp_size, H, D]
        x_t = self.relu(self.net2(x_t))                  # [B, W//tp_size, H, D]
        x = x_t.transpose(1, 2)                          # [B, H//tp_size, W, D]
        x = self.relu(self.net3(x))                      # [B, H//tp_size, W, D]
        logits = self.net4(x).squeeze(-1)                # [B, H//tp_size, W]
        return logits