import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super(TimeEmbedding, self).__init__()
        self.time_dim = time_dim

    def forward(self, t):
        half_dim = self.time_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1))).to(t.device)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=128):
        super(UNet, self).__init__()

        self.encoder1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.decoder1 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1))
        self.decoder2 = nn.ConvTranspose2d(256, 64, kernel_size=(2, 1), stride=(2, 1))
        self.decoder3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, 256)
        self.time_embedding = TimeEmbedding(time_dim)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        e1 = F.relu(self.encoder1(x))
        e2 = F.relu(self.encoder2(self.pool(e1)))
        e3 = F.relu(self.encoder3(self.pool(e2)))

        # Time embedding addition to bottleneck layer
        b = e3 + t_emb.unsqueeze(-1).unsqueeze(-1)

        # Decoder
        d1 = F.relu(self.decoder1(b))
        d1 = torch.cat((d1, e2), dim=1)
        d2 = F.relu(self.decoder2(d1))
        d2 = torch.cat((d2, e1), dim=1)
        d3 = self.decoder3(d2)

        return d3
    
class FCNet(nn.Module):
    def __init__(self, seq_length, vocab_size, embedding_size, hidden_size, timesteps=1000):
        super(FCNet, self).__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(timesteps, embedding_size)
        self.fc1 = nn.Linear(seq_length * vocab_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size + embedding_size, seq_length * vocab_size)
        self.unflatten = nn.Unflatten(1, (1, vocab_size, seq_length))

    def forward(self, x, t):
        batch_size = x.size(0)
        x_flat = self.flatten(x)
        t_embed = self.embedding(t).view(batch_size, -1)
        x_cat = torch.cat((x_flat, t_embed), dim=1)
        x = torch.relu(self.fc1(x_cat))
        x_cat = torch.cat((x, t_embed), dim=1)
        x = torch.relu(self.fc2(x_cat))
        x_cat = torch.cat((x, t_embed), dim=1)
        x = self.fc3(x_cat)
        
        x = self.unflatten(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return self.relu(out)

class ImprovedFCNet(nn.Module):
    def __init__(self, seq_length, vocab_size, embedding_size, hidden_size, timesteps=1000):
        super(ImprovedFCNet, self).__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(timesteps, embedding_size)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(seq_length * vocab_size + embedding_size, hidden_size)
        self.res_block1 = ResidualBlock(1, hidden_size)
        self.res_block2 = ResidualBlock(hidden_size, 1)
        
        self.fc2 = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.res_block3 = ResidualBlock(1, hidden_size)
        self.res_block4 = ResidualBlock(hidden_size, 1)
        
        self.fc3 = nn.Linear(hidden_size + embedding_size, seq_length * vocab_size)
        self.unflatten = nn.Unflatten(1, (1, vocab_size, seq_length))

    def forward(self, x, t):
        batch_size = x.size(0)
        x_flat = self.flatten(x)
        t_embed = self.embedding(t).view(batch_size, -1)

        x_cat = torch.cat((x_flat, t_embed), dim=1)
        x = torch.relu(self.fc1(x_cat))
        x = self.res_block1(x.unsqueeze(1))
        x = self.res_block2(x).squeeze(1)
        
        x_cat = torch.cat((x, t_embed), dim=1)
        x = torch.relu(self.fc2(x_cat))
        x = self.res_block3(x.unsqueeze(1))
        x = self.res_block4(x).squeeze(1)

        x_cat = torch.cat((x, t_embed), dim=1)
        x = self.fc3(x_cat)
        
        x = self.unflatten(x)
        return x


class PasswordDataset(Dataset):
    def __init__(self, data, charmap):
        self.data = data
        self.charmap = charmap
        self.vocab_size = len(charmap)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_idx = [self.charmap[char] if char in self.charmap else self.charmap['unk'] for char in self.data[idx]]
        one_hot_encoded = np.eye(self.vocab_size)[data_idx]
        one_hot_encoded = one_hot_encoded.astype(np.float32)
        return torch.tensor(one_hot_encoded, dtype=torch.float).permute(1,0).unsqueeze(0)


class DiffusionModel:
    def __init__(self, model, num_timesteps=1000, beta_schedule='linear', device='cpu'):
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device
        if beta_schedule == 'linear':
            self.betas = self._linear_beta_schedule()
        elif beta_schedule == 'sigmoid':
            self.betas = self._sigmoid_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _linear_beta_schedule(self):
        scale = 1000 / self.num_timesteps
        beta = np.linspace(scale * 0.0001, scale * 0.02, self.num_timesteps)
        return torch.tensor(beta, device=self.device).float()
    
    def _sigmoid_beta_schedule(self):
        beta = torch.sigmoid(torch.linspace(-6, 6, self.num_timesteps)) * (5e-3 - 1e-5) + 1e-5
        return beta

    def train(self, dataloader, optimizer, criterion, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                batch_size = batch.size(0)

                t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
                noise = torch.randn_like(batch, device=self.device)

                one_minus_betas_t = 1.0 - self.betas[t]
                sqrt_one_minus_betas_t = torch.sqrt(one_minus_betas_t).view(-1, 1, 1, 1)
                sqrt_betas_t = torch.sqrt(self.betas[t]).view(-1, 1, 1, 1)

                x_t = (batch * sqrt_one_minus_betas_t) + (noise * sqrt_betas_t)

                pred_noise = self.model(x_t, t)

                loss = criterion(pred_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch+1) % 10000 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def sample(self, num_samples, seq_length, vocab_size):
        self.model.eval()
        samples = torch.randn(num_samples, 1, vocab_size, seq_length).to(self.device)
        for t in reversed(range(self.num_timesteps)):
            noise = torch.randn_like(samples) if t > 0 else 0
            pred_noise = self.model(samples, torch.tensor([t] * num_samples, device=self.device))
            samples = (samples - self.betas[t] / torch.sqrt(1 - self.alphas_cumprod[t]) * pred_noise) / torch.sqrt(self.alphas[t])
            samples = samples + noise * torch.sqrt(self.betas[t])
        return torch.softmax(samples, dim=2).argmax(dim=2)
