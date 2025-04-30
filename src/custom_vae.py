import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentOutput:
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar

    @property
    def latent_dist(self):
        class Dist:
            def __init__(self, mu, logvar):
                self.mu = mu
                self.logvar = logvar

            def sample(self):
                std = torch.exp(0.5 * self.logvar)
                eps = torch.randn_like(std)
                return self.mu + eps * std

        return Dist(self.mu, self.logvar)

class CustomVAE(nn.Module):
    def __init__(self, latent_dim=4):
        super(CustomVAE,self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1), # 128x128
            nn.ReLU(), 
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1), # 64x64
            nn.ReLU(),  
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1), # 32x32
            nn.ReLU(), 
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1), # 16x16
            nn.ReLU(), 
            nn.Conv2d(512,512,kernel_size=4,stride=2,padding=1), # 8x8
            nn.ReLU(), 
        )
        self.conv_mu = nn.Conv2d(512, 4, kernel_size=1)
        self.conv_logvar = nn.Conv2d(512, 4, kernel_size=1)

        # Decoder: 4→8x8x512 → 128x128x3

        self.decoder_input = nn.Linear(latent_dim, 8*8*512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 512, 4, 2, 1), nn.ReLU(), # 8 -> 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(), # 16 ->32
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), # 32 ->64
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()     # 128 ->256
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        return LatentOutput(mu, logvar)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        mu = latent.mu
        logvar = latent.logvar
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar