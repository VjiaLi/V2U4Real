import torch
import torch.nn as nn


class NaiveCompressor(nn.Module):
    """
    A very naive compression that only compress on the channel.
    """
    def __init__(self, input_dim, compress_raito):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim//compress_raito, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim//compress_raito, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim//compress_raito, input_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3,
                           momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x, use_fp16=False):
        x = self.encoder(x)
        if use_fp16:
            x = x.to(torch.float16).to(torch.float32)
        x = self.decoder(x)

        return x



class ImprovedCompressor(nn.Module):
    """
    Compress both spatial dimensions and channels, while ensuring input and output sizes remain consistent.
    Used for point_pillar_baseline_multiscale beacuse the channel of some layers is not enough.
    """
    def __init__(self, input_dim, compress_ratio, stride=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // compress_ratio, kernel_size=3,
                      stride=stride, padding=1),  # Compress spatial dimensions
            nn.BatchNorm2d(input_dim // compress_ratio, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim // compress_ratio, input_dim, kernel_size=3,
                               stride=stride, padding=1, output_padding=stride-1),  # Restore spatial dimensions
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x, use_fp16=False):
        x = self.encoder(x)
        if use_fp16:
            x = x.to(torch.float16).to(torch.float32)
        x = self.decoder(x)
        return x