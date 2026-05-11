from torch import nn


class PCR(nn.Module):
    def __init__(self, num_input_features):
        super().__init__()
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(num_input_features, 640,1,1,0),
            nn.BatchNorm2d(640),
            nn.GELU(),
        )  
        self.generator_1 = nn.Sequential( # N,128,5,188,188
            nn.Conv3d(128,32,1,1,0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32,32,4,2,1), # N,32,10,376,376
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )
        self.gen_out_4 = nn.Sequential(
            nn.Conv3d(32,3,1,1,0),
        )
        self.gen_mask_4 = nn.Sequential(
            nn.Conv3d(32,1,1,1,0),
        )

        self.generator_2 = nn.Sequential(
            nn.Conv3d(32,16,1,1,0),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16,3,4,2,1), # N,16,20,752,752
            nn.BatchNorm3d(3),
            nn.ReLU(),
        )

        self.gen_out_2 = nn.Sequential(
            nn.Conv3d(3,3,1,1,0),
        )

        self.gen_mask_2 = nn.Sequential(
            nn.Conv3d(3,1,1,1,0),
        )


    def forward(self, x):
        
        N, _, H, W = x.shape
        gen = self.out_conv(x)
        gen = gen.view(N,128,5,H,W)
        gen = self.generator_1(gen)
        gen_offset_4 =self.gen_out_4(gen)
        gen_mask_4 = self.gen_mask_4(gen)

        return gen_offset_4, gen_mask_4
