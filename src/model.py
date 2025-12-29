import torch
from monai.networks.nets import UNet

def get_model(device):
    # MONAI'nin hazir 3D UNet'i
    model = UNet(
        spatial_dims=3,          # 3 Boyutlu
        in_channels=1,           # Giriş (Sadece T1c kullandık)
        out_channels=2,          # Çıkış (0: Arkaplan, 1: Tümör)
        channels=(16, 32, 64, 128, 256), # Katman genislikleri
        strides=(2, 2, 2, 2),    # Boyut küçültme adımları
        num_res_units=2,
    ).to(device)
    
    return model