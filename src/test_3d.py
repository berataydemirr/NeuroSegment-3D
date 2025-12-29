import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.inferers import sliding_window_inference
from dataset import get_brats_loaders
from model import get_model

def visualize_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test cihazi: {device}")

    # 1. Modeli Yukle
    model = get_model(device)
    model.load_state_dict(torch.load("models/brats_unet_epoch10.pth"))
    model.eval()

    # 2. Veriden Bir Ornek Al (Validation seti gibi dusun)
    # Egitim verisinden rastgele bir tane alacagiz
    DATA_DIR = "data/raw"
    loader = get_brats_loaders(DATA_DIR, batch_size=1)
    
    # Ilk batch'i cek
    data = next(iter(loader))
    image, label = data["image"].to(device), data["label"].to(device)
    
    print("Goruntu boyutu:", image.shape) # Ornek: (1, 1, 96, 96, 96)

    # 3. Tahmin Yap (Inference)
    with torch.no_grad():
        # Sliding Window: Buyuk resmi parca parca tarayip birlestirir (Hafiza dostu)
        output = sliding_window_inference(
            inputs=image, 
            roi_size=(96, 96, 96), 
            sw_batch_size=4, 
            predictor=model
        )
        # Ciktiyi 0 veya 1 yap (Threshold)
        output = (output.sigmoid() > 0.5).float()

    # 4. Gorsellestirme (Orta Kesiti Al)
    # image format: (Batch, Channel, H, W, Depth) -> (96, 96, 96)
    img_np = image[0, 0].cpu().numpy()
    lbl_np = label[0, 0].cpu().numpy()
    pred_np = output[0, 1].cpu().numpy() # 1. Kanal Tumordur

    # Dilim numarasi (Tam ortadan bakalim)
    slice_idx = img_np.shape[2] // 2 

    plt.figure("Sonuc", (12, 4))

    # A) Orijinal MR
    plt.subplot(1, 3, 1)
    plt.title("Orijinal MR (T1c)")
    plt.imshow(img_np[:, :, slice_idx], cmap="gray")
    plt.axis("off")

    # B) Gercek Maske (Doktorun Cizdigi)
    plt.subplot(1, 3, 2)
    plt.title("Gercek Maske (Ground Truth)")
    plt.imshow(lbl_np[:, :, slice_idx], cmap="gray")
    plt.axis("off")

    # C) Yapay Zeka Tahmini
    plt.subplot(1, 3, 3)
    plt.title(f"AI Tahmini (Dice Loss: 0.26)")
    plt.imshow(pred_np[:, :, slice_idx], cmap="gray")
    plt.axis("off")

    plt.savefig("sonuc_analizi.png")
    print("\n--> 'sonuc_analizi.png' dosyasi kaydedildi! Dosyayi acip bakabilirsin.")

if __name__ == "__main__":
    visualize_results()