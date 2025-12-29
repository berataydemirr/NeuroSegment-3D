import os
import glob
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader, CacheDataset
# Gerekli import'u en tepeye ekle
from monai.transforms import MapLabelValued

def get_brats_loaders(data_dir, batch_size=2):
    """
    BraTS 2024 verisini yükler.
    Giriş: T1c, T1n, T2f, T2w (4 kanal MRI)
    Çıkış: Tümör Maskesi (Segmentation)
    """
    # 1. Dosya Yollarını Bul
    # data_dir icindeki tüm klasörleri tara
    # Her hasta klasöründe t1c, t1n, t2f, t2w ve seg dosyaları vardir
    train_images = sorted(glob.glob(os.path.join(data_dir, "**", "*t1c.nii.gz"), recursive=True))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "**", "*seg.nii.gz"), recursive=True))

    data_dicts = []
    # Her hasta için dosya sözlüğü oluştur
    for t1c in train_images:
        patient_base = t1c.replace("-t1c.nii.gz", "")
        seg = patient_base + "-seg.nii.gz"
        
        # Diger modaliteleri de ekleyebiliriz (t1n, t2f, t2w)
        # Simdilik GPU hafizasi icin sadece T1c (Kontrastli) kullanalim
        if os.path.exists(seg):
            data_dicts.append({"image": t1c, "label": seg})

    print(f"Toplam Hasta Sayısı: {len(data_dicts)}")
  # 2. Transform (Veri İşleme) Pipeline
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]), 
        
        # --- YENİ EKLENECEK SATIR BAŞLANGICI ---
        # BraTS etiketleri (1, 2, 4) -> Tek bir Tümör etiketi (1) yapıyoruz.
        # "Whole Tumor" segmentasyonu için:
        MapLabelValued(
            keys=["label"], 
            orig_labels=[0, 1, 2, 3, 4], 
            target_labels=[0, 1, 1, 1, 1]
        ),
        # --- YENİ EKLENECEK SATIR BİTİŞİ ---

        # 3D Kırpma
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=1,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    # 3. CacheDataset (Veriyi RAM'e alıp hızlandırır)
    # Hata alırsan CacheDataset yerine normal Dataset kullan
    ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=0.5)
    
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader