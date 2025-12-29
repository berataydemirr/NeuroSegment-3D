import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from monai.transforms import Compose, ScaleIntensity, Resize, EnsureChannelFirst
from src.model import get_model # src klasorunden cagiriyoruz
import os

# 1. Cihaz Ayarlari (Docker icinde CPU kullanmasi daha guvenlidir, GPU opsiyonel)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Calisma Ortami: {device}")

# 2. Modeli Yukle
model = get_model(device)
model_path = "models/brats_unet_epoch10.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("--> Model Docker icine basariyla yuklendi.")
else:
    print("HATA: Model dosyasi bulunamadi!")

model.eval()

# 3. Analiz Fonksiyonu
def predict_slice(file_obj, slice_idx):
    if file_obj is None:
        return None
    
    # Dosyayi oku
    nifti_data = nib.load(file_obj.name)
    data = nifti_data.get_fdata() # (240, 240, 155) gibi gelir
    
    # Basit Preprocessing (96x96x96'ya indir)
    # Gercek urunde burasi daha detayli olur
    data_tensor = torch.tensor(data).float().unsqueeze(0).unsqueeze(0)
    data_tensor = torch.nn.functional.interpolate(data_tensor, size=(96, 96, 96), mode='trilinear')
    data_tensor = data_tensor.to(device)
    
    # Tahmin
    with torch.no_grad():
        output = model(data_tensor)
        output = (output.sigmoid() > 0.5).float()
        
    # Numpy'a cevir
    vol_img = data_tensor[0, 0].cpu().numpy()
    vol_seg = output[0, 1].cpu().numpy()
    
    # Kesiti al (Slider degerine gore)
    max_slice = vol_img.shape[2] - 1
    idx = int(slice_idx * max_slice / 100)
    
    # Gorsellestirme
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor='#f0f2f6')
    
    ax[0].imshow(vol_img[:, :, idx], cmap="gray")
    ax[0].set_title(f"Orijinal MR (Kesit: {idx})", fontsize=14)
    ax[0].axis("off")
    
    ax[1].imshow(vol_img[:, :, idx], cmap="gray")
    ax[1].imshow(vol_seg[:, :, idx], cmap="autumn", alpha=0.5) # Kirmizi maske
    ax[1].set_title("Yapay Zeka Tespiti", fontsize=14, color='red')
    ax[1].axis("off")
    
    return fig

# 4. Profesyonel Arayuz Tasarimi
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 hTumor-AI: Akıllı Tümör Tespit Sistemi
        **Hevi AI Demo Ürünü** | 3D Beyin MR Görüntüleme Analizi
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Hasta MR Dosyası (.nii.gz)", file_types=[".gz"])
            slider = gr.Slider(0, 100, value=50, label="Kesit Derinliği (Scroll)")
            btn = gr.Button("Analiz Et", variant="primary")
            gr.Markdown("ℹ️ *Bu sistem araştırma amaçlıdır. Kesin tanı için doktora danışınız.*")
            
        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Radyolojik İnceleme")
    
    # Etkilesimler
    btn.click(fn=predict_slice, inputs=[file_input, slider], outputs=output_plot)
    slider.change(fn=predict_slice, inputs=[file_input, slider], outputs=output_plot)
# share=True parametresi sana 72 saatlik herkese açık bir link verir
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)