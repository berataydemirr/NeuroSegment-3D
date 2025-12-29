import torch
import os
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from dataset import get_brats_loaders
from model import get_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    # Veri Yolu (Zip'i buraya açmıştın)
    DATA_DIR = "data/raw" 
    
    train_loader = get_brats_loaders(DATA_DIR, batch_size=2)
    model = get_model(device)
    
    # Dice Loss: Segmentasyon icin standart kayip fonksiyonu
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    
    max_epochs = 10
    print("3D Egitim Basliyor...")
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            step += 1
            print(f"{step}/{len(train_loader)}, Loss: {loss.item():.4f}", end="\r")
        
        print(f"\nEpoch {epoch+1} Ort. Loss: {epoch_loss/step:.4f}")
        
        # Modeli Kaydet
        if (epoch+1) % 2 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/brats_unet_epoch{epoch+1}.pth")
            print("Model kaydedildi.")

if __name__ == "__main__":
    train()