# 1. Baz Imaj: PyTorch yüklü hazir Linux (Python 3.10 icerir)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2. Calisma klasoru olustur
WORKDIR /app

# 3. Gereksinimleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Tum kodlari ve modelleri kopyala
COPY . .

# 5. Gradio portunu dısariya ac
EXPOSE 7860

# 6. Uygulamayi baslat
CMD ["python", "app_deploy.py"]