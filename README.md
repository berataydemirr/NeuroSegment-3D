# NeuroSegment-3D: Volumetrik Beyin Tümörü Segmentasyonu

Bu proje, manyetik rezonans görüntüleme (MRI) verileri üzerinden beyin tümörlerini otomatik olarak tespit etmek ve 3 boyutlu uzayda bölütlemek (segmentasyon) amacıyla geliştirilmiş bir derin öğrenme uygulamasıdır.

Sistem, geleneksel 2D kesit analizi yerine, MR görüntülerinin hacimsel (volumetric) yapısını koruyan 3D U-Net mimarisini temel alır. Bu yaklaşım, kesitler arasındaki mekansal derinlik bilgisinin korunmasını ve tümör sınırlarının daha yüksek doğrulukla belirlenmesini sağlar.

## Proje Özeti

Tıbbi görüntü analizinde, özellikle beyin tümörü gibi karmaşık yapılarda, 2 boyutlu analizler bazen derinlik bilgisini kaçırabilir. Bu çalışmada, BraTS (Brain Tumor Segmentation) veri seti standartlarına uygun NIfTI (.nii.gz) formatındaki veriler işlenerek, tümörlü doku ve sağlıklı doku ayrımı yapılmaktadır.

Model, PyTorch tabanlı MONAI (Medical Open Network for AI) kütüphanesi kullanılarak geliştirilmiş ve eğitilmiştir. Son kullanıcı için Gradio kütüphanesi ile interaktif bir web arayüzü oluşturulmuştur.

## Teknik Mimari

Model, Encoder-Decoder yapısına sahip 3D U-Net mimarisini kullanır.

* **Giriş (Input):** 1 Kanal (T1c MRI Sekansı)
* **Çıkış (Output):** 2 Kanal (Arkaplan ve Tümör Maskesi)
* **Derinlik:** Model, 16 kanaldan başlayıp 256 kanala kadar derinleşen 5 seviyeli bir yapıya sahiptir.
* **Optimizasyon:** Rezidüel Bloklar (Residual Units) kullanılarak eğitim verimliliği artırılmış ve gradyan kaybı minimize edilmiştir.

## Kullanılan Teknolojiler

* **Python 3.10**
* **PyTorch & MONAI:** Model eğitimi ve tensör işlemleri.
* **Nibabel:** 3D tıbbi görüntü formatlarının (NIfTI) okunması ve işlenmesi.
* **Gradio:** Modelin test edilmesi için geliştirilen web tabanlı kullanıcı arayüzü.
* **Matplotlib & NumPy:** Veri görselleştirme ve matris manipülasyonu.

## Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/berataydemirr/NeuroSegment-3D.git](https://github.com/berataydemirr/NeuroSegment-3D.git)
   cd NeuroSegment-3D