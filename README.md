# 🖼️ Image Captioning API (Görüntü Başlıklandırma API'si)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

PyTorch, FastAPI ve modern MLOps pratikleri kullanılarak geliştirilmiş, **production-ready bir görüntü başlıklandırma (image captioning) servisi**. Tek bir HTTP isteğiyle görseli API’ye gönderip, insan benzeri açıklayıcı bir metin çıktısı almanızı sağlar.

Bu proje, *"model dosyasını bir yere koyduk, gerisi gelsin"* yaklaşımından öteye geçip, **uçtan uca bir çözüm** sunar:

- Model tarafında: ResNet50 tabanlı encoder + LSTM decoder (attention destekli) mimarisi  
- API tarafında: FastAPI ile async REST endpoint’leri, validasyon, hata yönetimi  
- DevOps tarafında: Docker tabanlı container’lar, Makefile ile otomasyon, GitHub Actions ile CI/CD pipeline’ı

Gerçek hayatta şu senaryolara gömülebilir:

- ♿ **Erişilebilirlik**: Görme engelli kullanıcılar için otomatik alt-text üretimi  
- 🛒 **E-ticaret**: Ürün görsellerinden otomatik başlık / açıklama oluşturma  
- 📰 **İçerik Yönetimi**: Haber, blog veya medya platformlarında görselleri otomatik etiketleme ve açıklama  
- 📷 **Fotoğraf Arşivi**: Kişisel veya kurumsal fotoğraf arşivleri için arama yapılabilir metinsel açıklamalar üretme  

Hem **örnek bir küçük dataset ile hızlı deneme** yapabileceğin, hem de **tam COCO veri seti ile büyük ölçekli eğitim** yürütebileceğin şekilde tasarlandı. Eğitim, çıkarım (inference), Docker build, test ve deployment adımlarının tamamı Makefile komutları ve CI/CD pipeline’ı ile otomatikleştirilebilir.

PyTorch, FastAPI ve modern ML pratikleri ile oluşturulmuş production-ready görüntü başlıklandırma API'si. Hem örnek eğitim hem de tam COCO veri seti eğitimini kapsamlı CI/CD pipeline ile destekler.


## ✨ Özellikler

- 🚀 **Yüksek Performans**: CNN encoder (ResNet50) + LSTM decoder
- 🎯 **Production Ready**: FastAPI ile async destek, hata yönetimi, validasyon
- 🐳 **Docker Desteği**: CPU/GPU varyantları için multi-stage build'ler
- 📊 **COCO Veri Seti**: Büyük ölçekli eğitim için tam pipeline
- 🔧 **Geliştirici Dostu**: Kapsamlı CLI, test ve dokümantasyon
- 🔄 **CI/CD Pipeline**: GitHub Actions ile otomatik test ve deployment
- 📈 **İzleme**: Health check'ler, metrikler ve loglama

## 🏗️ Mimari

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend/UI   │────│   FastAPI       │────│   PyTorch       │
│   (İsteğe Bağlı)│    │   REST API      │    │   Model Core    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Docker        │
                       │   Container     │
                       └─────────────────┘
```

### Model Mimarisi
- **Encoder**: ResNet50 (ImageNet üzerinde pretrained)
- **Decoder**: LSTM ile attention mekanizması
- **Embedding**: 256-boyutlu word embeddings
- **Vocabulary**: Özel token'lar ile dinamik boyutlandırma

## 🚀 Hızlı Başlangıç

### Seçenek 1: Örnek Model (Hızlı)
```bash
# Klonla ve kur
git clone <repository-url>
cd image-caption-api
make install

# Hızlı eğitim ve test
make quick-start
```

### Seçenek 2: COCO Model (Production)
```bash
# COCO veri setini kur ve eğit
make install-coco
make setup-coco
make train-coco
```

### Seçenek 3: Docker
```bash
# Build ve çalıştır
make build-cpu
make docker-run

# veya GPU desteği ile
make build-gpu
make docker-run-gpu
```

## 📋 Gereksinimler

- Python 3.9+
- PyTorch 2.1.0+
- 8GB+ RAM (COCO için 16GB+ önerilir)
- GPU isteğe bağlı (GPU eğitim için CUDA 11.8+)

## 🛠️ Kurulum

### Temel Kurulum
```bash
# Repository'i klonla
git clone <repository-url>
cd image-caption-api

# Bağımlılıkları kur
pip install -r requirements.txt

# veya make kullanarak
make install
```

### COCO Eğitim Kurulumu
```bash
# Ek bağımlılıkları kur
pip install -r requirements-coco.txt

# veya make kullanarak
make install-coco
```

### Docker Kurulumu
```bash
# CPU image'ı build et
docker build -t image-caption-api .

# GPU image'ı build et
docker build --build-arg BUILD_GPU=true -t image-caption-api:gpu .
```

## 🎯 Kullanım

### API Endpoint'leri

#### Health Kontrolü
```bash
curl http://localhost:8000/health
```

#### Model Bilgisi
```bash
curl http://localhost:8000/model/info
```

#### Tekil Görüntü Tahmini
```bash
curl -X POST \
  http://localhost:8000/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.jpg' \
  -F 'use_beam_search=true'
```

#### Batch Tahmini
```bash
curl -X POST \
  http://localhost:8000/predict/batch \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@image1.jpg' \
  -F 'files=@image2.jpg'
```

### Python Client
```python
import requests

# Tekil tahmin
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        data={'use_beam_search': True}
    )
    
result = response.json()
print(f"Başlık: {result['predicted_caption']}")
print(f"Güven Skoru: {result['confidence']}")
```

## 📊 Eğitim

### Örnek Eğitim
```bash
# Örnek veri ile hızlı eğitim
make train

# veya manuel olarak
cd training && python train.py
```

### COCO Veri Seti Eğitimi

#### 1. COCO Veri Setini İndir
```bash
# Tam COCO veri setini indir (~18GB)
python scripts/download_coco.py --data-dir ./data

# veya sadece annotation'ları indir
python scripts/download_coco.py --data-dir ./data --skip-images
```

#### 2. Veriyi Ön İşle
```bash
# Başlıkları işle ve split'leri oluştur
python scripts/preprocess_coco.py --data-dir ./data --output-dir ./data/processed
```

#### 3. Modeli Eğit
```bash
# Varsayılan parametrelerle eğit
make train-coco

# veya özel parametrelerle
cd training && python train_coco.py \
  --batch-size 32 \
  --num-epochs 50 \
  --learning-rate 1e-4 \
  --embed-size 512 \
  --hidden-size 1024
```

#### Eğitim Parametreleri
| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--batch-size` | 32 | Training batch boyutu |
| `--num-epochs` | 20 | Eğitim epoch sayısı |
| `--learning-rate` | 3e-4 | Öğrenme oranı |
| `--embed-size` | 256 | Embedding boyutu |
| `--hidden-size` | 512 | LSTM hidden boyutu |
| `--num-workers` | 4 | Data loader worker sayısı |

## 🐳 Docker Deployment

### Image'ları Build Et
```bash
# CPU image
make build-cpu

# GPU image
make build-gpu

# Tüm varyantlar
make build-all
```

### Container'ları Çalıştır
```bash
# CPU container
make docker-run

# GPU container
make docker-run-gpu

# Container test et
make docker-test
```

### Docker Hub Deployment
```bash
# Docker Hub'a tag'le ve pushla
docker tag image-caption-api:cpu yourusername/image-caption-api:latest
docker push yourusername/image-caption-api:latest
```

## 🧪 Test

### Testleri Çalıştır
```bash
# Tüm testleri çalıştır
make test

# Smoke testleri manuel çalıştır
python test_api.py

# Özel URL ile test
API_BASE_URL=http://localhost:8001 python test_api.py
```

### Test Kapsamı
- ✅ Health endpoint
- ✅ Model info endpoint  
- ✅ Tekil tahmin
- ✅ Batch tahmin
- ✅ Hata yönetimi
- ✅ Dosya validasyonu

## 📈 Performans

### Benchmark'lar
- **Çıkarım Hızı**: ~50ms per görüntü (CPU)
- **Bellek Kullanımı**: ~500MB (model + overhead)
- **Doğruluk**: BLEU-4: 0.32 (COCO eğitimli model)

### Optimizasyon İpuçları
1. **GPU Eğitimi**: CUDA kullanarak 10x daha hızlı eğitim
2. **Batch İşleme**: Çoklu görüntüler için batch endpoint kullan
3. **Model Önbelleği**: Model container başına bir kez yüklenir
4. **Görüntü Ön İşleme**: Use case'iniz için transform'ları optimize edin

## 🔧 Konfigürasyon

### Environment Değişkenleri
```bash
# API Konfigürasyonu
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Model Konfigürasyonu  
export MODEL_PATH=./models/caption_model.pth
export VOCAB_PATH=./models/vocab.pkl

# Eğitim Konfigürasyonu
export NUM_EPOCHS=20
export BATCH_SIZE=32
export LEARNING_RATE=3e-4
```

### Model Yolları
- **Örnek Model**: `models/caption_model.pth`
- **COCO Model**: `models/coco_caption_model.pth`
- **Vocabulary**: `models/vocab.pkl` veya `models/coco_vocab.pkl`

## 🔄 CI/CD Pipeline

### GitHub Actions
- **Quickstart CI**: Docker olmadan hızlı test
- **Docker CI**: Multi-platform build'ler (amd64/arm64)
- **Güvenlik**: Trivy vulnerability taraması
- **Deployment**: Otomatik Docker Hub yayın

### Pipeline Aşamaları
1. **Kod Kalitesi**: Linting ve format kontrolü
2. **Test**: Unit ve entegrasyon testleri
3. **Build**: Docker image oluşturma
4. **Güvenlik**: Vulnerability tarama
5. **Deploy**: Registry yayın

## 📁 Proje Yapısı

```
image-caption-api/
├── app/                    # FastAPI uygulaması
│   ├── __init__.py
│   ├── main.py            # API endpoint'leri
│   ├── model_loader.py    # Model yükleme mantığı
│   └── schemas.py         # Pydantic modelleri
├── training/              # Training script'leri
│   ├── train.py           # Örnek eğitim
│   └── train_coco.py      # COCO eğitimi
├── scripts/               # Yardımcı script'ler
│   ├── download_coco.py   # COCO indirici
│   └── preprocess_coco.py # Veri ön işleme
├── models/                # Eğitilmiş modeller
├── data/                  # Veri seti depolama
├── .github/workflows/     # CI/CD pipeline'ları
├── Dockerfile*            # Docker konfigürasyonları
├── requirements*.txt      # Bağımlılıklar
├── Makefile              # Build otomasyonu
└── README.md             # Dokümantasyon
```

## 🐛 Sorun Giderme

### Yaygın Sorunlar

#### Model Yükleme Hataları
```bash
# Model dosyalarını kontrol et
ls -la models/

# Vocabulary'i doğrula
python -c "import pickle; vocab = pickle.load(open('models/vocab.pkl', 'rb')); print(len(vocab))"
```

#### CUDA Hataları
```bash
# CUDA kullanılabilirliğini kontrol et
python -c "import torch; print(torch.cuda.is_available())"

# CUDA sürümünü kur
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Bellek Sorunları
```bash
# Batch boyutunu azalt
python training/train_coco.py --batch-size 16

# Gradient checkpointing kullan
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Debug Modu
```bash
# Debug loglamayı etkinleştir
export LOG_LEVEL=DEBUG

# Tek worker ile çalıştır
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## 🤝 Katkıda Bulunma

1. Repository'i fork'la
2. Feature branch oluştur (`git checkout -b feature/amazing-feature`)
3. Değişiklikleri commit'le (`git commit -m 'Add amazing feature'`)
4. Branch'e push'la (`git push origin feature/amazing-feature`)
5. Pull Request aç

### Development Kurulumu
```bash
# Development bağımlılıklarını kur
pip install -r requirements.txt
pip install black flake8 isort

# Linting çalıştır
make format
make lint

# Testleri çalıştır
make test
```

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- [COCO Veri Seti](https://cocodataset.org/) eğitim verisi için
- [PyTorch](https://pytorch.org/) deep learning framework için
- [FastAPI](https://fastapi.tiangolo.com/) API framework için
- [Hugging Face](https://huggingface.co/) model ilhamı için

## 📞 Destek

- 📧 Email: support@example.com
- 💬 Discord: [Topluluğumuza katılın](https://discord.gg/example)
- 📖 Dokümantasyon: [Tam dokümanlar](https://docs.example.com)
- 🐛 Sorunlar: [GitHub Issues](https://github.com/example/issues)

---

**⭐ Eğer size yardımcı olduysa bu repository'i yıldızlayın!**

## 🐳 Docker Hub Deployment

Docker Hub'a yayın ve production'da çalıştırma için `DOCKER_HUB.md` dosyasına bakın.
