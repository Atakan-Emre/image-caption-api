# 🐳 Docker Hub Deployment Rehberi

Bu rehber, Image Captioning API'sini Docker Hub'a otomatik CI/CD pipeline ile dağıtmayı ve production'da çalıştırmayı kapsar.

## 📋 Gereksinimler

- Docker Hub hesabı
- GitHub repository'si (kod ile birlikte)
- Docker Hub Access Token (otomatik push'lar için)

## 🔧 Kurulum

### 1. Docker Hub Konfigürasyonu

1. **Docker Hub Repository Oluştur**
   ```bash
   # Docker Hub'da repository oluştur: yourusername/image-caption-api
   # İhtiyacınıza göre Public veya Private olarak ayarla
   ```

2. **Access Token Oluştur**
   - Docker Hub → Account Settings → Security gidin
   - "New Access Token" tıklayın
   - İsim verin (ör: "github-actions")
   - İzinleri seçin: Read, Write, Delete
   - Token'ı kopyalayın (tekrar göremeyeceksiniz)

### 2. GitHub Secrets

Bu secret'ları GitHub repository'nize ekleyin:

```bash
DOCKER_USERNAME=your_dockerhub_username
DOCKER_PASSWORD=your_dockerhub_access_token
```

### 3. Repository Konfigürasyonu

Aşağıdaki dosyaları bilgilerinizle güncelleyin:

**`.github/workflows/docker-hub.yml`:**
```yaml
env:
  REGISTRY: docker.io
  IMAGE_NAME: yourusername/image-caption-api  # Bunu güncelleyin
```

**`Dockerfile.hub`:**
```dockerfile
LABEL maintainer="your-email@example.com"  # Bunu güncelleyin
```

## 🚀 Deployment Süreci

### Otomatik Deployment (Önerilen)

Pipeline otomatik olarak tetiklenir:

- **Main/develop'e push**: Build eder ve branch tag'leri ile push'lar
- **Release**: Versioned tag'ler oluşturur
- **Git tag'leri**: Semantic version tag'ler oluşturur

### Manuel Deployment

1. **Yerel Build ve Push**
   ```bash
   # Optimize edilmiş image'ları build et
   make build-hub-cpu
   make build-hub-gpu
   
   # Docker Hub için tag'le
   docker tag image-caption-api:cpu yourusername/image-caption-api:latest-cpu
   docker tag image-caption-api:gpu yourusername/image-caption-api:latest-gpu
   
   # Docker Hub'a pushla
   docker push yourusername/image-caption-api:latest-cpu
   docker push yourusername/image-caption-api:latest-gpu
   ```

2. **Versioned Release**
   ```bash
   # Version ile tag'le
   docker tag image-caption-api:cpu yourusername/image-caption-api:v1.0.0-cpu
   docker tag image-caption-api:gpu yourusername/image-caption-api:v1.0.0-gpu
   
   # Versioned tag'leri pushla
   docker push yourusername/image-caption-api:v1.0.0-cpu
   docker push yourusername/image-caption-api:v1.0.0-gpu
   ```

## 📦 Image Varyantları

### CPU Varyantı
- **Tag**: `latest-cpu`, `v1.0.0-cpu`
- **Boyut**: ~800MB
- **Kullanım alanı**: GPU'suz production sunucuları
- **Performans**: ~50ms per çıkarım

### GPU Varyantı
- **Tag**: `latest-gpu`, `v1.0.0-gpu`
- **Boyut**: ~2.5GB
- **Kullanım alanı**: GPU destekli sunucular
- **Performans**: ~10ms per çıkarım

## 🔍 Image Özellikleri

### Güvenlik
- Non-root user çalıştırma
- Minimal attack surface
- Trivy ile vulnerability tarama
- SBOM generation
- Cosign ile image signing

### Optimizasyon
- Multi-stage build'ler
- Layer caching
- Minimal base image'ler
- Proper health check'ler
- Efficient dependency management

### Labels ve Metadata
```dockerfile
org.opencontainers.image.title="Image Captioning API"
org.opencontainers.image.description="Production-ready image captioning API"
org.opencontainers.image.version="1.0.0"
org.opencontainers.image.created="2024-01-01T00:00:00Z"
org.opencontainers.image.revision="abc123"
org.opencontainers.image.licenses="MIT"
```

## 🚢 Image'ları Çalıştırma

### CPU Versiyonu
```bash
docker run -d \
  --name caption-api \
  -p 8000:8000 \
  yourusername/image-caption-api:latest-cpu
```

### GPU Versiyonu
```bash
docker run -d \
  --name caption-api \
  --gpus all \
  -p 8000:8000 \
  yourusername/image-caption-api:latest-gpu
```

### Production Kurulumu
```bash
docker run -d \
  --name caption-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=8000 \
  -e LOG_LEVEL=INFO \
  yourusername/image-caption-api:latest-cpu
```

### Docker Compose
```yaml
version: '3.8'
services:
  caption-api:
    image: yourusername/image-caption-api:latest-cpu
    container_name: caption-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## 📊 İzleme ve Bakım

### Health Check'ler
```bash
# Container sağlığını kontrol et
docker ps --filter "name=caption-api"

# Health log'larını görüntüle
docker inspect caption-api --format='{{json .State.Health}}'
```

### Log'lar
```bash
# Log'ları görüntüle
docker logs caption-api

# Log'ları takip et
docker logs -f caption-api

# Son 100 satır
docker logs --tail 100 caption-api
```

### Güncellemeler
```bash
# Son versiyonu çek
docker pull yourusername/image-caption-api:latest-cpu

# Yeni image ile yeniden oluştur
docker stop caption-api
docker rm caption-api
docker run -d --name caption-api -p 8000:8000 yourusername/image-caption-api:latest-cpu
```

## 🔄 CI/CD Pipeline

### Workflow Trigger'ları

| Event | Action | Result |
|-------|--------|--------|
| Push to main | Build & test | Push `main-cpu`, `main-gpu` |
| Push to develop | Build & test | Push `develop-cpu`, `develop-gpu` |
| Create release | Build & test | Push versioned tags |
| Create tag v1.0.0 | Build & test | Push `v1.0.0-cpu`, `v1.0.0-gpu` |

### Pipeline Aşamaları

1. **Test**: Smoke test'leri çalıştır
2. **Build**: Multi-platform build'ler (amd64/arm64)
3. **Security**: Trivy vulnerability tarama
4. **Sign**: Cosign image signing
5. **Deploy**: Docker Hub'a push
6. **Docs**: Docker Hub README'sini güncelle

### Güvenlik Özellikleri

- **Vulnerability Tarama**: Trivy CVE taraması
- **Image Signing**: Bütünlük için Cosign imzaları
- **SBOM**: Software Bill of Materials
- **Non-root**: Container non-root user olarak çalışır
- **Minimal Base**: Slim Python image'ler attack surface'i azaltır

## 📈 Performans Optimizasyonu

### Resource Limitleri
```bash
docker run -d \
  --name caption-api \
  --memory=2g \
  --cpus=1.0 \
  -p 8000:8000 \
  yourusername/image-caption-api:latest-cpu
```

### GPU Optimizasyonu
```bash
docker run -d \
  --name caption-api \
  --gpus '"device=0"' \
  --shm-size=1g \
  -p 8000:8000 \
  yourusername/image-caption-api:latest-gpu
```

### Caching
```bash
docker run -d \
  --name caption-api \
  -v cache:/app/cache \
  -p 8000:8000 \
  yourusername/image-caption-api:latest-cpu
```

## 🐛 Sorun Giderme

### Yaygın Sorunlar

1. **Permission Denied**
   ```bash
   # Docker Hub kimlik bilgilerini kontrol et
   docker login -u yourusername -p yourtoken
   
   # GitHub secret'larını doğrula
   echo $DOCKER_USERNAME
   echo $DOCKER_PASSWORD
   ```

2. **Build Hataları**
   ```bash
   # Debug için yerel build
   docker build -f Dockerfile.hub --no-cache .
   
   # Build log'larını kontrol et
   docker buildx build --progress=plain .
   ```

3. **Runtime Hataları**
   ```bash
   # Container log'larını kontrol et
   docker logs caption-api
   
   # Container içine gir
   docker exec -it caption-api bash
   ```

4. **GPU Sorunları**
   ```bash
   # GPU kullanılabilirliğini kontrol et
   nvidia-smi
   
   # GPU container'ı test et
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

### Debug Komutları

```bash
# Image inceleme
docker inspect yourusername/image-caption-api:latest-cpu

# Layer analizi
docker history yourusername/image-caption-api:latest-cpu

# Boyut analizi
docker system df

# Temizlik
docker system prune -f
```

## 📚 Ek Kaynaklar

- [Docker Hub Dokümantasyonu](https://docs.docker.com/docker-hub/)
- [GitHub Actions Dokümantasyonu](https://docs.github.com/en/actions)
- [Cosign Dokümantasyonu](https://sigstore.github.io/cosign/)
- [Trivy Dokümantasyonu](https://aquasecurity.github.io/trivy/)

## 🆘 Destek

Deployment sorunları için:
1. GitHub Actions log'larını kontrol et
2. Docker Hub build log'larını gözden geçir
3. Repository konfigürasyonunu doğrula
4. Yerel build sürecini test et

---

**🎉 Image Captioning API'niz artık Docker Hub'da production deployment için hazır!**
