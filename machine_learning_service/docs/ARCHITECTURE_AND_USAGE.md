# ML/DS Şablonu - Klasör Amaçları

Bu dokuman, tüm ML/DS projelerinizde kullanılabilecek genel şablon yapısını açıklar. Her klasörün ve dosyanın NE AMAÇLA var olduğunu anlatır.

---

## Klasör Yapısı & Amaçları

```
machine_learning_service/
├── docs/                    # Dokümantasyon saklama
├── env/                     # Python virtual environment (git'e yüklenmez)
├── ml_service/              # Ana çalışma paketi
│   ├── applications/        # Eğitim ve prediction giriş noktaları
│   ├── data_layer/          # Veri okuma/yazma işlemleri
│   └── machine_learning/    # ML algoritması ve işlemlemeler
├── tests/                   # Unit testler
└── requirements.txt         # Kütüphane versiyonları
```

---

## Klasörlerin Görevleri

### `docs/` - Dokümantasyon
**Ne için:** Proje hakkında notlar, veri şeması açıklaması, eğitim parametreleri vb. yazılır.

---

### `env/` - Virtual Environment
**Ne için:** Proje kütüphaneleri izole ortamda yüklenir. İşletim sistembilgisayarınızdaki diğer Python projelerini etkilemez.

Tek seferlik kurulum:
```bash
python -m venv env
env\Scripts\activate  # Windows
source env/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

---

### `ml_service/` - Ana Çalışma Paketi
Tüm ML işlemleri burada olur. Organize edilmiş 3 alt klasör vardır.

**`ml_service/applications/`** - Eğitim ve Tahmin Uygulamaları

Amacı: Başlangıç noktaları. Veriyi alıp, işleyip, modeli eğit ve kaydet yada yüklü bir model ile tahmin yap.

- `training.py` → Veri yükle → Process → Eğit → Model kaydet
- `inference.py` → Yüklü model → Yeni veri → Tahmin döndür

**`ml_service/data_layer/`** - Veri I/O (Input/Output)

Amacı: Veriyi okumak ve yazmak. Dosyalardan veri gelir, dosyalara ağırlıklı model kaydedilir.

- `data_connector.py` → CSV, NumPy dosyaları okuma + stream işleme
- `object_connector.py` → Eğitilmiş modeli kaydetme + yükleme, JSON metadata

**`ml_service/machine_learning/`** - ML İşlemleri

Amacı: Veriyi temizlemek, özellik çıkarmak, model kurmak, değerlendirmek.

- `data_processor.py` → Parsing, filtering, normalization (veri temizleme)
- `model.py` → Classifier seçimi (Logistic Regression, SVM, Random Forest)
- `training_pipeline.py` → Standardization → PCA (boyut azaltma) → Classifier (pipeline montajı)
- `cross_validator.py` → Modeli değerlendirme (K-fold cross-validation)

---

### `tests/` - Testler
**Ne için:** Kodun düzgün çalışıp çalışmadığını otomatik kontrol etmek.

- `test_smoke.py` → Basit veri ile pipeline çalışıyor mu kontrolü

Proje geliştikçe daha fazla test ekleyin.

---

### `requirements.txt` - Kütüphane Listesi
**Ne için:** Projede kullanılan Python paketi ve versiyonları yazılır.

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.4
scipy>=1.11
joblib>=1.3
```

Yeni kütüphane eklediğinizde bunu güncelleyin.

---

## Veri Akışı

```
Veri (CSV/NumPy) 
  ↓ (data_connector)
Parse & Filter & Normalize (data_processor)
  ↓ (training_pipeline)
Eğit / Tahmin (model)
  ↓ (object_connector)
Model kaydet veya çıktı döndür
```

---

##  Projenizi Özelleştirme

### 1. Veri Kaynağı Değişirse
`ml_service/data_layer/data_connector.py` → Yeni metod ekle (örn: `load_database()`, `load_api()`)

### 2. Data Processing Adımları Değişirse
`ml_service/machine_learning/data_processor.py` → Yeni metod ekle (örn: `wavelet_transform()`)

### 3. Farklı Modeller Denemek İsterseniz
`ml_service/machine_learning/model.py` → Yeni model seçeneği ekle

### 4. Farklı Validation Metriği İsterseniz
`ml_service/machine_learning/cross_validator.py` → `scoring` parametresini değiştir (f1, roc_auc, vb.)

---

## Standartlaştırılmış Kullanım

### Eğitim Akışı
1. Veriyi `data_layer` ile oku
2. `data_processor` ile temizle
3. `training_pipeline` ile eğit
4. `cross_validator` ile değerlendir
5. `object_connector` ile kaydet

### Tahmin Akışı
1. Model `object_connector` ile yükle
2. Veri `data_layer` ile oku
3. Model ile tahmin yap
4. Sonuç döndür

---


## 📌 Örnek: End-to-End Workflow

### Training Akışı

```python
# 1. Imports
from ml_service.data_layer.data_connector import DataConnector
from ml_service.machine_learning.data_processor import DataProcessor
from ml_service.machine_learning.training_pipeline import TrainingPipeline
from ml_service.machine_learning.cross_validator import evaluate_pipeline
from ml_service.data_layer.object_connector import ObjectConnector

# 2. Setup
connector = DataConnector()
processor = DataProcessor()
obj_connector = ObjectConnector()

# 3. Veri yükle (DATA_CONNECTOR)
df = connector.load_csv("data/train.csv")

# 4. Hazırla (DATA_PROCESSOR)
X, y = processor.parse_eeg_frame(df, label_column="motion")
X = processor.moving_average_filter(X)
X = processor.zscore_normalize(X)

# 5. Değerlendir (CROSS_VALIDATOR)
pipeline = TrainingPipeline()
metrics = evaluate_pipeline(pipeline.pipeline, X, y)

# 6. Eğit (TRAINING_PIPELINE)
pipeline.fit(X, y)

# 7. Kaydet (OBJECT_CONNECTOR)
obj_connector.save_model(pipeline.pipeline, "artifacts/model.joblib")
obj_connector.save_json(metrics, "artifacts/metrics.json")

### Inference Akışı

# 1. Model yükle (OBJECT_CONNECTOR)
model = obj_connector.load_model("artifacts/model.joblib")

# 2. Test verisi yükle (DATA_CONNECTOR)
df_test = connector.load_csv("data/test.csv")
X_test = df_test.drop(columns=["Y"]).to_numpy()

# 3. Tahmin yap
predictions = model.predict(X_test)
print(predictions)

```