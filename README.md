# K-Means Clustering (From Scratch) - Streamlit App

Bu proje, K-Means kümeleme algoritmasının mantığını anlamak ve görselleştirmek amacıyla geliştirilmiş interaktif bir web uygulamasıdır. 

🚨 **Önemli Özellik:** K-Means algoritması **`scikit-learn` vb. hazır kütüphaneler kullanılmadan**, tamamen saf Python mantığı ile sıfırdan (from scratch) yazılmıştır.

## 🌟 Özellikler

* **Veri Yükleme:**
    * CSV ve Excel (.xlsx) dosyalarını destekler.
    * Test amaçlı rastgele (random) veri seti üretebilir.
* **Ön İşleme (Preprocessing):**
    * Min-Max Normalizasyonu (0-1 arası).
    * Z-Score Standardizasyonu.
    * Ham veri kullanımı.
* **Algoritma Kontrolü:**
    * Küme sayısı (k), maksimum iterasyon ve yeniden başlatma (restarts) sayısını belirleme.
    * **Manuel Merkez Atama:** Başlangıç merkez noktalarını elle girme imkanı.
* **Görselleştirme ve Analiz:**
    * 2 Boyutlu (2D) dağılım grafiği (Scatter Plot).
    * SSE (Sum of Squared Errors) hesaplaması.
    * Küme merkezleri ve eleman sayılarının raporlanması.
* **Dışa Aktarma:**
    * Sonuçların (Featurelar + Etiketler) CSV olarak indirilmesi.

## 🛠 Kurulum

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/REPO_ADINIZ.git)
cd REPO_ADINIZpip install streamlit pandas matplotlib openpyxl
