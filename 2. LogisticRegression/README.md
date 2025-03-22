# YZM212_LogisticRegression

Bu projede, Telco Customer Churn veri seti kullanılarak Logistic Regression yöntemiyle ikili sınıflandırma uygulanmıştır. İki farklı yaklaşım incelenmiştir:  
- **Scikit-learn ile Logistic Regression**  
- **Custom Logistic Regression**

## 1. Proje Genel Bakışı
- **Amaç:** Telco Customer Churn veri seti üzerinde logistic regression modeli oluşturarak müşteri kaybını tahmin etmek.
- **Yöntemler:** Veri ön işleme, özellik ölçeklendirme, logistic regression uygulaması, model değerlendirmesi (confusion matrix, accuracy, precision, recall, f1-score).

## 2. Veri Seti
- **Kaynak:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Özellikler:** Tabular veri, 1000’den fazla örnek, sayısal ve kategorik değişkenler.
- **Hedef Değişken:** `Churn` (0: No, 1: Yes)

## 3. Veri Ön İşleme
- Gereksiz sütunların kaldırılması (örn. `customerID`).
- Eksik verilerin doldurulması.
- Hedef değişkenin dönüştürülmesi (string ise 'Yes'/'No' → 1/0, numeric ise direk kullanılması).
- Kategorik değişkenlerin one-hot encoding ile sayısallaştırılması.
- Özellik ölçeklendirme (StandardScaler).
- Gerekirse log dönüşümü (örn. `TotalCharges` → `TotalCharges_log`).

## 4. Yöntemler
### 4.1. Scikit-learn Logistic Regression
- **Kütüphane:** `sklearn.linear_model.LogisticRegression`
- **Avantajlar:** Hızlı eğitim, optimize edilmiş algoritma.
- **Değerlendirme:** Eğitim/tahmin süresi, confusion matrix, classification report.

### 4.2. Custom Logistic Regression
- **Uygulama:** Gradient descent algoritması kullanılarak sıfırdan kodlanmıştır.
- **Adımlar:** 
  - Sigmoid fonksiyonu ile olasılık hesaplama.
  - Cost function hesaplama ve maksimum likelihood estimation.
  - Ağırlık (W) ve bias (b) güncelleme.
- **Değerlendirme:** Scikit-learn modeli ile benzer metrikler üzerinden karşılaştırma.

## 5. Sonuç ve Yorum
- **Performans Metrikleri:**  
  Scikit-learn modeli %74 genel doğruluk (accuracy) ve her iki sınıf için de dengeli precision, recall ve f1-score değerleri göstermiştir. Karmaşıklık matrisine baktığımızda, sınıf 0 (churn olmayan) ve sınıf 1 (churn olan) arasında bir miktar hatalı sınıflandırma gözlemlenmiştir; örneğin, churn olmayan müşterilerden bazıları yanlışlıkla churn olarak sınıflandırılmış, churn olan müşterilerden de bir kısmı atlanmıştır. Bu durum, modelin bazı örneklerde sınıf dengesizliğinden veya verinin özelliklerinden kaynaklanan zorluklar yaşadığını göstermektedir.
- **Model Karşılaştırması:**  
  - **Scikit-learn Modeli:** Optimize edilmiş algoritmalar sayesinde oldukça hızlı eğitim ve tahmin süreleri elde edilmiştir. Modelin hiperparametre ayarları (örneğin, max_iter) ile performans daha da iyileştirilebilir.
  - **Custom Model:** Gradient descent tabanlı kendi Logistic Regression modelimiz, algoritmanın temel mekanizmasını anlamamıza yardımcı olurken, eğitim süresi bakımından kütüphane tabanlı modele göre daha yavaş kalabilmektedir. Bu modelin performansı da yeterli seviyede olsa, hiperparametre ayarları (learning_rate, num_iterations) üzerinde daha fazla çalışma yapılabilir.
- **Yorumlar ve İyileştirme Önerileri:**  
  - **Veri Ön İşleme:** Uygulanan eksik veri doldurma, one-hot encoding ve ölçeklendirme işlemleri modelin performansına önemli katkı sağlamıştır. Ancak, özellik mühendisliği aşamasında daha fazla çalışma (örneğin, yeni özellikler türetme veya gereksiz özellikleri eleme) modelin performansını artırabilir.
  - **Model Tuning:** Hem Scikit-learn modelinde hem de custom modelde, hiperparametre optimizasyonu (örneğin, grid search veya random search kullanarak) ile modelin tahmin performansı ve genel doğruluğu artırılabilir.
  - **İşletme Perspektifi:** Müşteri kaybını doğru tahmin etmek, işletmeler için büyük önem taşımaktadır. Doğru sınıflandırma, müşteriye yönelik müdahale stratejilerinin geliştirilmesine yardımcı olabilir. Bu nedenle, false negative (gerçek churn olan müşterinin yanlış sınıflandırılması) ve false positive (churn olmayan müşterinin yanlışlıkla churn olarak sınıflandırılması) oranlarının iş stratejisine etkileri dikkate alınarak model tercih edilebilir veya alternatif modeller denenebilir.
  - **Ek Değerlendirme Metrikleri:** ROC eğrisi, AUC değeri gibi ek metrikler, modelin farklı eşik değerlerindeki performansını değerlendirmek açısından faydalı olabilir. Bu metrikler, modelin duyarlılık (sensitivity) ve özgüllük (specificity) dengesini daha detaylı incelememizi sağlar.

Sonuç olarak, her iki yaklaşım da kendine özgü avantajlara sahiptir. Scikit-learn modeli hızlı ve kullanımı kolay bir çözüm sunarken, custom model algoritmanın temel mantığını daha iyi kavramamızı sağlamaktadır. İlerleyen çalışmalar için veri seti üzerinde daha kapsamlı özellik mühendisliği, hiperparametre optimizasyonu ve alternatif model yaklaşımlarının denenmesi, müşteri churn tahmininde daha yüksek performans elde edilmesini sağlayabilir.


  
