# İleri ve Geri Yayılım Algoritmaları ile Sinir Ağı Uygulaması

## Proje Hakkında

Bu projede, sinir ağları için temel yapı taşları olan ileri yayılım (forward propagation) ve geri yayılım (backward propagation) algoritmaları sıfırdan uygulanmıştır. Hazır kütüphaneler kullanılmadan, NumPy ile matematiksel işlemler gerçekleştirilmiş ve bir sınıflandırma problemi çözülmüştür.

## Veri Seti

Projede, sınıflandırma problemi için UCI Machine Learning Repository'den alınan Iris veri seti kullanılmıştır. Bu veri seti:

- 150 örnek içerir
- Her örnek için 4 özellik vardır: sepal uzunluğu, sepal genişliği, petal uzunluğu ve petal genişliği
- 3 farklı iris çiçeği türünü sınıflandırır: Iris Setosa, Iris Versicolor ve Iris Virginica

Veri seti, sinir ağı modelimizin performansını test etmek için ideal bir başlangıç noktasıdır çünkü görece küçük ancak çok sınıflı bir sınıflandırma problemi sunar.

## Teorik Altyapı

### Sinir Ağları

Yapay sinir ağları, insan beyninin çalışma prensibinden esinlenerek geliştirilmiş makine öğrenmesi modellerdir. Temel yapı taşları nöronlardır ve bu nöronlar katmanlar halinde düzenlenir:

1. **Giriş Katmanı**: Veri özelliklerini alır
2. **Gizli Katmanlar**: Verideki karmaşık ilişkileri öğrenir
3. **Çıkış Katmanı**: Tahminleri üretir

### İleri Yayılım (Forward Propagation)

İleri yayılım, giriş verilerinin ağ boyunca ileriye doğru işlenmesi sürecidir:

1. Her nöron, önceki katmandaki nöronlardan gelen ağırlıklı toplamı alır
2. Bu toplama bir bias (yanlılık) değeri eklenir
3. Sonuç, bir aktivasyon fonksiyonundan geçirilir
4. Çıktı, bir sonraki katmana iletilir

Matematiksel olarak, her katman için:

```
Z[l] = W[l] * A[l-1] + b[l]
A[l] = g(Z[l])
```

Burada:
- Z[l]: l. katmanın doğrusal çıktısı
- W[l]: l. katmanın ağırlık matrisi
- A[l-1]: (l-1). katmanın aktivasyonları
- b[l]: l. katmanın bias vektörü
- g(): Aktivasyon fonksiyonu
- A[l]: l. katmanın aktivasyonları

### Geri Yayılım (Backward Propagation)

Geri yayılım, ağın parametrelerini (ağırlıklar ve bias değerleri) güncellemek için kullanılan bir algoritmadır:

1. Çıkış katmanındaki hata hesaplanır
2. Hata, ağ boyunca geriye doğru yayılır
3. Her katmandaki parametrelerin gradyanları hesaplanır
4. Parametreler, gradyan iniş algoritması kullanılarak güncellenir

Matematiksel olarak:

```
dZ[L] = A[L] - Y  (çıkış katmanı için)
dW[L] = (1/m) * dZ[L] * A[L-1].T
db[L] = (1/m) * sum(dZ[L])

dZ[l] = W[l+1].T * dZ[l+1] * g'(Z[l])  (gizli katmanlar için)
dW[l] = (1/m) * dZ[l] * A[l-1].T
db[l] = (1/m) * sum(dZ[l])
```

Burada:
- dZ[l]: l. katmanın hata terimi
- dW[l]: l. katmanın ağırlık gradyanları
- db[l]: l. katmanın bias gradyanları
- g'(): Aktivasyon fonksiyonunun türevi
- m: Örnek sayısı

### Aktivasyon Fonksiyonları

Projede kullanılan aktivasyon fonksiyonları:

1. **Sigmoid**: f(x) = 1 / (1 + e^(-x))
   - Türevi: f'(x) = f(x) * (1 - f(x))
   - Genellikle gizli katmanlarda kullanılır

2. **ReLU (Rectified Linear Unit)**: f(x) = max(0, x)
   - Türevi: f'(x) = 1 if x > 0 else 0
   - Derin ağlarda vanishing gradient problemini azaltır

3. **Tanh (Hiperbolik Tanjant)**: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Türevi: f'(x) = 1 - f(x)^2
   - Çıktıları -1 ile 1 arasında normalize eder

4. **Softmax**: f(x_i) = e^(x_i) / sum(e^(x_j))
   - Çok sınıflı sınıflandırma problemlerinde çıkış katmanında kullanılır
   - Her sınıf için olasılık dağılımı üretir

### Kayıp Fonksiyonları

1. **Ortalama Kare Hata (MSE)**: L = (1/2m) * sum((y_pred - y_true)^2)
   - Regresyon problemlerinde kullanılır

2. **Çapraz Entropi**: L = -(1/m) * sum(y_true * log(y_pred))
   - Sınıflandırma problemlerinde kullanılır
   - Softmax aktivasyonu ile birlikte kullanıldığında etkilidir

### Gradyan İniş

Gradyan iniş, kayıp fonksiyonunu minimize etmek için parametreleri güncelleme yöntemidir:

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

Burada öğrenme oranı (learning rate), adım büyüklüğünü kontrol eden hiperparametredir.

## Uygulama Detayları

### Model Mimarisi

Projede uygulanan sinir ağı modeli şu özelliklere sahiptir:

- Giriş katmanı: 4 nöron (Iris veri setindeki özellik sayısı)
- Gizli katman: 8 nöron (sigmoid aktivasyonu)
- Çıkış katmanı: 3 nöron (softmax aktivasyonu, her bir Iris türü için)

### Hiperparametreler

- Öğrenme oranı: 0.1
- Epoch sayısı: 1000

### Veri Ön İşleme

1. Özellikler standartlaştırılmıştır (ortalama=0, standart sapma=1)
2. Hedef değişken one-hot encoding ile kodlanmıştır
3. Veri seti %80 eğitim, %20 test olarak bölünmüştür

## Sonuçlar ve Değerlendirme

### Model Performansı

Eğitilen sinir ağı modeli, test veri seti üzerinde mükemmel bir performans göstermiştir:

- Doğruluk (Accuracy): %100
- Hassasiyet (Precision): %100
- Duyarlılık (Recall): %100
- F1 Skoru: %100

### Karmaşıklık Matrisi

Karmaşıklık matrisi, modelin her sınıf için tahmin performansını gösterir. Test veri setindeki tüm örnekler doğru sınıflandırılmıştır.

### Eğitim Süreci

Loss-epoch eğrisi, modelin eğitim sürecinde kaybın sürekli azaldığını göstermektedir. Bu, modelin veri setindeki desenleri başarıyla öğrendiğini gösterir.

### Karar Sınırları

Karar sınırı görselleştirmeleri, modelin farklı özellik çiftleri için nasıl sınıflandırma yaptığını gösterir. Özellikle petal uzunluğu ve genişliği, türleri ayırt etmede en etkili özelliklerdir.

## Sonuç

Bu projede, ileri ve geri yayılım algoritmalarını kullanarak sıfırdan bir sinir ağı uygulaması geliştirilmiştir. Model, Iris veri seti üzerinde mükemmel bir sınıflandırma performansı göstermiştir. Uygulama, sinir ağlarının temel bileşenlerini ve çalışma prensiplerini anlamak için değerli bir örnek sunmaktadır.

Projenin başarısı, sinir ağlarının basit sınıflandırma problemlerinde ne kadar etkili olabileceğini göstermektedir. Daha karmaşık veri setleri ve problemler için, model mimarisi ve hiperparametreler üzerinde daha fazla deneme yapılabilir.
