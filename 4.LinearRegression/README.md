#  Linear Regression  

## Veri Seti

- **Kaynak:** Kaggle’daki “Diabetes” veri seti  
- **Gözlem Sayısı:** 442  
- **Özellikler (X):**  
  - `age` – Yaş (z-puanı ölçekli)  
  - `sex` – Cinsiyet (binary kodlu)  
  - `bmi` – Vücut Kitle İndeksi  
  - `bp` – Ortalama Kan Basıncı  
  - `s1…s6` – 6 adet biyokimyasal ölçüm (hepsi standartlaştırılmış)  
- **Hedef Değişken (y):**  
  - `target` – 1 yıl sonraki diyabet ilerleme skoru (sayısal)  
- **Önişleme:** Tüm özellikler ortalama = 0, std ≈ 0.05 olacak şekilde z-score normalizasyonu uygulanmıştır.

---

## Amaç

Bu laboratuvarda Lineer Regresyon yöntemini kullanarak:

1. **Özellikler ile hedef değişken arasındaki doğrusal ilişkiyi** keşfetmek,  
2. Katsayı tahminini üç farklı yaklaşımla (Normal Denklem, Gradient Descent, scikit-learn) yapmak,  
3. Her modelin performansını **MSE (Mean Squared Error)** cinsinden hesaplayıp karşılaştırmak,  

hedeflenmektedir.  



## Cost Fonksiyonları

Modelin tahmin performansını ölçmek ve optimizasyon sürecini yönlendirmek için iki temel maliyet fonksiyonu kullanıyoruz:

### 1. Mean Squared Error (MSE)

$$
J_{\mathrm{MSE}}(\theta)
= \frac{1}{m}\sum_{i=1}^{m}\bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^{2}
$$

- **$m$**: Eğitim örneklerinin sayısı  
- **$h_\theta(x) = \theta^\top x$**: Modelin doğrusal tahmin fonksiyonu  
- **$y^{(i)}$**: Gerçek hedef değeri  

### 2. Gradient Descent için Cost (Yarı MSE)

$$
J(\theta)
= \frac{1}{2m}\sum_{i=1}^{m}\bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^{2}
$$

- Bu form, **Gradient Descent** algoritmasında türev hesaplamayı basitleştirmek için tercih edilir.  
- **Gradyan**:
  $$
  \nabla_\theta J(\theta)
  = \frac{1}{m}\,X^\top\bigl(X\theta - y\bigr)
  $$
- **Güncelleme Adımı**:
  $$
  \theta \;\gets\; \theta - \alpha\,\nabla_\theta J(\theta)
  $$

---

- **MSE**: Farklı modellerin nihai performansını karşılaştırmak için  
- **$J(\theta)$**: Gradient Descent sürecinin yakınsamasını izlemek ve optimize etmek için kullanılır  
 

## Model Karşılaştırması

| Yöntem                                    | Test MSE    | Cost = MSE/2 |
|-------------------------------------------|-------------|--------------|
| **Closed-Form (Normal Denklem)**          | 2 900.19    | 1 450.10     |
| **scikit-learn `LinearRegression`**       | 2 900.19    | 1 450.10     |
| **Gradient Descent (α=0.01, 1 000 it.)**  | 4 716.46    | 2 358.23     |

---

## Yorumlar - Öneriler

- **Closed-Form & scikit-learn** aynı katsayıları buldu: Test MSE ve Cost eşleşiyor.  
- **Gradient Descent** henüz optimuma tam yakınsamamış;  
  - İterasyon sayısını (≥ 5 000)  veya  
  - Learning rate’i (0.001–0.05)  
  değiştirerek **Cost** düşürülebilir.  


## Kaynakça

1. **Kaggle.** [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) 
2. **Scikit-learn Documentation.** *LinearRegression* (Erişim: 06 Mayıs 2025)  
3. **GeeksforGeeks.** “Solving Linear Regression without using sklearn and tensorflow” (Erişim: 05 Mayıs 2025)  

