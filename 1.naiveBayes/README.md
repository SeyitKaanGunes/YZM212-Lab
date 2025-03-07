Naive Bayes Projesi

Problem Tanımı : 
Bu projede, yetişkinlerin gelir seviyelerini (50K doların altında veya üstünde) tahmin etmek için Gaussian Naive Bayes algoritması kullanılmıştır. Uygulama iki farklı şekilde gerçekleştirilmiştir:
1-Scikit-learn kütüphanesi ile hazır GaussianNB modeli,
2-Python ile elle yazılmış Gaussian Naive Bayes algoritması.

Veri seti:
Kullanılan veri seti Adult veri setidir.
Toplam veri sayısı: 32,561.
Sayısal Özellikler: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
Hedef Değişken: `income` (<=50K, >50K)
Eksik veri içermemektedir.

Sonuçlar:
 Scikit-learn GaussianNB: doğruluk %80.01 , eğitim süresi 0.0026 saniye , tahmin süresi 0.0011 saniye.
 Custom GaussianNB : doğruluk %80.01 , eğitim süresi 0.0020  saniye ,  tahmin süresi 0.1795 saniye.         
 karmaşıklık matrisi her iki model için aynıdır. ([[7092 363] [1590 724]])

Kendi Yorumum : 
İki modelin doğruluğu aynıdır. Ancak custom modelin tahmin süresi daha uzundur.
Hedef sınıfların dağılımının dengesizliği (<=50K %75, >50K %25 civarı) performansı etkilemektedir.
Gelecekte farklı yöntemlerle daha iyi sonuçlar alınabilir.


Seyit Kaan Güneş 23291060