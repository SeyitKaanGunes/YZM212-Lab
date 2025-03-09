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
Veri seti, eksik değerler '?' ile belirtilmişse uygun şekilde doldurulmuş (sayısal özellikler için ortalama, kategorik özellikler için mod kullanılmıştır). Kategorik veriler için one hot encoding uygulanmamıştır çünkü seçilen özellikler sayısal veri içerir.
Veri, eğitim ve test setlerine ayrılmıştır.
Sadece modelde kullanılacak 5 sayısal özellik seçilmiştir.


Sonuçlar:
 Scikit-learn GaussianNB: doğruluk %80.01 , eğitim süresi 0.0026 saniye , tahmin süresi 0.0011 saniye.
 Custom GaussianNB : doğruluk %80.01 , eğitim süresi 0.0020  saniye ,  tahmin süresi 0.1795 saniye.         
 karmaşıklık matrisi her iki model için aynıdır. ([[7092 363] [1590 724]])

Kendi Yorumum : 
İki modelin doğruluğu aynıdır. Ancak custom modelin tahmin süresi daha uzundur.
Hedef sınıfların dağılımının dengesizliği (<=50K %75, >50K %25 civarı) performansı etkilemektedir.
Gelecekte farklı yöntemlerle daha iyi sonuçlar alınabilir.
Veri setinde `<=50K` sınıfı çok daha fazla örnek içerdiğinden, modelin `>50K` sınıfı için recall değeri düşebilmektedir. Bu durum, modelin az temsil edilen sınıfı tanımada zorlandığını göstermektedir.

kaynakça:
https://archive.ics.uci.edu/ml/datasets/Adult
https://scikit-learn.org/stable/modules/naive_bayes.html
https://github.com/git-guides/#learning-git-basics


Seyit Kaan Güneş 23291060