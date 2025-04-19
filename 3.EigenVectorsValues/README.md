#  SORU 1-Matris Manipülasyonu, Özdeğerler ve Özvektörlerin Makine Öğrenmesiyle İlişkisi

## 1)Matris Manipülasyonu

Makine öğrenmesinde matris manipülasyonu, verileri ve model parametrelerini matrisler halinde tutup üzerinde temel cebirsel işlemler (çarpma, toplama, transpoze, inversiyon) ve ayrıştırma (örneğin SVD, özdeğer-özvektör dekompozisyonu) uygulayarak;

-   Özellik uzayını dönüştürme ve boyut indirgeme (PCA, LDA)
    
-   Ağır­lık güncellemeleri ve hata yayılımı (backpropagation)
    
-   Veri ilişkileri ve kovaryans matrislerinin hesaplanması  
    gibi adımları gerçekleştirir.

## 2)Özdeğer (Eigenvalue)

Bir kare matris A için
Av=λv
eşitliğini sağlayan skaler  λ’ya **özdeğer** denir. Yani, bazı özel vektörleri (v, özvektör) yalnızca ölçeklendirip yönünü değiştirmeden çarpan katsayıdır.
Makine öğrenmesinde ise boyut indirgeme, öznitelik seçimi , spektral kümeleme ,gürültü Filtrasyonu ve Regularizasyonu gibi matrix manipülasyonu işlemlerinde kullanılır.

## 3)Özvektör(Eigenvector)
Bir kare matris A için sıfır vektöründen farklı olan ve
Av=λv
eşitliğini sağlayan v  vektörüne **özvektör** denir. Burada λ ilgili özdeğerdir.
Makine öğrenmesinde ise boyut indirgeme, öznitelik seçimi , spektral kümeleme gibi alanlarda kullanılır.

# Kullanılan yöntemler :
 **PCA (Ana Bileşen Analizi):**  
Verideki en büyük “dalgalanmaları” (farklılıkları) yakalayıp veriyi daha küçük boyutta ifade etmeye yarar. Yani, orijinal karmaşık veri setini, bilgi kaybını en aza indirerek daha sade bir görünümle temsil eder.

**SVD (Tekil Değer Ayrıştırması):**  
Elinizdeki herhangi bir veri tablosunu, temel yapı taşlarına (“temel desenlere”) ayırır. Böylece hem gürültüyü (önemsiz ayrıntıları) azaltabilir hem de verinin asıl kalıplarını ön plana çıkarabilirsiniz.

**LDA (Lineer Ayrım Analizi):**  
Etiketli verideki farklı grupları (örneğin “kedi” vs. “köpek”) en net ayıracak biçimde yeni eksenler bulur. Böylece gruplar arasında karışıklığı en aza indirerek sınıflandırmayı kolaylaştırır.    

**Spektral Kümeleme:**  
Veriyi bir ağ (graf) gibi düşünür; her nokta aralarındaki bağlantı güçlü olanlarla bir arada toplanır. Karmaşık ilişki ağlarından birden çok küme çıkararak benzerlikleri gruplar hâline getirir.

**Kernel PCA:**  
PCA’yı düz hatlarla ayıramayacağınız karmaşık verilerde de kullanabilmek için veriyi “gizli bir alana” taşır. Orada PCA’yı uygulayıp sonra tekrar geri getirerek, doğrusal olmayan kalıpları da yakalar.

**LSI (Gizli Anlamsal İndeksleme):**  
Metin madenciliğinde, belge-terim ilişkilerini örüntüleriyle birlikte inceler. Böylece farklı kelime ve belgeler arasındaki “örtük” (gizli) bağlantıları fark edip metinleri anlam temelli gruplara ayırır.


Her yöntemin ortak amacı, veri tablosunun içindeki en önemli düzeni bulup öne çıkarmak; gereksiz ayrıntıları ise geri planda tutarak modelleri hem daha hızlı hem de genelleyici hâle getirmektir.

Kaynakça:
-   Brownlee, Jason. “Introduction to Matrices and Matrix Arithmetic for Machine Learning.” _Machine Learning Mastery_. Published 3.5 years ago. (https://www.machinelearningmastery.com/introduction-matrices-machine-learning)
    
-   Brownlee, Jason. “A Gentle Introduction to Matrix Operations for Machine Learning.” _Machine Learning Mastery_. Published 5.7 years ago.(https://www.machinelearningmastery.com/matrix-operations-for-machine-learning)
    
-   Brownlee, Jason. “Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning.” _Machine Learning Mastery_. Published 5.7 years ago. (https://www.machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors)
    
-   Brownlee, Jason. “Principal Component Analysis for Visualization.” _Machine Learning Mastery_. Published 3.5 years ago. (https://www.machinelearningmastery.com/principal-component-analysis-for-visualization)
    
-   Brownlee, Jason. “How to Calculate the SVD from Scratch with Python.” _Machine Learning Mastery_. Published 5.5 years ago.(https://www.machinelearningmastery.com/singular-value-decomposition-for-machine-learning)
    
-   “Linear Discriminant Analysis in Machine Learning.” _GeeksforGeeks_. Published February 2025. (https://www.geeksforgeeks.org/ml-linear-discriminant-analysis)
     
-   “Introduction to Kernel PCA.” _GeeksforGeeks_. Published 2 years ago. (https://www.geeksforgeeks.org/ml-introduction-to-kernel-pca)
    


#  SORU 2- np.linalg.eig dokümantasyon Özeti

-   **Parametre**
    
    -   `a` : (..., M, M) boyutlarında kare dizi.
        
-   **Dönüş**
    
    -   `eigenvalues` : (..., M) boyutunda özdeğerler dizisi.
        
    -   `eigenvectors` : (..., M, M) boyutunda, her sütunu bir özvektör olacak şekilde normalleştirilmiş özvektörler dizisi.
        
-   **Hata**
    
    -   Eğer algoritma yakınsama sağlamazsa `LinAlgError` fırlatılır.
        
-   **Not**
    
    -   Gerçek değerli giriş için özdeğerler reel veya eşlenik çiftler olarak döner.


## kaynak kod

Fonksiyon, **Python** tarafında `numpy.linalg._linalg` modülünde tanımlı.

-   `a`, `_makearray` ile `ndarray`’e çevirilir,
    
-   `_assert_stacked_2d` ve `_assert_stacked_square` ile gerçekten 2D ve kare olduğundan emin olunur.

-   `_commonType` ile içerikteki tip (real/complex) tespit edilir,
    
-   Buna göre LAPACK çağrısının hangi veri tipini (ör. `d->d` veya `D->D`) kullanacağı belirlenir.

w, v = _umath_linalg.eig(a, signature=signature)
Burada `_umath_linalg.eig`, C++’ta yazılmış `umath_linalg.cpp` içindeki wrapper aracılığıyla  
Fortran’daki `dgeev` (real) veya `zgeev` (complex) LAPACK rutinine yönlendirme yapar.

-   Dönen `w` (özdeğerler) ve `v` (özvektörler), `result_t` tipine cast edilip
    
-   `wrap` fonksiyonu ile orijinal array durumuna (örneğin `matrix` nesnesi ise matrix olarak) çevrilir.
    
-   Python’a `(eigenvalues, eigenvectors)` tuple’ı olarak iletilir.

 -   Eğer LAPACK rutinleri yakınsama sağlayamazsa, otomatik olarak `LinAlgError` fırlatılır.

# SORU 3 - Saf Python ile Özdeğer Hesaplaması ve NumPy Karşılaştırması

Aşağıdaki adımları izleyerek, “LucasBN/Eigenvalues-and-Eigenvectors” deposundaki saf Python kodunu referans alıp yeniden uyguladım ve elde ettiğim sonuçları NumPy’nın `np.linalg.eig` fonksiyonuyla karşılaştırdım:

`find_determinant` fonksiyonu, verilen kare matrisin determinantını özyinelemeli kofaktör açılımı ile hesaplıyor.

`characteristic_equation` ve `determinant_equation` fonksiyonları, matrisi (A−λI) biçimine çevirip bu matrisin determinantını bir polinom (karakteristik denklem) katsayı listesi olarak üretiyor.

`find_eigenvalues` ise bu polinomun köklerini `numpy.roots` ile bularak özdeğerleri elde ediyor.

Yukarıdaki işlevlerin tümünü tek bir Python dosyasına kopyaladım.

Dosyada ayrıca `compare_with_numpy` adında bir yardımcı fonksiyon oluşturdum; bu fonksiyon hem saf Python hem de NumPy yöntemlerini aynı matris üzerinde çalıştırıp sonucu döndürüyor.

Sonuç : 

Custom eigenvalues: [7. 5. 3.]

NumPy eigenvalues:  [5. 3. 7.]


Saf Python yöntemiyle elde edilen {7,  5,  3} değerleri, NumPy’nın döndürdüğü {5,  3,  7} ile aynı küme hâlindedir.

Aralarındaki tek fark, NumPy’nın sıralamayı farklı bir permütasyonla vermesidir ki özdeğerler kümesi açısından hiçbir işlevsel fark yaratmaz.


Seyit Kaan GÜNEŞ 23291060
