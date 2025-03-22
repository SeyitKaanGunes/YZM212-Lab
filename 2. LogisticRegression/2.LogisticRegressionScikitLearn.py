import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("telco_final.csv")


print("Initial Data Info:")
print(df.info())


if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

print("\nChurn sütunu benzersiz değerleri (doldurmadan önce):", df['Churn'].unique())
print("Churn sütunundaki eksik değer sayısı (doldurmadan önce):", df['Churn'].isnull().sum())

if df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].fillna('No')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
else:
    df['Churn'] = df['Churn'].fillna(0)

print("\nChurn sütunu benzersiz değerleri (dönüşüm sonrası):", df['Churn'].unique())
print("Churn sütunundaki eksik değer sayısı (dönüşüm sonrası):", df['Churn'].isnull().sum())

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn', axis=1).values
y = df['Churn'].values

if np.isnan(y).any():
    print("Hata: y değişkeninde hâlâ eksik değer var!")
else:
    print("y değişkeninde eksik değer yok.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start_time = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

start_time = time.time()
y_pred = model.predict(X_test_scaled)
test_time = time.time() - start_time

print("\nScikit-learn Model Eğitimi Süresi: {:.4f} saniye".format(train_time))
print("Scikit-learn Model Tahmin Süresi: {:.4f} saniye".format(test_time))
print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Scikit-learn Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.colorbar()
plt.show()
