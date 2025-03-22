import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

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


class CustomLogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0

        for i in range(self.num_iterations):
            Z = np.dot(X, self.W) + self.b
            A = self.sigmoid(Z)
            dw = (1 / self.m) * np.dot(X.T, (A - y))
            db = (1 / self.m) * np.sum(A - y)
            self.W -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict_proba(self, X):
        Z = np.dot(X, self.W) + self.b
        return self.sigmoid(Z)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)


custom_model = CustomLogisticRegression(learning_rate=0.1, num_iterations=2000)
start_time = time.time()
custom_model.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

start_time = time.time()
y_pred_custom = custom_model.predict(X_test_scaled)
test_time = time.time() - start_time

print("\nCustom Model Eğitimi Süresi: {:.4f} saniye".format(train_time))
print("Custom Model Tahmin Süresi: {:.4f} saniye".format(test_time))
print("\nKarmaşıklık Matrisi (Custom):")
print(confusion_matrix(y_test, y_pred_custom))
print("\nSınıflandırma Raporu (Custom):")
print(classification_report(y_test, y_pred_custom))

cm_custom = confusion_matrix(y_test, y_pred_custom)
plt.imshow(cm_custom, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Custom Logistic Regression Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.colorbar()
plt.show()
