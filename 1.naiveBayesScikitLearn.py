import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

zip_path = r"C:\Users\Seyit Kaan\Desktop\adult.zip"
extract_path = r"C:\Users\Seyit Kaan\Desktop\extracted_adult"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

with zipfile.ZipFile(zip_path, 'r') as z:
    print("ZIP içeriği:", z.namelist())
    z.extract("adult.data", path=extract_path)

data_file = os.path.join(extract_path, "adult.data")

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv(data_file, header=None, names=col_names, na_values='?')

print("\nİlk 5 Satır:")
print(df.head())

print("\nVeri Seti Bilgisi:")
print(df.info())

print("\nİstatistiksel Özet:")
print(df.describe())

print("\nSınıf Dağılımı ('income' sütunu):")
print(df['income'].value_counts())

print("\nEksik Değer Sayıları:")
print(df.isnull().sum())

categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
target = 'income'

df[target] = df[target].apply(lambda x: 1 if x.strip() in ['>50K', '>50K.'] else 0)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n--- Scikit-learn GaussianNB ---")
gnb = GaussianNB()

start_train = time.time()
gnb.fit(X_train, y_train)
train_time_sklearn = time.time() - start_train

start_pred = time.time()
y_pred_sklearn = gnb.predict(X_test)
predict_time_sklearn = time.time() - start_pred

print("Eğitim Zamanı (Scikit-learn):", train_time_sklearn)
print("Tahmin Zamanı (Scikit-learn):", predict_time_sklearn)

acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print("Doğruluk (Scikit-learn):", acc_sklearn)
print("Classification Report (Scikit-learn):")
print(classification_report(y_test, y_pred_sklearn))

cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
print("Karmaşıklık Matrisi (Scikit-learn):\n", cm_sklearn)

plt.figure(figsize=(6, 5))
plt.imshow(cm_sklearn, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Scikit-learn GaussianNB - Karmaşıklık Matrisi")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['<=50K', '>50K'], rotation=45)
plt.yticks(tick_marks, ['<=50K', '>50K'])
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()
