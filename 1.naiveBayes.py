import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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


print("\n--- Custom GaussianNB ---")


class CustomGaussianNB:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = X_c.mean(axis=0)
            self.vars[c] = X_c.var(axis=0)

    def gaussian_probability(self, x, mean, var):
        epsilon = 1e-9
        coeff = 1.0 / np.sqrt(2.0 * np.pi * (var + epsilon))
        exponent = np.exp(- ((x - mean) ** 2) / (2 * (var + epsilon)))
        return coeff * exponent

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian_probability(X[i], self.means[c], self.vars[c])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)


custom_gnb = CustomGaussianNB()

start_train = time.time()
custom_gnb.fit(X_train.values, y_train.values)
train_time_custom = time.time() - start_train

start_pred = time.time()
y_pred_custom = custom_gnb.predict(X_test.values)
predict_time_custom = time.time() - start_pred

print("Eğitim Zamanı (Custom):", train_time_custom)
print("Tahmin Zamanı (Custom):", predict_time_custom)

acc_custom = accuracy_score(y_test, y_pred_custom)
print("Doğruluk (Custom):", acc_custom)
print("Classification Report (Custom):")
print(classification_report(y_test, y_pred_custom))

cm_custom = confusion_matrix(y_test, y_pred_custom)
print("Karmaşıklık Matrisi (Custom):\n", cm_custom)


plt.figure(figsize=(6, 5))
plt.imshow(cm_custom, interpolation='nearest', cmap=plt.cm.Oranges)
plt.title("Custom GaussianNB - Karmaşıklık Matrisi")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['<=50K', '>50K'], rotation=45)
plt.yticks(tick_marks, ['<=50K', '>50K'])
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()
