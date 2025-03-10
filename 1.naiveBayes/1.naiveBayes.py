
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

zip_file_path = os.path.join(os.getcwd(), "adult.zip")
extracted_data_folder = os.path.join(os.getcwd(), "adult_extracted")

if not os.path.exists(extracted_data_folder):
    os.makedirs(extracted_data_folder)

with zipfile.ZipFile(zip_file_path, 'r') as zf:
    print("ZIP İçeriği:", zf.namelist())
    zf.extract("adult.data", path=extracted_data_folder)

data_file_path = os.path.join(extracted_data_folder, "adult.data")

column_names = [
    'age','workclass','fnlwgt','education','education-num',
    'marital-status','occupation','relationship','race','sex',
    'capital-gain','capital-loss','hours-per-week','native-country','income'
]

df = pd.read_csv(data_file_path, header=None, names=column_names, na_values='?')

print("İlk 5 Satır:\n", df.head())

for c in df.select_dtypes(include=["object"]).columns:
    df[c].fillna(df[c].mode()[0], inplace=True)

for c in df.select_dtypes(include=["int64","float64"]).columns:
    df[c].fillna(df[c].mean(), inplace=True)

features = ['age','education-num','capital-gain','capital-loss','hours-per-week']

df['income'] = df['income'].apply(lambda x: 1 if x.strip() in ['>50K','>50K.'] else 0)

X = df[features]

y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class SimpleGaussianNB:

    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / len(X)
            self.class_means[c] = X_c.mean(axis=0)
            self.class_vars[c] = X_c.var(axis=0)

    def _gauss_prob(self, x, m, v):
        eps = 1e-9
        coef = 1 / np.sqrt(2 * np.pi * (v + eps))
        exp_val = np.exp(-((x - m) ** 2) / (2 * (v + eps)))
        return coef * exp_val

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            row = X.iloc[i].values
            posterior_list = []
            for c in self.classes:
                prior = np.log(self.class_priors[c])
                likelihood = np.sum(np.log(self._gauss_prob(row, self.class_means[c], self.class_vars[c])))
                posterior_list.append(prior + likelihood)
            preds.append(self.classes[np.argmax(posterior_list)])
        return np.array(preds)

model = SimpleGaussianNB()

start_train = time.time()

model.fit(X_train, y_train)

train_time = time.time() - start_train

start_pred = time.time()

y_pred = model.predict(X_test)

pred_time = time.time() - start_pred

print("\nEğitim süresi:", train_time)

print("Tahmin süresi:", pred_time)

print("Doğruluk:", accuracy_score(y_test, y_pred))

print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Karmaşıklık Matrisi:\n", cm)

plt.figure(figsize=(6,5))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)

plt.title("SimpleGaussianNB - Confusion Matrix")

plt.colorbar()

plt.xticks([0,1], ['<=50K','>50K'], rotation=45)

plt.yticks([0,1], ['<=50K','>50K'])

plt.xlabel("Tahmin")

plt.ylabel("Gerçek")

plt.show()

