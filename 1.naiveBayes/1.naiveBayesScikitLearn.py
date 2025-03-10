
import os
import zipfile
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def load_and_preprocess_data(zip_filename, extract_to, csv_filename):

    zip_path = os.path.join(os.getcwd(), zip_filename)

    if not os.path.exists(extract_to):

        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zfile:

        zfile.extract(csv_filename, path=extract_to)

    data_path = os.path.join(extract_to, csv_filename)

    columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']

    adult_df = pd.read_csv(data_path, header=None, names=columns, na_values='?')

    for col in adult_df.select_dtypes(include=["object"]).columns:

        adult_df[col].fillna(adult_df[col].mode()[0], inplace=True)

    for col in adult_df.select_dtypes(include=["int64","float64"]).columns:

        adult_df[col].fillna(adult_df[col].mean(), inplace=True)

    return adult_df



df_adult = load_and_preprocess_data("adult.zip", "extracted_adult_unique", "adult.data")

selected_features = ['age','education-num','capital-gain','capital-loss','hours-per-week']

df_adult['income'] = df_adult['income'].apply(lambda x: 1 if x.strip() in ['>50K','>50K.'] else 0)

X = df_adult[selected_features]

y = df_adult['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_gnb = GaussianNB()

start_train = time.time()

model_gnb.fit(X_train, y_train)

train_duration = time.time() - start_train

start_predict = time.time()

y_predictions = model_gnb.predict(X_test)

predict_duration = time.time() - start_predict

print(train_duration, predict_duration, accuracy_score(y_test, y_predictions))

print(classification_report(y_test, y_predictions))

cm = confusion_matrix(y_test, y_predictions)

print(cm)

plt.figure(figsize=(6,5))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

plt.title("GaussianNB - Confusion Matrix")

plt.colorbar()

tick_labels = ['<=50K','>50K']

plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45)

plt.yticks(np.arange(len(tick_labels)), tick_labels)

plt.xlabel("Tahmin")

plt.ylabel("Gerçek")

plt.show()

