import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.info())

if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

print("Sınıf Dağılımı:")
print(df['Churn'].value_counts())

min_class = df['Churn'].value_counts().min()
df_balanced = df.groupby('Churn').sample(n=min_class, random_state=42).reset_index(drop=True)
print("Dengelenmiş Sınıf Dağılımı:")
print(df_balanced['Churn'].value_counts())


selected_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
df_selected = df_balanced[selected_columns]

df_selected['TotalCharges'] = pd.to_numeric(df_selected['TotalCharges'], errors='coerce')
df_selected = df_selected.dropna()

df_selected['TotalCharges_log'] = np.log(df_selected['TotalCharges'] + 1)
df_selected = df_selected.drop(columns=['TotalCharges'])

scaler = StandardScaler()
df_selected['TotalCharges_log'] = scaler.fit_transform(df_selected[['TotalCharges_log']])

print(df_selected.head())

plt.figure(figsize=(10,6))
sns.histplot(df_selected['TotalCharges_log'], bins=30, kde=True, edgecolor='black')
plt.title("Log Dönüşümü Yapılmış TotalCharges Dağılımı")
plt.xlabel("TotalCharges Log Değeri")
plt.ylabel("Frekans")
plt.grid(True)
plt.show()

df_selected.to_csv("telco_final.csv", index=False)
