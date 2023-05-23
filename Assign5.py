import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r"Social_Network_Ads.csv")
print(df)

print(df.info())
print(df.size)
print(df.shape)
print(df.columns)
print(df.head())
print(df.tail())
print(df.sample())
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
print(df.isna().sum())

sns.heatmap(df.corr(), annot=True)
plt.show()

l = ['Age', 'EstimatedSalary', 'Purchased']
for i in l:
    sns.boxplot(df[i])
    plt.show()

x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=0)

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, linewidths=.3)
plt.show()