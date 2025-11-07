from sklearn.preprocessing import Normalizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("sales_data_sample.csv")
df.head()
df.dtypes
df.isnull().sum()
df.info()
plt.figure(figsize=(30, 26))
sns.heatmap(df.corr(), annot=True)
df_drop = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS', 'POSTALCODE', 'CITY',
           'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME',
           'CUSTOMERNAME', 'ORDERNUMBER']
df = df.drop(df_drop, axis=1)
df.head()
df.shape
df.isnull().sum()
df.dtypes
country = pd.get_dummies(df['COUNTRY'])
productline = pd.get_dummies(df['PRODUCTLINE'])
Dealsize = pd.get_dummies(df['DEALSIZE'])
df = pd.concat([df, country, productline, Dealsize], axis=1)
df.head()
df_drop = ['COUNTRY', 'PRODUCTLINE', 'DEALSIZE']
df = df.drop(df_drop, axis=1)
df.dtypes
df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes
df.dtypes
df.drop('ORDERDATE', axis=1, inplace=True)
df.dtypes
WCSS = []  # Withhin Cluster Sum of Squares from the centroid
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
kmeanModel = KMeans(n_clusters=3)
y_kmeans = kmeanModel.fit_predict
plt.scatter(df['y'])
print(y_kmeans)

plt.figure(figsize=(30, 26))
sns.heatmap(df.corr(), annot=True)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 0), timings=False)
visualizer.fit(df)
visualizer.show()
df.head()
df_scaled = Normalizer(df)
df_x = pd.DataFrame(df_scaled, columns=df.columns)
