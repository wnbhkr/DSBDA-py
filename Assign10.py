# Assignment no, 10 Data Visualization on IRIS Dataset (Histogram & Boxplot)

# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read Dataset
df=pd.read_csv(r"iris - 2023-05-20T153708.481.csv")
print("iris dataset is successfully loaded into Data Frame....")


#Display information of dataset
print( "Information of Dataset: \n", df.info)
print( "Shape of Dataset (row x column):",df.shape)
print( "Columns Name:", df.columns)
print( "Total elements in dataset:", df.size)
print( "Datatype of attributes (columns):", df.dtypes)
print( "First 5 rows: \n", df.head().T)
print( "Last 5 rows:\n",df.tail().T)
print( "Any 5 rows: \n",df.sample(5).T)

#Find missing values
print ("Missing values")
print(df.isnull().sum())

#Histogram of 1-variable
fig, axes= plt.subplots(2,2)
fig.suptitle("Histogram 1-variables")
sns.histplot(data=df,x="sepal.length", ax=axes[0,0])
sns.histplot(data=df,x="sepal.width", ax=axes [0,1])
sns.histplot(data=df,x="petal.length", ax=axes[1,0])
sns.histplot(data=df,x="petal.width", ax=axes[1,1])
plt.show()

#Histogram of 2-variables
fig, axes =plt.subplots(2,2)
fig.suptitle("Histogram of 2-variables")
sns.histplot(data=df, x="sepal.length",hue="variety", multiple="dodge", ax=axes[0,0])
sns.histplot(data=df, x="sepal.width",hue="variety", multiple="dodge", ax=axes[0,1])
sns.histplot(data=df, x="petal.length",hue="variety", multiple ="dodge",ax=axes[1,0])
sns.histplot(data=df, x="petal.width",hue="variety", multiple="dodge", ax=axes[1,1])
plt.show()

#Boxplot of 1-variable.
fig, axes= plt.subplots(2,2)
fig.suptitle("Boxplot of 1-variables")
sns.histplot(data=df,x="sepal.length", ax=axes[0,0])
sns.histplot(data=df,x="sepal.width", ax=axes [0,1])
sns.histplot(data=df,x="petal.length", ax=axes[1,0])
sns.histplot(data=df,x="petal.width", ax=axes[1,1])
plt.show()


#Boxplot of 2-variables
fig, axes =plt.subplots(2,2)
fig.suptitle("Histogram of 2-variables")
sns.histplot(data=df, x="sepal.length",y="variety", hue="variety", ax=axes[0,0])
sns.histplot(data=df, x="sepal.width",y="variety", hue="variety", ax=axes[0,1])
sns.histplot(data=df, x="petal.length",y="variety", hue="variety",ax=axes[1,0])
sns.histplot(data=df, x="petal.width",y="variety", hue="variety", ax=axes[1,1])
plt.show()
