# Assignment no. 9 Data Visualization on Titanic Dataset (Boxplot)

#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read Dataset
df=pd.read_csv(r"titanic.csv")
print("Titanic dataset is successfully loaded into Data Frame....")


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

#Fill the missing values
df ["Age"].fillna (df["Age"].median(), inplace=True)
print ("Null values are: \n",df.isnull().sum())

#Boxplot of 1-variable
fig, axes =plt.subplots(1,2)
fig.suptitle("Boxplot of 1-variables (Age & Fare)")
sns.boxplot (data =df, x="Age", ax=axes[0])
sns.boxplot (data =df, x="Fare",ax=axes[1])
plt.show()

#Boxplot of 2-variables
fig, axes =plt.subplots(2,2)
fig.suptitle("Boxplot of 2-variables")
sns.boxplot (data = df, x="Survived", y="Age", hue="Survived", ax=axes[0,0])
sns.boxplot (data = df, x="Survived", y="Fare", hue="Survived",ax=axes[0,1])
sns.boxplot (data = df, x="Sex", y="Age", hue= "Sex", ax=axes[1,0])
sns.boxplot (data = df, x="Sex", y="Fare", hue="Sex", ax=axes[1,1])
plt.show()

#Boxplot of 3-variables
fig, axes= plt.subplots(1,2)
fig.suptitle("Boxplot of 3-variables")
sns.boxplot (data=df, x="Sex", y="Age", hue="Survived", ax=axes[0])
sns. boxplot(data=df, x = "Sex", y="Fare",hue="Survived", ax=axes [1])
plt.show()