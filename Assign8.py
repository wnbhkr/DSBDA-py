#Assignment no. 8 Data Visualization on Titanic Dataset (Histogram)
# Program by Prof. Rahul Dagade & Prof. Varsha Nale

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

#Histogram of 1-variable
fig, axes= plt.subplots(1,2)
fig. suptitle(" Histogram 1-variables (Age & Fare)")
sns. histplot(data=df, x="Age", ax=axes[0])
sns. histplot(data=df, x="Fare", ax=axes [1])
plt.show()

#Histogram of 2-variables
fig, axes =plt.subplots(2,2)
fig.suptitle( "Histogram of 2-variables")
sns.histplot(data = df, x="Age",hue="Survived", multiple="dodge", ax=axes[0,0])
sns.histplot(data = df, x="Fare",hue="Survived", multiple="dodge", ax=axes[0,1])
sns.histplot(data = df, x="Age",hue="Sex", multiple= "dodge", ax=axes[1,0])
sns.histplot(data = df, x="Fare", hue="Sex", multiple="dodge", ax=axes [1, 1])
plt.show()