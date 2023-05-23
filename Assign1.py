# Assignment No. 1 Data Wrangling I:
# Program by Prof. Rahul Dagade & Prof. Varsha Nale
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read Dataset
df=pd.read_csv(r"placement_data.csv")
print("Placement dataset is successfully loaded into Data Frame....")

#Display information of dataset
print( "Information of Dataset: \n", df.info)
print( "Shape of Dataset (row x column):",df.shape)
print( "Columns Name:", df.columns)
print( "Total elements in dataset:", df.size)
print( "Datatype of attributes (columns):", df.dtypes)
print( "First 5 rows: \n", df.head().T)
print( "Last 5 rows:\n",df.tail().T)
print( "Any 5 rows: \n",df.sample(5).T)

#Display statistical information of dataset
print( "Statistical information of Numerical Columns: \n", df.describe())

#Display null values
print("Total Number of Null Values in Dataset:", df.isna().sum())

# Data type conversion
print("Converting Data type of variables: ")
df["sl_no"]=df[ "sl_no"].astype( "int8")
print("Check Datatype of sl_no",df.dtypes)
df[ "ssc_p"]=df[ "ssc_p"].astype("int8")
print("Check Datatype of ssc_p",df.dtypes)

# Label Encoding Conversion of Categorical to Quantitative
print("Encoding using Label Encoding (Cat Codes)")
df["gender"] =df["gender"].astype("category")
print("Data types of Gender=", df.dtypes["gender"])
df["gender"]=df["gender"].cat.codes
print("Data types of gender after label encoding =", df. dtypes['gender'])
print("Gender Values:",df["gender"].unique())
# Normalization
print("Normalization using Min-Max Feature Scaling: ")
df["salary"]=(df["salary"]-df["salary"].min())/(df ["salary"].max()-df["salary"].min())
print(df.head().T)
