#Assignment number: 3 Statistical Information #Program by Prof. Rahul Dagade & Prof. Varsha Nale
#import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Read dataset
df = pd.read_csv(r"C:\Users\niraj\Downloads\Employee_Salary.csv")
print("Employee_Salary Dataset is successfully loaded.\n")

#Display information of dataset
print("Iris Dataset is successfully loaded...")
print("Information of Dataset:\n", df.info)
print("Shape of Dataset (row x column):", df.shape)
print("Columns Name: df.columns")
print("Total elements in dataset:", df. size)
print("Datatype of attributes (columns):", df.dtypes)
print("First 5 rows: \n", df.head().T)
print("Last 5 rows: \n", df.tail().T)
print("Any 5 rows: \n", df.sample(5).T)

print("Statistical information of Numerical Columns: \n",)
columns = ["Experience_Years", "Age", "Salary"]
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format("Columns", "Min","Max","Mean","Median","STD"))
for column in columns:
    m1,m2,m3=df[column].min(),df[column].max(),df [column].mean()
    m4,s=df[column].median(),df[column].std()
    print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format(column, m1, m2, m3, m4,5))

print("roupwise Statistical Summary...")
columns=["Experience_Years", "Age", "Salary"]
for column in columns:
    print("\n----------------------",column,"-------------------\n")
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format("Columns","Min","Max","Mean","Median","STD"))
m1=df[column].groupby(df ["Gender"]).min()
m2=df[column].groupby(df["Gender"]).max()
m3=df[column].groupby(df["Gender"]).mean()
m4=df[column].groupby(df["Gender"]).median()
s=df[column].groupby(df["Gender"]).std()
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format("Female",m1[0],m2[0],m3[0],m4[0],s[0]))
print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format("Male",m1[1],m2[1],m3[1],m4[1],s[1]))
X = ["min", "max","mean", "median", "std"]
features = ["Salary", "Age","Experience_Years"]
df1 = pd.DataFrame(columns=["min", "max", "mean", "median", "std"])
for var in features:
 df1["min"] =df[var].groupby(df["Gender"]).min()
 df1["max"] =df[var].groupby(df["Gender"]).max()
 df1["mean"] =df[var].groupby(df["Gender"]).mean()
 df1["median"]=df [var].groupby(df["Gender"]).median ()
 df1["std"]=df[var].groupby(df["Gender"]).std()

X_axis=np.arange(len(X))
plt.bar (X_axis-0.2,df1.iloc[0], 0.4, label = "Female")
plt.bar(X_axis+0.2,df1.iloc[1], 0.4, label = "Male")
plt.xticks(X_axis, X)
plt.xlabel("Statistical information")
plt.ylabel(var)
plt.title("Groupwise Statistical Information of Employee Salary Dataset")
plt.legend()
plt.show()