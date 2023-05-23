#Assignment No. 2 Data Wrangling II Program by Prof. Rahul Dagade & Prof. Varsha Nale

def RemoveOutlier(df,var):
    Q1 =df[var].quantile(0.25)
    Q3 =df[var].quantile(0.75)
    IQR = Q3-Q1
    high, low=Q3+1.5*IQR, Q1-1.5*IQR

    df=df[((df[var] >= low) & (df[var] <= high))]
    print("Outliers removed in", var)
    return df

def DisplayOutliers(df, message):
    fig, axes = plt.subplots(2,2)
    fig.suptitle(message)
    sns.boxplot(data=df, x="raisedhands", ax=axes[0,0])
    sns.boxplot(data=df, x="VisITedResources", ax=axes[0,1])
    sns.boxplot(data=df, x="AnnouncementsView", ax=axes[1,0])
    sns.boxplot(data=df, x="Discussion", ax=axes[1,1])
    fig.tight_layout()
    plt.show()

#import libraries
import  numpy as np
import  pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt

#Read Dataset
df=pd.read_csv(r"C:\Users\niraj\Downloads\student_data.csv")
print("Student Academic Performance Dataset is successfully loaded...")

#Display information of dataset
print("Information of Dataset\:n", df.info)
print("Shape of Dataset (row x column):", df.shape)
print("Columns Name:", df.columns)
print("Total elements in dataset:", df.size)
print("Datatype of attributes {columns}:", df.dtypes)
print("First 5 rows: \n",df.head().T)
print("Last 5 rows:\n",df.tail().T)
print("Any 5 rows: \n",df.sample(5).T)

#Display statistical information of dataset
print("Statistical information of Numerical Columns: \n",df.describe())

#Display null values
print("Total Number of Null values in Dataset:\n", df.isna().sum())

#Handling outliers
DisplayOutliers(df, "Before removing Outliers")
df=RemoveOutlier(df, "raisedhands")
df=RemoveOutlier(df, "VisITedResources")
df=RemoveOutlier(df, "AnnouncementsView")
df=RemoveOutlier(df, "Discussion")
DisplayOutliers(df, "After removing Outliers")
#Conversion of Categorical to Quantitative (Encoding)
df["gender"]=df["gender"].astype("category")
df["gender"]=df["gender"].cat.codes
print("Data types of gender after label encoding =",df.dtypes['gender'])
print("Gender Values:", df["gender"].unique())

sns.boxplot(data=df, x="gender", y="raisedhands", hue="gender")
plt.title("Boxplot with 2 variables gender and raisedhands")
plt.show()

sns.boxplot(data=df,x="NationalITy", y="Discussion", hue="gender")
plt.title("Boxplot with 3 variables gender, nationality, discussion")
plt.show()

print ("Relationship between variables using Scatterplot:")
sns.scatterplot(data=df,x= "raisedhands", y="VisITedResources")
plt.title("Scatterplot for raisedhands, VisITedResources")
plt.show()