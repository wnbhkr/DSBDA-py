#Assignment Number: 4 Boston Housing (Linear Regression)Program by Prof. Rahul Dagade & Prof. Varsha Nale

def RemoveOutlier(df, var):
    Q1=df[var].quantile(0.25)
    Q3=df[var].quantile(0.75)
    IQR =Q3-Q1
    high, low=Q3+1.5*IQR, Q1-1.5+IQR

    df=df[((df[var] >= low) & (df[var] <=high))]
    return df
def DisplayOutlier(df, msg):
    fig,axes=plt.subplots(1,2)
    fig.suptitle(msg)
    sns.boxplot(data=df, x="rm", ax=axes[0])
    sns.boxplot(data = df, x="lstat", ax=axes[1])
    fig.tight_layout()
    plt.show()
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read Dataset
df=pd.read_csv (r"C:\Users\niraj\Downloads\Boston.csv")
print("Boston dataset is successfully loaded")

#Display information of dataset
print("Information of Dataset:\n", df.info)
print("Shape of Dataset row x column):", df.shape)
print("Columns Name:", df.columns)
print("Total elements in dataset:",df.size)
print("Datatype of attributes (columns):" ,df.dtypes)
print("First 5 rows: \n", df.head().T)
print("last 5 rows:\n",df.tail().T)
print("Any 5 rows: \n",df.sample(5).T)

#Find missing values
print ("Missing values")
print(df.isnull().sum())

#Find correlation matrix-
print("Finding correlation matrix using heatmap:")
sns.heatmap(df.corr(),annot=True)
plt.show()

#Finding and removing outliers
print("Finding and removing outliers: ")
DisplayOutlier(df, "Before removing Outliers:")
print("Identifying overall outliers in Column Name variables")
df = RemoveOutlier(df, "lstat")
df = RemoveOutlier(df,"rm")
DisplayOutlier(df,"After removing Outliers")

#Split the data into inputs and outputs
x=df[["rm","lstat"]] #input data
y=df["medv"]         #output data

#Training and testing data
from sklearn.model_selection import train_test_split

#Assign test data size 20%
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.20, random_state=0)

#Apply linear regression model on training data
from sklearn.linear_model import LinearRegression
model=LinearRegression().fit(x_train, y_train)
y_pred=model.predict(x_test)

#Display accuracy of the model
from sklearn.metrics import mean_absolute_error
print("MAE:",mean_absolute_error(y_test,y_pred))
print("Model Score:",model.score(x_test,y_test))

#Test the model using user input
print("Predict House Price by giving User Input:")
features= np.array([[6,19]])
prediction= model.predict(features)
print("Prediction: ()".format(prediction))