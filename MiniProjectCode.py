# %%
"""
# DSBDA Mini Project: 
Use the following covid_vaccine_statewise.csv dataset and perform following
analytics on the given dataset https://www.kaggle.com/sudalairajkumar/covid19-inindia?select=covid_vaccine_statewise.csv
a. Describe the dataset
b. Number of persons state wise vaccinated for first dose in India
c. Number of persons state wise vaccinated for second dose in India
d. Number of Males vaccinated
e. Number of females vaccinated
"""

# %%
"""
## Name: Shivam Dattatray Shinde; Roll Number: C34444; Division: 4
"""

# %%
"""
## Name: Vedant Santosh Pawar;      Roll Number: C34446; Division: 4
"""

# %%
"""
## Name: Dev Ramu Ghop;                Roll Number: C34449 ; Division: 4
"""

# %%
# Importing necessary libraries
import pandas as pd

# %%
import matplotlib.pyplot as plt

# %%
# Loading the dataset
df = pd.read_csv('covid_vaccine_statewise.csv')

# %%
# a. Describe the dataset
print(df.head())
print(df.info())
print(df.describe())

# %%
# b. Number of persons state wise vaccinated for first dose in India
first_dose_df = df[['State', 'Total Individuals Vaccinated', 'First Dose Administered']]
first_dose_df = first_dose_df.groupby('State').sum().sort_values(by='First Dose Administered', ascending=False)
print(first_dose_df)

# %%
# c. Number of persons state wise vaccinated for second dose in India
second_dose_df = df[['State', 'Total Individuals Vaccinated', 'Second Dose Administered']]
second_dose_df = second_dose_df.groupby('State').sum().sort_values(by='Second Dose Administered', ascending=False)
print(second_dose_df)

# %%
# d. Number of Males vaccinated
males_df = df[['State', 'Male(Individuals Vaccinated)', 'Total Individuals Vaccinated']]
males_df = males_df.groupby('State').sum().sort_values(by='Male(Individuals Vaccinated)', ascending=False)
print(males_df)

# %%
# e. Number of females vaccinated
females_df = df[['State', 'Female(Individuals Vaccinated)', 'Total Individuals Vaccinated']]
females_df = females_df.groupby('State').sum().sort_values(by='Female(Individuals Vaccinated)', ascending=False)
print(females_df)

# %%
# GRAPHS AND VIUALIZED RESULT

# %%
# a. Describe the dataset
print(df.head())
print(df.info())
print(df.describe())

# %%
# b. Number of persons state wise vaccinated for first dose in India
first_dose = df[['State', 'First Dose Administered']].sort_values('First Dose Administered', ascending=False)
plt.figure(figsize=(15, 8))
plt.bar(first_dose['State'], first_dose['First Dose Administered'])
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Number of persons vaccinated for first dose')
plt.title('Number of persons state wise vaccinated for first dose in India')
plt.show()

# %%
# c. Number of persons state wise vaccinated for second dose in India
second_dose = df[['State', 'Second Dose Administered']].sort_values('Second Dose Administered', ascending=False)
plt.figure(figsize=(15, 8))
plt.bar(second_dose['State'], second_dose['Second Dose Administered'])
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Number of persons vaccinated for second dose')
plt.title('Number of persons state wise vaccinated for second dose in India')
plt.show()

# %%
# d. Number of Males vaccinated
male = df['Male(Individuals Vaccinated)'].sum()
print("Number of Males vaccinated:", male)

# %%
# e. Number of females vaccinated
female = df['Female(Individuals Vaccinated)'].sum()
print("Number of Females vaccinated:", female)