import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

df = pd.read_csv("titanic.csv")
#reading the csv file

columns_to_remove = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]
df = df.drop(columns=columns_to_remove)
#removing irrelevant columns

df.isnull().sum()
#checking data set for null values

df.dropna(how='all',inplace=True)
#removing all rows that have null values

df.isnull().sum()
#checking to see that all nulls have been removed

def query_1():
    # Calculate survival percentages by Passenger Class
    survival_percentages_class = df.groupby(['Pclass', 'Survived']).size().unstack().fillna(0)
    survival_percentages_class = survival_percentages_class.div(survival_percentages_class.sum(axis=1), axis=0) * 100
    survival_percentages_class = survival_percentages_class.rename({0: 'Did not survive', 1: 'Survived'}, axis=1)

    print("Survival percentages by Passenger Class:")
    print(survival_percentages_class)
    print("As seen above, there appears to be a correlation with passenger's class and their survival rate\n")

def query_2():
    df_age_not_null = df.dropna(subset=['Age']).copy()
    #creates a new DataFrame excluding rows with no age values, also makes a copy of the data set to avoid the original data set

    age_divisions = [0, 18, 30, 50, 100]
    #defines age classes based on present day age grouping
    age_ranges = ['0-17', '18-29', '30-49', '50+']
    #defines the age range
    df_age_not_null.loc[:, 'Age_class'] = pd.cut(df_age_not_null['Age'], bins=age_divisions, labels=age_ranges, right=False)

    #calculates survival percentages by age class
    survival_counts_age = df_age_not_null.groupby(['Age_class', 'Survived'], observed=True).size().unstack(fill_value=0)
    survival_percentages_age = survival_counts_age.div(survival_counts_age.sum(axis=1), axis=0) * 100
    survival_percentages_age = survival_percentages_age.rename({0: 'Did not survive', 1: 'Survived'}, axis=1)

    print("Survival percentages by Age class:")
    print(survival_percentages_age)
    print("As seen above, the highest age class to survive where those within the 0-17 age class. This suggests that children were prioritized when evacuating the Titanic\n")

def query_3():
    df_embarked_not_null = df.dropna(subset=['Embarked']).copy()
    #removes rows with missing embarkation data, also creates a copy of the dataset

    #calculate survival percentages by embarkation port
    survival_percentages_embarked = df_embarked_not_null.groupby(['Embarked', 'Survived']).size().unstack(fill_value=0)
    survival_percentages_embarked = survival_percentages_embarked.div(survival_percentages_embarked.sum(axis=1), axis=0) * 100
    survival_percentages_embarked = survival_percentages_embarked.rename({0: 'Did not survive', 1: 'Survived'}, axis=1)

    print("Survival percentages by Embarkation Port:")
    print(survival_percentages_embarked)
    print("As seen above, there is a 16 percent higher chance of survival if you embarked from the C Port. This could suggest those passengers were closer to lifevest/evacuation boats\n")

query_1()
#calling the first query
query_2()
#calling the second query
query_3()
#calling the third query
