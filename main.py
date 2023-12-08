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

    survival_percentages_class.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Percentage')
    plt.title('Survival Percentages by Passenger Class')
    plt.legend(title='Survival Status', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #plotting commands
    print("As seen in the figure, there appears to be a correlation with passenger's class and their survival rate\n")

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

    survival_percentages_age.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel('Age Class')
    plt.ylabel('Survival Percentage')
    plt.title('Survival Percentages by Age Class')
    plt.legend(title='Survival Status', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #plotting commands
    print("As seen in the figure, the highest age class to survive where those within the 0-17 age class. This suggests that children were prioritized when evacuating the Titanic\n")

def query_3():
    df_embarked_not_null = df.dropna(subset=['Embarked']).copy()
    #removes rows with no embarkation data, also creates a copy of the dataset

    #calculate survival percentages by embarkation port
    survival_percentages_embarked = df_embarked_not_null.groupby(['Embarked', 'Survived']).size().unstack(fill_value=0)
    survival_percentages_embarked = survival_percentages_embarked.div(survival_percentages_embarked.sum(axis=1), axis=0) * 100
    survival_percentages_embarked = survival_percentages_embarked.rename({0: 'Did not survive', 1: 'Survived'}, axis=1)

    survival_percentages_embarked.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel('Embarkation Port')
    plt.ylabel('Survival Percentage')
    plt.title('Survival Percentages by Embarkation Port')
    plt.legend(title='Survival Status', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #plotting commands
    print("As seen in the figure, there is a 16 percent higher chance of survival if you embarked from the C Port. This could suggest those passengers were closer to lifevest/evacuation boats\n")

def query_4():
    df_fare_not_null = df.dropna(subset=['Fare']).copy()
    #removes rows with no fare data, also makes a copy of the dataframe
    
    fare_divisions = [0, 50, 100, 200, max(df_fare_not_null['Fare'])]
    #defines fare groups. we chose to seperate fare groups by sets of 50.
    fare_ranges = ['0-49', '50-99', '100-199', '200+']
    #defines the ranges for the fare prices
    df_fare_not_null.loc[:, 'Fare_Group'] = pd.cut(df_fare_not_null['Fare'], bins=fare_divisions, labels=fare_ranges, right=False)

    #calculates survival percentages by fare group
    survival_counts_fare = df_fare_not_null.groupby(['Fare_Group', 'Survived'], observed=True).size().unstack(fill_value=0)
    survival_percentages_fare = survival_counts_fare.div(survival_counts_fare.sum(axis=1), axis=0) * 100
    survival_percentages_fare = survival_percentages_fare.rename({0: 'Did not survive', 1: 'Survived'}, axis=1)

    survival_percentages_fare.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel('Fare Group')
    plt.ylabel('Survival Percentage')
    plt.title('Survival Percentages by Fare Group')
    plt.legend(title='Survival Status', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #plotting commands
    print("As seen in the figure, those who paid 50 dollars or more had a significantly higher chance of surviving than those who paid less than 50 dollars. This could suggest that cheaper tickets were located on lower floors and therefore closest to the impact zones\n")

def query_5():
    df_age_fare_not_null = df.dropna(subset=['Age', 'Fare']).copy()
    #remove rows with no age or fare data, also creates a copy of the original data set.

    fare_divisions = [0, 50, 100, 200, max(df_age_fare_not_null['Fare'])]
    #defines fare groups. we chose to seperate fare groups by sets of 50.
    fare_ranges = ['0-49', '50-99', '100-199', '200+']
    #defines the ranges for the fare prices
    df_age_fare_not_null.loc[:, 'Fare_Group'] = pd.cut(df_age_fare_not_null['Fare'], bins=fare_divisions, labels=fare_ranges, right=False)

    age_divisions = [0, 18, 30, 50, 100]
    #defines age classes based on present day age grouping
    age_ranges = ['0-17', '18-29', '30-49', '50+']
    #defines the age range
    df_age_fare_not_null.loc[:, 'Age_Group'] = pd.cut(df_age_fare_not_null['Age'], bins=age_divisions, labels=age_ranges, right=False)

    age_fare_counts = df_age_fare_not_null.groupby(['Age_Group', 'Fare_Group'], observed=True).size().unstack(fill_value=0)
    #calculates counts of passengers by age group and fare group

    age_fare_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.xlabel('Age Group')
    plt.ylabel('Passenger Counts')
    plt.title('Passenger Counts by Age Group and Fare Group')
    plt.legend(title='Fare Group', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #plotting commands
    print("As seen in the figure, the majority of the sold tickets were under 50 dollars. It is also observed that the majority of the tickets sold above 50 dollars were sold to passengers who were between the age of 18 and 49.\n")

def query_6():
    df_fare_embarked_not_null = df.dropna(subset=['Fare', 'Embarked']).copy()
    #removes rows with no fare or embarkation data, also copies the original data set

    fare_divisions = [0, 50, 100, 200, max(df_fare_embarked_not_null['Fare'])]
    #defines fare groups. we chose to seperate fare groups by sets of 50.
    fare_ranges = ['0-49', '50-99', '100-199', '200+']
    #defines the ranges for the fare prices
    df_fare_embarked_not_null.loc[:, 'Fare_Group'] = pd.cut(df_fare_embarked_not_null['Fare'], bins=fare_divisions, labels=fare_ranges, right=False)
    fare_group_embarkation_counts = df_fare_embarked_not_null.groupby(['Fare_Group', 'Embarked'], observed=True).size().unstack(fill_value=0)
    #counts tickets by fare group and embarkation port

    fare_group_embarkation_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.xlabel('Fare Group')
    plt.ylabel('Ticket Counts')
    plt.title('Ticket Counts by Fare Group and Embarkation Port')
    plt.legend(title='Embarkation Port', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #plotting commands
    print("As seen from the figure, the majority of passengers boarded from port S regardless of price. However, it is observed that no tickets above 100 dollars embarked from port Q.\n")

def query_7():
    df_gender_survival_not_null = df[['Sex', 'Survived']].dropna().copy()
    #removes rows with no sex or survived data, also copies the original data set
    #calculates survival percentages by gender
    survival_percentages_gender = df_gender_survival_not_null.groupby(['Sex', 'Survived']).size().unstack().fillna(0)
    survival_percentages_gender = survival_percentages_gender.div(survival_percentages_gender.sum(axis=1), axis=0) * 100
    survival_percentages_gender = survival_percentages_gender.rename({0: 'Did not survive', 1: 'Survived'}, axis=1)

    survival_percentages_gender.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.xlabel('Gender')
    plt.ylabel('Survival Percentage')
    plt.title('Survival Percentages by Gender')
    plt.legend(title='Survival Status', loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #plotting commands
    print("As seen in the figure, the large majority of survivors were women. This suggests they prioritized women when evacuating the Titanic.")

query_1()
query_2()
query_3()
query_4()
query_5()
query_6()
query_7()
#calling the queries
