import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("titanic.csv")
#reading the csv file

columns_to_remove = ["Passenger-ID", "Name", "Ticket", "Cabin", "Sibsp", "Parch"]
df = df.drop(columns=columns_to_remove)
#removing irrelevant columns

df.isnull().sum()
#checking data set for null values

df.dropna(how='all',inplace=True)
#removing all rows that have null values

df.isnull().sum()
#checking to see that all nulls have been removed

print(df)
