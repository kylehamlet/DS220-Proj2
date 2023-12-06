import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("titanic.csv")
columns_to_remove = ["Passenger-ID", "Name", "Ticket", "Cabin", "Sibsp", "Parch"]
df = df.drop(columns=columns_to_remove)
print(df)
