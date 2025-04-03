import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

df = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

df.to_csv("adult_income.csv", index=False)

print("Dataset downloaded! Saved as 'adult_income.csv'")