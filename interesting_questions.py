import pandas as pd

from src.core.dataframe import DataFrame


DATA_PATH = "Data/train.csv"

def print_result(question, mini_result, pandas_result, match):
    print("=" * 10)
    print(question)
    print("Mini-pandas result:")
    print(mini_result)
    print("Standard pandas result:")
    print(pandas_result)
    print("Match:", match)
    print()


def age_range_mask(series, low, high):
    mask = []

    for value in series.data:
        if value == "":
            mask.append(False)
        else:
            mask.append(low <= float(value) <= high)

    return mask


def clean_value(value):
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, float):
        return round(value, 4)

    return value


def survival_rate(frame):
    total = len(frame)

    if total == 0:
        return 0

    survived = 0

    for value in frame["Survived"].data:
        survived += int(value)

    return round(survived / total, 4)


mini_df = DataFrame.from_csv(DATA_PATH)
pandas_df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

question = "Question 1: How many female passengers are in the dataset?"
mini_female = mini_df[mini_df["Sex"] == "female"]
pandas_female = pandas_df[pandas_df["Sex"] == "female"]
mini_result = len(mini_female)
pandas_result = len(pandas_female)
print_result(question, mini_result, pandas_result, mini_result == pandas_result)


question = "Question 2: How many passengers are between ages 18 and 30?"
mini_age_mask = age_range_mask(mini_df["Age"], 18, 30)
mini_age_range = mini_df[mini_age_mask]
pandas_age = pd.to_numeric(pandas_df["Age"], errors="coerce")
pandas_age_range = pandas_df[(pandas_age >= 18) & (pandas_age <= 30)]
mini_result = len(mini_age_range)
pandas_result = len(pandas_age_range)
print_result(question, mini_result, pandas_result, mini_result == pandas_result)


question = "Question 3: What are the survival rates for male and female passengers?"
mini_male = mini_df[mini_df["Sex"] == "male"]
mini_female = mini_df[mini_df["Sex"] == "female"]
pandas_survived = pd.to_numeric(pandas_df["Survived"], errors="coerce")
pandas_male = pandas_df[pandas_df["Sex"] == "male"]
pandas_female = pandas_df[pandas_df["Sex"] == "female"]
mini_result = {
    "male": survival_rate(mini_male),
    "female": survival_rate(mini_female)
}
pandas_result = {
    "male": clean_value(round(pandas_survived[pandas_df["Sex"] == "male"].mean(), 4)),
    "female": clean_value(round(pandas_survived[pandas_df["Sex"] == "female"].mean(), 4))
}
print_result(question, mini_result, pandas_result, mini_result == pandas_result)


question = "Question 4: What are the survival rates for first-class and third-class passengers?"
mini_first_class = mini_df[mini_df["Pclass"] == "1"]
mini_third_class = mini_df[mini_df["Pclass"] == "3"]
mini_result = {
    "first_class": survival_rate(mini_first_class),
    "third_class": survival_rate(mini_third_class)
}
pandas_result = {
    "first_class": clean_value(round(pandas_survived[pandas_df["Pclass"] == "1"].mean(), 4)),
    "third_class": clean_value(round(pandas_survived[pandas_df["Pclass"] == "3"].mean(), 4))
}
print_result(question, mini_result, pandas_result, mini_result == pandas_result)
