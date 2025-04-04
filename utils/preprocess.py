import pandas as pd
import os
from data import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_and_preprocess_data(filepath=None):
    """Loads and preprocesses the Adult Income dataset."""

    get_dataset()

    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, "../adult_income.csv")

    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    X = df.drop(columns=["income"])
    y = df["income"]

    categorical_cols = ["workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_cols])

    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    X = pd.concat([X.drop(columns=categorical_cols), X_encoded_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=69)

    return X_train, X_test, y_train, y_test
    