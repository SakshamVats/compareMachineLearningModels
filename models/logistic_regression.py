from sklearn.linear_model import LogisticRegression
from utils import train_model, evaluate_model

def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("Running Logistic Regression...")
    model, cv_accuracy = train_model(X_train, y_train, LogisticRegression, max_iter=1000)
    test_accuracy = evaluate_model(model, X_test, y_test, "Logistic Regression", cv_accuracy)
    return "Logistic Regression", cv_accuracy, test_accuracy




