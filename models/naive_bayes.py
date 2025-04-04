from sklearn.naive_bayes import GaussianNB
from utils import train_model, evaluate_model

def run_naive_bayes(X_train, X_test, y_train, y_test):
    print("Running Naive Bayes...")
    model, cv_accuracy = train_model(X_train, y_train, GaussianNB)
    test_accuracy = evaluate_model(model, X_test, y_test, "Naive Bayes", cv_accuracy)
    return "Naive Bayes", cv_accuracy, test_accuracy

