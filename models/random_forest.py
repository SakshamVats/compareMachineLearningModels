from sklearn.ensemble import RandomForestClassifier
from utils import train_model, evaluate_model

def run_random_forest(X_train, X_test, y_train, y_test):
    print("Running Random Forest...")
    model, cv_accuracy = train_model(X_train, y_train, RandomForestClassifier, n_estimators=100, random_state=42)
    test_accuracy = evaluate_model(model, X_test, y_test, "Random Forest", cv_accuracy)
    return "Random Forest", cv_accuracy, test_accuracy

