from sklearn.tree import DecisionTreeClassifier
from utils import train_model, evaluate_model

def run_decision_tree(X_train, X_test, y_train, y_test):
    print("Running Decision Tree...")
    model, cv_accuracy = train_model(X_train, y_train, DecisionTreeClassifier, random_state=42)
    test_accuracy = evaluate_model(model, X_test, y_test, "Decision Tree", cv_accuracy)
    return "Decision Tree", cv_accuracy, test_accuracy
