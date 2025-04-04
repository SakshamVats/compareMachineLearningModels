from sklearn.svm import SVC
from utils import train_model, evaluate_model

def run_svm(X_train, X_test, y_train, y_test):
    print("Running SVM...")
    model, cv_accuracy = train_model(X_train, y_train, SVC, kernel="rbf")
    test_accuracy = evaluate_model(model, X_test, y_test, "SVM", cv_accuracy)
    return "SVM", cv_accuracy, test_accuracy

