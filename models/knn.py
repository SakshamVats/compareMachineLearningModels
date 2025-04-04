from sklearn.neighbors import KNeighborsClassifier
from utils import train_model, evaluate_model

def run_knn(X_train, X_test, y_train, y_test):
    print("Running KNN...")
    model, cv_accuracy = train_model(X_train, y_train, KNeighborsClassifier, n_neighbors=5)
    test_accuracy = evaluate_model(model, X_test, y_test, "KNN", cv_accuracy)
    return "KNN", cv_accuracy, test_accuracy

