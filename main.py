import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from data import get_dataset
from utils import load_and_preprocess_data

from models import (
    run_logistic_regression,
    run_decision_tree,
    run_random_forest,
    run_svm,
    run_knn,
    run_naive_bayes,
    run_neural_net
)

def main():
    print("Starting Pipeline:\n")

    # Step 1: Load and preprocess dataset
    df = get_dataset()
    X_train, X_test, y_train, y_test = load_and_preprocess_data(df)

    # Step 2: Run all models and collect results
    results = []
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test))
    results.append(run_decision_tree(X_train, X_test, y_train, y_test))
    results.append(run_random_forest(X_train, X_test, y_train, y_test))
    results.append(run_svm(X_train, X_test, y_train, y_test))
    results.append(run_knn(X_train, X_test, y_train, y_test))
    results.append(run_naive_bayes(X_train, X_test, y_train, y_test))
    results.append(run_neural_net(X_train, X_test, y_train, y_test))

    # Step 3: Plot performance
    model_names = [r[0] for r in results]
    cv_scores = [r[1] for r in results]
    test_scores = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(model_names))

    plt.bar(index, cv_scores, bar_width, label='Cross-Validation Accuracy')
    plt.bar([i + bar_width for i in index], test_scores, bar_width, label='Test Accuracy')

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks([i + bar_width / 2 for i in index], model_names, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
