from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test, model_name="Model", cv_accuracy=None):
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"{model_name} - Cross-Validation Accuracy: {cv_accuracy:.4f}")
    print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}\n")

    return test_accuracy


    