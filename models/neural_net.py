from sklearn.neural_network import MLPClassifier
from utils import train_model, evaluate_model

def run_neural_net(X_train, X_test, y_train, y_test):
    print("Running Neural Network...")
    model, cv_accuracy = train_model(
        X_train, y_train,
        MLPClassifier,
        hidden_layer_sizes=(50,),
        max_iter=300,
        random_state=42
    )
    test_accuracy = evaluate_model(model, X_test, y_test, "Neural Network", cv_accuracy)
    return "Neural Network", cv_accuracy, test_accuracy

