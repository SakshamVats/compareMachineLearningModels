from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train, model_class, **kwargs):
    model = model_class(**kwargs)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    cv_accuracy = cv_scores.mean()

    return model, cv_accuracy

