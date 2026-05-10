from sklearn.metrics import recall_score


def fit_model(model, X_train, y_train):
    return model.fit(X_train, y_train)

def evaluate_performance(y_true, y_pred):
    """
    """
    score = recall_score(y_true, y_pred)

    return score