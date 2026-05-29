from sklearn.metrics import f1_score

def evaluate_performance(y_true, y_pred):
    score = f1_score(y_true, y_pred)
    return score