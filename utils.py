from sklearn.metrics import recall_score

def evaluate_performance(y_true, y_pred):
    score = recall_score(y_true, y_pred)
    return score