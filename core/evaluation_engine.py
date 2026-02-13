import numpy as np

def stability_analysis(model, X, y, problem_type, runs=5):
    scores = []

    for i in range(runs):
        model.random_state = i
        model.fit(X, y)
        scores.append(model.score(X, y))

    return np.std(scores)
