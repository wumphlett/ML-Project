from sklearn.neural_network import MLPClassifier

def train_mlp(inp):
    X_train, y_train, X_test, y_test = inp
    clf = MLPClassifier(
        hidden_layer_sizes=(5),
        activation='relu',
        early_stopping=True,
    )
    clf.fit(X_train, y_train)
    return clf
    # return clf.score(X_test, y_test)
