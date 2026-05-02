from sklearn.linear_model import LogisticRegression
import numpy as np

def train_logistic(train_loader):
    X = []
    y = []

    for images, labels in train_loader:
        # flatten 图片
        X.append(images.view(images.size(0), -1).numpy())
        y.append(labels.numpy())

    X = np.vstack(X)
    y = np.hstack(y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model