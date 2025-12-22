import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def plot_confusion(y_true, y_pred, labels, title):
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True, fmt="d",
                xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_title(title)
    return fig

def plot_pca(X, y):
    coords = PCA(2).fit_transform(X)
    fig, ax = plt.subplots()
    ax.scatter(coords[:,0], coords[:,1], c=y, cmap="tab10")
    ax.set_title("PCA – Taste Clustering")
    return fig

def plot_regression(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.plot([min(y_true), max(y_true)],
            [min(y_true), max(y_true)], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Docking Regression")
    return fig
