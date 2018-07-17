from sklearn.model_selection import learning_curve
from mlxtend.plotting import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt

# It is good to randomize the data before drawing Learning Curves
def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation,:]
    Y2 = Y[permutation]
    return X2, Y2

def draw_learning_curves(X, y, estimator, num_trainings):
    X2, y2 = randomize(X, y)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.title("Learning Curves-" + estimator.__class__.__name__)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")


    plt.legend(loc="best")

    plt.show()


def draw_points_with_model(X, y, model):
    # fig config
    plt.figure()
    plt.grid(True)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.yticks(np.arange(-2, 2, 0.25))
    plt.xticks(np.arange(-2, 2, 0.25))
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    plt.axis([x_min, x_max, y_min, y_max])

    X_min = X.min()
    X_max = X.max()

    # plot input points
    for input, target in zip(X, y):
        plt.plot(input[0], input[1], 'ro' if (target == 1.0) else 'bo', markersize=3)

    y = y.astype(np.integer)

    # plot the svm boundaries
    plot_decision_regions(X=X,
                          y=y,
                          clf=model,
                          legend=2)

    # Update plot object with X/Y axis labels and Figure Title
    plt.xlabel("x1 label", size=14)
    plt.ylabel("x2 label", size=14)
    plt.title(model.__class__.__name__, size=16)
    plt.legend(loc="best")
    plt.show()
