import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_pts(csv_name):
    data = np.asarray(pd.read_csv(csv_name, header=None))
    X = data[:, 0:2]
    y = data[:, 2]

    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])

    plt.scatter(X[np.argwhere(y == 0).flatten(), 0], X[np.argwhere(y == 0).flatten(), 1], s=50, color='blue',
                edgecolor='k')
    plt.scatter(X[np.argwhere(y == 1).flatten(), 0], X[np.argwhere(y == 1).flatten(), 1], s=50, color='red',
                edgecolor='k')

    plt.xlim(-2.05, 2.05)
    plt.ylim(-2.05, 2.05)
    plt.grid(False)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False)
    plt.axis([x_min-.1, x_max+.1, y_min-.1, y_max+.1])

    return X, y

def plot_model(X, y, clf):
    plt.scatter(X[np.argwhere(y==0).flatten(),0],X[np.argwhere(y==0).flatten(),1],s = 50, color = 'blue', edgecolor = 'k')
    plt.scatter(X[np.argwhere(y==1).flatten(),0],X[np.argwhere(y==1).flatten(),1],s = 50, color = 'red', edgecolor = 'k')

    plt.xlim(-2.05,2.05)
    plt.ylim(-2.05,2.05)
    plt.grid(False)
    plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False)

    r = np.linspace(-2.1,2.1,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    z = clf.predict(h)

    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    plt.contourf(s,t,z,colors = ['blue','red'],alpha = 0.2,levels = range(-1,2))
    if len(np.unique(z)) > 1:
        plt.contour(s,t,z,colors = 'k', linewidths = 2)
    plt.show()