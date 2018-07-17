# Import statements
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('../data/data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# TODO: Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(kernel='rbf', gamma=27)

# TODO: Fit the model.
model.fit(X,y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)

# plot the results
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


# fig config
plt.figure()
plt.grid(True)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.yticks(np.arange(-2, 2, 0.25))
plt.xticks(np.arange(-2, 2, 0.25))
x_min, x_max = min(X.T[0]), max(X.T[0])
y_min, y_max = min(X.T[1]), max(X.T[1])
plt.axis([x_min,x_max,y_min,y_max])

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
plt.title('SVM Decision Region Boundary', size=16)
plt.show()
