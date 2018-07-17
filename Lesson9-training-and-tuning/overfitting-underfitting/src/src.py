# Import, read, and split data
import pandas as pd
import numpy as np
from utils.utils import randomize, draw_learning_curves, draw_points_with_model

data = pd.read_csv('../data/data.csv')

X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.
num_trainings = 10

### Logistic Regression
estimator1 = LogisticRegression()
estimator1.fit(X,y)
draw_learning_curves(X,y,estimator1,num_trainings)
draw_points_with_model(X,y,estimator1)
### Decision Tree
estimator2 = GradientBoostingClassifier()
estimator2.fit(X,y)
draw_points_with_model(X,y,estimator2)
draw_learning_curves(X,y,estimator2,num_trainings)
### Support Vector Machine
estimator3 = SVC(kernel='rbf', gamma=1000)
estimator3.fit(X,y)
draw_learning_curves(X,y,estimator3,num_trainings)
draw_points_with_model(X,y,estimator3)
