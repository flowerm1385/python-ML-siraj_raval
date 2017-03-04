from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data and labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = list('mmffmmfffmm')

# Classifiers
# using the default values for all the hyperparameters
models = [dict(name="", obj=M, pred=[], acc=0) for M in 
          [tree.DecisionTreeClassifier(), SVC(), Perceptron(), KNeighborsClassifier()]]

# Training the models
for model in models: 
        model['obj'].fit(X,Y)
        model['pred'] = model['obj'].predict(X)
        model['name'] = type(model['obj']).__name__
        model['acc'] = accuracy_score(Y, model['pred']) * 100
        print('Accuracy for {}: {}'.format(model['name'], model['acc']))

# The best classifier
index = np.argmax([model['acc'] for model in models])
print('Best gender classifier is {}'.format(models[index]['name']))
