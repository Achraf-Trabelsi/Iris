from datetime import date
from sklearn import datasets
import matplotlib
import pylab as pl
from itertools import cycle
from sklearn import naive_bayes
import numpy as np
import random as r
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


irisData = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(irisData.data,irisData.target, test_size=0.3, random_state=0)
clf=naive_bayes.MultinomialNB(fit_prior=True)
clf.fit(X_train[:], y_train[:])
p = clf.predict(X_test[:])
a=clf.score(X_test,y_test)
print(1-a)



scores = cross_val_score(clf, irisData.data,irisData.target, cv=8)
print(scores)