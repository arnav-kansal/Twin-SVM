import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

from new_TVSVM import *

###############################################################################
# Fit Classification model
#params3 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':1,'kernel_param': 1}
params4 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1.09, 'C2': 1.09,'kernel_type':0,'kernel_param': 1,'fuzzy': 1}
# params2 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':2,'kernel_param': 1}
# params3 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':3,'kernel_param': 10}

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
#tvsvm1 = OneVsOneClassifier(TwinSVMClassifier(**params1)).fit(X, y)
import time
blah=[]
# t1 = time.time()
# tvsvm3 = OneVsOneClassifier(TwinSVMClassifier(**params3)).fit(X, y)
# t2 = time.time()
# blah.append(t2-t1)
t1 = time.time()
tvsvm4 = OneVsOneClassifier(TwinSVMClassifier(**params4)).fit(X, y)
t2 = time.time()
blah.append(t2-t1)
#tvsvm6 = OneVsOneClassifier(TwinSVMClassifier(**params6)).fit(X, y)
#tvsvm7 = OneVsOneClassifier(TwinSVMClassifier(**params7)).fit(X, y)
t1 = time.time()
svc = OneVsOneClassifier(svm.SVC(kernel='linear')).fit(X, y)
t2 = time.time()
blah.append(t2-t1)
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
# lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
# titles = ['Twin SVM with linear kernel',
#           'Twin with RBF kernel',
#           'Twin with polynomial (degree 3) kernel',
#           'SVC with linear kernel',
#           'LinearSVC (linear kernel)',
#           'SVC with RBF kernel',
#           'SVC with polynomial (degree 3) kernel']

zzz=[]
import time
for i, clf in enumerate((tvsvm4, svc)):#, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(1, 3, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    t1 = time.time()
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    t2 = time.time()
    zzz.append(t2-t1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    # plt.title(titles[i])
    print i, "hello"

#plt.show()
