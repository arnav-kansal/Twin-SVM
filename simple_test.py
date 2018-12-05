import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from TVSVM import TwinSVMClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

h = .02  # step size in the mesh
params1 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':0,'kernel_param': 1,'fuzzy' :0}
params2 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':3,'kernel_param': 2,'fuzzy' :0}
params3 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':3,'kernel_param': 2,'fuzzy' :1}

names = ["Twin SVM","Twin SVM with RBF Kernel","Twin SVM RBF Kernel with fuzzy membership"]
classifiers = [
    TwinSVMClassifier(**params1),
    TwinSVMClassifier(**params2),
    TwinSVMClassifier(**params3),]

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1,n_classes=3)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
for name, clf in zip(names, classifiers):
    clf = OneVsOneClassifier(clf).fit(X_train, y_train) # or OneVsRestClassifier
    score = clf.score(X_test, y_test)
    print(score)
