# Twin-SVM

## Usage

```
# import Twin SVM Classifier
from TVSVM import TwinSVMClassifier
```

### set up params:

TODO : documentation regarding importance of parameters

```
# example SVM without kernel
params1 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':0,'kernel_param': 1,'fuzzy':0}

# example SVM RBF kernel params
params2 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':3,'kernel_param': 2,'fuzzy':0}

# example SVM RBF kernel params with fuzzy membership function
params3 = {'Epsilon1': 0.1, 'Epsilon2': 0.1, 'C1': 1, 'C2': 1,'kernel_type':3,'kernel_param': 2,'fuzzy':1}

```

## Train/Test using TWIN SVM Classifier

```
clf = TwinSVMClassifier(**params3)
# for multiclass classification
# clf = sklearn.multiclass.OneVsOneClassifier(clf) or sklearn.multiclass.OneVsRestClassifier(clf) depending on your needs

# Train the classifier
clf = clf.fit(X_train, y_train)

# Output prediction
y_predicted = clf.predict(X_test)

# Test classifier prediction
score = clf.score(y_test, y_predict)


```

### References

1) 	Twin Support Vector Machines for Pattern Classification
	Jayadeva, Senior Member, IEEE, R.Khemchandani, Student Member, IEEE, and Suresh Chandra
	IEEE Transactions on Pattern Analysis & Machine Intelligence, Vol. 29, No. 5, May 2007

2) 	Xiufeng Jiang , Zhang Yi and Jian Cheng Lv
	Fuzzy SVM with a new fuzzy membership function
	Neural Comput & Applic (2006)
