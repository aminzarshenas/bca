# BCA
Binary Coordinate Ascent (BCA) algorithm for feature subset selection

## Description
Binary coordinate ascent (BCA) algorithm is a simple and efficient wrapper method for feature subset selection (FSS) which is one of the major steps in almost every machine learning application. As an advantage to other wrapper FSS meta-heuristics, e.g., sequential forward selection (SFS) and sequential floating forward selection (SFFS), BCA is computationally significantly more efficient. A study [1] showed that the efficiency in terms of the number of subset evaluations was improved significantly (by factors of 5-37), comparing to SFS and SFFS techniques, while maintaining the classification performance for unseen data.

BCA-based FSS has been successfully applied for support vector machine (SVM), multilayer perceptron (MLP) and naive Bayes (NB) classifiers with 12 public datasets and showed higher efficiency when compared to conventional methods. The method is directly extendable to other classifiers and other binary and multi-class datasets.

The advantages of BCA algorithm are:
- Efficient against conventional techniques, i.e., SFS and SFFS.
- Easy to implement.

The disadvantages of BCA algorithm are:
- When coupled with classifiers with computationally expensive training, e.g., MLP, the execution time can be long. This is, however, a common cons. of wrapper based techniques. 

## Illustration:

The class BCA implements a plain BCA algorithm which can be optionally coupled with specific classifiers such as GaussianNB(), SVC() and MLPClassifier(), as the main classifier, while performing wrapper-based FSS. A user can also choose the type of metric to be used as the goodness of the feature subsets, e.g., roc_auc or accuracy of the given classifier. Additionally, a user might also indicate the type of cross-validation or the number of folds to be used during performance estimation. BCA class returns the final selected feature set, as well as the final estimator, trained on the final subset of features selected by BCA.

BCA has to be fitted with two arrays: an array X of size [n_samples, n_features] holding the training inputs and an array y of size [n_samples] holding the target values (class labels) for a given dataset. Once the BCA is fitted, the predict method can be used to intrinsically transform an input matrix X using the selected features, and predict its class using the underlying estimator. Following shows an example of feature selection using BCA, coupled with a NB classifier, for a binary classification task on the breast cancer dataset:

```python
# Import libraries 
>>> from bca import BCA
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.naive_bayes import GaussianNB
# Load the dataset, e.g., we loaded the breast cancer dataset.
>>> X, y = load_breast_cancer().data, load_breast_cancer().target
# Select a classifier, e.g., we used the NB classifier. 
>>> estimator = GaussianNB()
# Initiate a BCA object using the selected estimator, indicate the scoring metric and also the 
# number of folds for a cross-validation estimation of the score
>>> selector = BCA(estimator, scoring='accuracy', cv=5)
# Use fit method to find the “best” features and the final trained estimator
>>> selector.fit(X, y)
>>> selector.features
[ 1  4  6  7 16 20 21 22 23 27 28]
>>> selector.score
0.971989226626
>>> selector.predict(X[20:25])
[1 1 0 0 0]
```

## Reference:
[1] Zarshenas, A. and Suzuki, K., "Binary coordinate ascent: An efficient optimization technique for feature subset selection for machine learning", Knowledge-Based Systems 110 (2016): 191-201.
 


