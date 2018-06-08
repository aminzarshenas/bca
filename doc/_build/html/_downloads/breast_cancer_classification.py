"""
================================================================
Example
================================================================

An example of :class:`bca.BCA`
"""
from bca import BCA
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB

# reading the input features and class labels from the breast cancer dataset
X, y = load_breast_cancer().data, load_breast_cancer().target

# setting the main estimator (e.g., naive Bayes in this example)
estimator = GaussianNB()

# setting the feature selection class and indicating the main estimator
selector = BCA(estimator)

# fitting the estimator while performing wrapper feature selection
selector.fit(X, y)

# best selected features
print(selector.features) 

# best validation score (default is accuracy but can be set to other metrics)
print(selector.score)  

# predict function transforms the features intrinsically and predict the class label
print(selector.predict(X[20:25]))  

