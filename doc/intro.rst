Introduction
============================================

Binary coordinate ascent (BCA) algorithm is a simple and efficient wrapper method for feature subset selection (FSS) which is one of the major steps in almost every machine learning application. As an advantage to other wrapper FSS meta-heuristics, e.g., sequential forward selection (SFS) and sequential floating forward selection (SFFS), BCA is computationally significantly more efficient. A study [1] showed that the efficiency in terms of the number of subset evaluations was improved significantly (by factors of 5-37), comparing to SFS and SFFS techniques, while maintaining the classification performance for unseen data.

BCA-based FSS has been successfully applied for support vector machine (SVM), multilayer perceptron (MLP) and naive Bayes (NB) classifiers with 12 public datasets and showed higher efficiency when compared to conventional methods. The method is directly extendable to other classifiers and other binary and multi-class datasets.

The advantages of BCA algorithm are:

- Efficient against conventional techniques, i.e., SFS and SFFS.
- Easy to implement.


The disadvantages of BCA algorithm are:

- When coupled with classifiers with computationally expensive training, e.g., MLP, the execution time can be long. This is, however, a common cons. of wrapper based techniques.

|

**Reference**:

[1] Zarshenas, A. and Suzuki, K., "Binary coordinate ascent: An efficient optimization technique for feature subset selection for machine learning", Knowledge-Based Systems 110 (2016): 191-201.