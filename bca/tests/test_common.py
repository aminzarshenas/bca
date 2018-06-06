from sklearn.utils.estimator_checks import check_estimator
from bca import (BCA)


def test_estimator():
    return check_estimator(BCA)

test_estimator()