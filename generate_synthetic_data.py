import numpy as np
import numpy.random
import pandas as pd
from scipy.stats import multivariate_normal

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

def synthetic_dataset(p=0.6):
    np.random.seed(42)
    y1 = np.ones(10000)
    y2 = np.zeros(10000)
    y = np.concatenate((y1, y2))

    mean1 = [2, 2]
    cov1 = [[5, 1], [1, 5]]
    x1 = np.random.multivariate_normal(mean1, cov1, 10000)

    mean2 = [-2, -2]
    cov2 = [[10, 1], [1, 3]]
    x2 = np.random.multivariate_normal(mean2, cov2, 10000)

    x = np.concatenate((x1, x2))

    s1 = np.random.binomial(1, p, size=10000)
    s2 = np.random.binomial(1, 1-p, size=10000)
    s = np.concatenate((s1, s2))

    xp = np.random.normal(s, 0.5, 20000)

    x = np.concatenate((x, s.reshape(-1, 1), xp.reshape(-1, 1)), axis=1)

    sensitive_feature_idx = 2

    return x, sensitive_feature_idx, y


if __name__ == '__main__':

    synthetic_dataset(0.65)




