## RQ1.3 How does procedural unfairness of the ML model affect its distributive fairness?

import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
config = tf.ConfigProto()
sess = tf.Session(config=config)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
import numpy as np

import torch

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, ReLU
from tensorflow.keras.optimizers import Adam

from torch_two_sample import *
from scipy.spatial import distance
from scipy.stats import ks_2samp

import shap

from generate_synthetic_data import synthetic_dataset
from GPF_FAE_metric import GPF_FAE_metric

def data_encode(X, sensitive_feature_idx, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42,
                                                        stratify=X[:, sensitive_feature_idx])
    return X_train, X_test, y_train, y_test

def DP(dataset, y_pred):
    metric_dp = ClassificationMetric(dataset, y_pred,
                                     unprivileged_groups=[{'s': 0}],
                                     privileged_groups=[{'s': 1}])
    DP_result = metric_dp.statistical_parity_difference()
    return DP_result

def DP_metric(X_test, y_test, y_test_pred):
    test_pd = pd.concat([X_test, y_test['y']], axis=1)
    binary_test = BinaryLabelDataset(df=test_pd, label_names=['y'],
                                     protected_attribute_names=['s'])
    binary_test_pred = binary_test.copy()
    binary_test_pred.labels = y_test_pred

    DP_result = DP(binary_test, binary_test_pred)

    return DP_result

if __name__ == '__main__':
    # Using the synthetic dataset as an example, you can replace it with the dataset you wish to use
    X, sensitive_feature_idx, y = synthetic_dataset(0.65)
    X = X[:, :-1]

    # Data pre-process
    X_train, X_test, y_train, y_test = data_encode(X, sensitive_feature_idx, y)



    sensitive_feature_train = X_train[:, sensitive_feature_idx]
    sensitive_feature_test = X_test[:, sensitive_feature_idx]

    # Constructing the ML model to be explained
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # control the parameter w_s
    ws_range = np.linspace(-5, 5, 10)
    for ws_par in ws_range:
        model.coef_[0, sensitive_feature_idx] = ws_par

        # Evaluating the procedural fairness of the model with the GPF_FAE metric
        D1_select, D2_select, D1_select_explain_result, D2_select_explain_result, GPF_FAE_result = GPF_FAE_metric(X_train,
                                                                                                                  X_test,
                                                                                                                  model,
                                                                                                                  sensitive_feature_test,
                                                                                                                  n=100)
        # Evaluating the distributive fairness of the model with the DP metric
        y_test_pred = model.predict(X_test)
        X_test_pd = pd.DataFrame(X_test, columns=['x1', 'x2', 's'])
        y_test_pd = pd.DataFrame(y_test, columns=['y'])
        DP_result = DP_metric(X_test_pd, y_test_pd, y_test_pred)

        print('w_s: ', ws_par)
        print("GPF_FAE Metric: ", GPF_FAE_result)
        print("DP Metric: ", DP_result)

