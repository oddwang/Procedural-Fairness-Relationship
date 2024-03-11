## RQ1.4 When the Dataset Is Biased and the ML Model Is Procedural-Unfair, What Is the Impact on its Distributive Fairness If They Exhibit the Same or Opposite Bias?

import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
config = tf.ConfigProto()
sess = tf.Session(config=config)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

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
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42,
                                                        stratify=X[:, sensitive_feature_idx])
    return X_train, X_test, y_train, y_test

def gradient_saliency(model, input_data, class_index=0):
    input_data = tf.convert_to_tensor(input_data)
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        predictions = model(input_data)
        class_score = predictions[:, class_index]
    gradient = tape.gradient(class_score, input_data)
    # gradient_saliency = tf.reduce_sum(tf.square(gradient), axis=1)
    return gradient

def explain_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

    explain_result = gradient_saliency(model, np.array(X_train))

    male_explain = tf.gather(explain_result, group1_idx)
    female_explain = tf.gather(explain_result, group2_idx)

    male_close_female_explain = tf.gather(explain_result, group1_min_distances_group2)
    female_close_male_explain = tf.gather(explain_result, group2_min_distances_group1)

    distance1 = tf.norm(male_explain - male_close_female_explain, axis=1, ord=1)
    distance2 = tf.norm(female_explain - female_close_male_explain, axis=1, ord=1)

    distance1_mean = tf.reduce_mean(distance1)
    distance2_mean = tf.reduce_mean(distance2)

    distance1_mean = tf.cast(distance1_mean, dtype=tf.float32)
    distance2_mean = tf.cast(distance2_mean, dtype=tf.float32)

    return bce + 0.5 * (distance1_mean + distance2_mean)

def construct_model(X_train, y_train):

    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], kernel_initializer='normal', activation=ReLU()))
    model.add(Dense(2, kernel_initializer='normal'))
    model.add(Activation('softmax'))
    return model

def DP(dataset, y_pred):
    metric_dp = ClassificationMetric(dataset, y_pred,
                                     unprivileged_groups=[{'s': 0}],
                                     privileged_groups=[{'s': 1}])
    DP_result = metric_dp.statistical_parity_difference()
    return DP_result

def DP_metric(X_test, y_test, y_test_pred):
    test_pd = pd.concat([X_test, y_test['y_1.0']], axis=1)
    binary_test = BinaryLabelDataset(df=test_pd, label_names=['y_1.0'],
                                     protected_attribute_names=['s'])
    binary_test_pred = binary_test.copy()
    binary_test_pred.labels = y_test_pred

    DP_result = DP(binary_test, binary_test_pred)

    return DP_result

if __name__ == '__main__':
    # Using the synthetic dataset as an example, you can replace it with the dataset you wish to use
    X, sensitive_feature_idx, y = synthetic_dataset(0.65)

    # Data pre-process
    X_train, X_test, y_train, y_test = data_encode(X, sensitive_feature_idx, y)

    sensitive_feature_train = X_train[:, sensitive_feature_idx]
    sensitive_feature_test = X_test[:, sensitive_feature_idx]

    # divide X_train into two groups
    group1_idx = np.where(np.array(X_train)[:, sensitive_feature_idx] == 1)[0]
    group2_idx = np.where(np.array(X_train)[:, sensitive_feature_idx] == 0)[0]

    # find the similar data point in X_train to construct the explained dataset X'

    encoder = MinMaxScaler(feature_range=[0, 1])
    x_encoder = encoder.fit_transform(np.array(X))
    x_train_encoder, x_test_encoder, y_train_encoder, y_test_encoder = train_test_split(x_encoder,
                                                                                        np.array(y),
                                                                                        test_size=0.2,
                                                                                        shuffle=True,
                                                                                        random_state=42,
                                                                                        stratify=X[:,
                                                                                            sensitive_feature_idx])
    group1_X_train_data = x_train_encoder[group1_idx, :]
    group2_X_train_data = x_train_encoder[group2_idx, :]

    group1_min_distances_group2 = []
    group2_min_distances_group1 = []

    for row1 in group1_X_train_data:
        group1_min_distances_group2.append(group2_idx[np.argmin(np.linalg.norm(group2_X_train_data - row1, axis=1))])

    for row1 in group2_X_train_data:
        group2_min_distances_group1.append(group1_idx[np.argmin(np.linalg.norm(group1_X_train_data - row1, axis=1))])
    group1_min_distances_group2 = np.array(group1_min_distances_group2)
    group2_min_distances_group1 = np.array(group2_min_distances_group1)

    # Constructing the ML model to be explained
    model = construct_model(X_train, y_train)
    model.compile(loss=explain_loss, optimizer=Adam(lr=0.01), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=len(X_train), epochs=300, verbose=1, shuffle=False)


    # Evaluating the procedural fairness of the model with the GPF_FAE metric
    D1_select, D2_select, D1_select_explain_result, D2_select_explain_result, GPF_FAE_result = GPF_FAE_metric(X_train,
                                                                                                              X_test,
                                                                                                              model,
                                                                                                              sensitive_feature_test,
                                                                                                              n=100)

    print("GPF_FAE Metric: ", GPF_FAE_result)

    # Evaluating the distributive fairness of the model with the DP metric
    y_test_pred = model.predict_classes(X_test)
    X_test_pd = pd.DataFrame(X_test, columns=['x1', 'x2', 's', 'xp'])
    y_test_pd = pd.DataFrame(y_test, columns=['y_0.0', 'y_1.0'])
    DP_result = DP_metric(X_test_pd, y_test_pd, y_test_pred)
    print("DP Metric: ", DP_result)

