import tensorflow as tf
tf.enable_eager_execution()
config = tf.ConfigProto()
sess = tf.Session(config=config)

import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, ReLU
from tensorflow.keras.optimizers import Adam

import shap

from torch_two_sample import *
from scipy.spatial import distance
import torch

from generate_synthetic_data import synthetic_dataset

def data_encode(X, sensitive_feature_idx, y):
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42,
                                                        stratify=X[:, sensitive_feature_idx])
    return X_train, X_test, y_train, y_test

def bce(y_true, y_pred, from_logits=False):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

def construct_model(X_train, y_train):

    model = Sequential()
    model.add(Dense(16, input_dim=X_train.shape[1], kernel_initializer='normal', activation=ReLU()))
    model.add(Dense(2, kernel_initializer='normal'))
    model.add(Activation('softmax'))
    model.compile(loss=bce, optimizer=Adam(lr=0.01), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=len(X_train), epochs=300, verbose=1, shuffle=False)

    return model

def generate_explain(model, x_train, x_test):
    shap.initjs()
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(x_train, 100))

    shap_result = explainer.shap_values(x_test)[1]
    return shap_result

def MMD(male_explain, female_explain):
    mmd_test = MMDStatistic(len(male_explain), len(female_explain))

    if len(male_explain.shape) == 1:
        male_explain = male_explain.reshape((len(male_explain), 1))
        female_explain = female_explain.reshape((len(female_explain), 1))
        all_dist = distance.cdist(male_explain, male_explain, 'euclidean')
    else:
        all_dist = distance.cdist(male_explain, female_explain, 'euclidean')
    median_dist = np.median(all_dist)

    # Calculate MMD.
    t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(male_explain)),
                             torch.autograd.Variable(torch.tensor(female_explain)),
                             alphas=[1 / median_dist], ret_matrix=True)
    p_val = mmd_test.pval(matrix)
    return p_val

def GPF_FAE_metric(X_train, X_test, model, sensitive_feature_test, n=100):
    D1 = X_test[sensitive_feature_test == 1, :]
    D2 = X_test[sensitive_feature_test == 0, :]

    ## Generate the datasets for evaluating D_1^{'} and D_2^{'}

    # Select n/2 data points from D_1 and D_2, respectively
    temp_D1 = D1[:int(n/2), :]
    temp_D2 = D2[:int(n/2), :]

    # Select the data points that are most similar to them, respectively
    similar_D1 = []
    similar_D2 = []
    for k in np.arange(int(n/2)):
        distances = np.sqrt(np.sum(np.square(D2 - temp_D1[k, :]), axis=1))
        min_index = np.argmin(distances)
        similar_D2.append(D2[min_index, :])

        distances = np.sqrt(np.sum(np.square(D1 - temp_D2[k, :]), axis=1))
        min_index = np.argmin(distances)
        similar_D1.append(D1[min_index, :])
    similar_D1 = np.array(similar_D1)
    similar_D2 = np.array(similar_D2)

    D1_select = np.concatenate((temp_D1, similar_D1))
    D2_select = np.concatenate((similar_D2, temp_D2))

    # Generate FAE explanation result
    D1_select_explain_result = generate_explain(model, X_train, D1_select).astype('float32')
    D2_select_explain_result = generate_explain(model, X_train, D2_select).astype('float32')

    GPF_FAE_result = MMD(D1_select_explain_result, D2_select_explain_result)

    return D1_select, D2_select, D1_select_explain_result, D2_select_explain_result, GPF_FAE_result


if __name__ == '__main__':
    # Using the synthetic dataset as an example, you can replace it with the dataset you wish to use
    X, sensitive_feature_idx, y = synthetic_dataset(0.5)

    # Data pre-process
    X_train, X_test, y_train, y_test = data_encode(X, sensitive_feature_idx, y)

    sensitive_feature_train = X_train[:, sensitive_feature_idx]
    sensitive_feature_test = X_test[:, sensitive_feature_idx]

    # Constructing the ML model to be explained
    model = construct_model(X_train, y_train)

    # Evaluating the procedural fairness of the model with the GPF_FAE metric
    D1_select, D2_select, D1_select_explain_result, D2_select_explain_result, GPF_FAE_result = GPF_FAE_metric(X_train, X_test, model, sensitive_feature_test, n=100)

    print("GPF_FAE Metric: ", GPF_FAE_result)