import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model

def download_skl_dataset(path, return_X_y = False):
    if os.path.isdir(path):
        if return_X_y:
            return fetch_california_housing(path, 'return_X_y', True)
        else:
            return fetch_california_housing(path)
            
    else:
        raise ValueError('Cannot find path {}'.format(path))

def data_process(ds_dict):
    data = ds_dict['data']
    target = ds_dict['target']
    feature_names = ds_dict['feature_names']
    DESCR = ds_dict['DESCR']
    return data, target, feature_names, DESCR

def feature_normalization(data, log_flag = False):
    if not log_flag:
        std_data = np.std(data, axis=0)
        std_data[std_data==0] = 1
        mean_data = np.mean(data, axis=0)
        norm_data = (data - mean_data) / std_data
    else: 
        log_data = data.copy()
        log_data[:, 2:6] = np.log(log_data[:, 2:6])
        std_data = np.std(log_data, axis=0)
        std_data[std_data==0] = 1
        mean_data = np.mean(log_data, axis=0)
        norm_data = (log_data - mean_data) / std_data
    return norm_data, std_data, mean_data

def train_set_prepare(data, target, random_state=1):
    x_train_val, x_test, y_train_val, y_test = train_test_split(data, target, test_size=0.2, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(data, target, test_size = 0.2, random_state=random_state)
    return x_train, y_train, x_val, y_val, x_test, y_test

def land_map(data, target, bar_label='Default bar', clim=None, step=None):
    lat_longi = data[:,6:]
    fig = plt.figure(figsize=(10,10))
    plt.grid()
    if step is not None:
        cmap = plt.cm.get_cmap('RdYlBu', step)
    else:
        cmap = None
    plt.scatter(lat_longi[:,0], lat_longi[:,1], 15, target, alpha=0.4, cmap=cmap)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title('Housing price distribution map')
    if clim is not None:
        plt.clim(clim[0], clim[1])
    cbar = plt.colorbar(cmap=cmap)
    cbar.set_label(bar_label)
    plt.show()
    
def plot_pred_true(x, y, y_pred):
    # Check the linearity between y and y_pred
    plt.scatter(y, y_pred)
    plt.xlabel('True housing price')
    plt.ylabel('Predicted housing price')

# Craete model
class nn_model:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = None
        self.history = None
    def build_model(self):
        input_ = layers.Input(shape = self.data.shape[1])
        hidden1 = layers.Dense(1024, activation="tanh",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-6))(input_)
        hidden2 = layers.Dense(512, activation="tanh",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden1)
        hidden3 = layers.Dense(512, activation="tanh",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden2)
        hidden4 = layers.Dense(256, activation="tanh",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden3)
        hidden5 = layers.Dense(256, activation="tanh",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden4)
        hidden6 = layers.Dense(64, activation="tanh",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden5)
        hidden7 = layers.Dense(32, activation="tanh",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden6)
        hidden8 = layers.Dense(8, activation="tanh",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden7)
        output = layers.Dense(1, activation="linear",
                             kernel_regularizer=tf.keras.regularizers.l2(1e-6))(hidden8)
        
        self.model = Model(inputs=[input_], outputs=[output])
    def compile_model(self, optimizer="Adam", loss="mse", metrics=["mse"]):
        self.model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)    
    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs, callbacks=None):
        self.history = self.model.fit(x_train, y_train,
                                     batch_size = batch_size,
                                     epochs = epochs,
                                     callbacks = callbacks,
                                     validation_data=(x_val, y_val))
    def test(self, x_test, y_test):
        return self.model.predict(x_test)

def scheduler(epoch):
    if epoch <= 2:
        return 1e-3
    elif epoch <= 20:
        return 1e-4
    else:
        return 5e-5
        
