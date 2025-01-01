#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from pandas import concat
from pandas import read_csv
from helper import series_to_supervised, stage_series_to_supervised
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


# In[3]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PyPluMA


# ### Dataset

# In[4]:

class RNN_MAEPlugin:
 def input(self, inputfile):
  self.dataset = pd.read_csv(inputfile, index_col=0)

 def run(self):
     pass

 def output(self, outputfile):
  self.dataset.fillna(0, inplace=True)
  data = self.dataset[:]
  n_hours = 24*7
  K = 24
  stages = self.dataset[['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
  stages_supervised = series_to_supervised(stages, n_hours, K)
  non_stages = data[['WS_S4', 'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26', 'PUMP_S26', 'PUMP_S25B', 'MEAN_RAIN']]
  non_stages_supervised = series_to_supervised(non_stages, n_hours-1, 1)
  non_stages_supervised_cut = non_stages_supervised.iloc[24:, :]
  n_features = stages.shape[1] + non_stages.shape[1]   # 1 rainfall + FGate_S25A + FGate_S25B + FGate_S26 + 8WS + PUMP_S26
  non_stages_supervised_cut.reset_index(drop=True, inplace=True)
  stages_supervised.reset_index(drop=True, inplace=True)
  all_data = concat([
                   non_stages_supervised_cut.iloc[:, :],
                   stages_supervised.iloc[:, :]],
                   axis=1)
  all_data = all_data.values
  n_train_hours = int(len(all_data)*0.8)
  train = all_data[:n_train_hours, :]    # 0 column is the rainfall to measure heavy/medium/light
  test = all_data[n_train_hours:, :]

  n_obs = n_hours * n_features
  train_X, train_y = train[:, :n_obs], train[:, -stages.shape[1]*K:]
  test_X, test_y = test[:, :n_obs], test[:, -stages.shape[1]*K:]

  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  train_X = scaler.fit_transform(train_X)
  train_y = scaler.fit_transform(train_y)
  test_X = scaler.fit_transform(test_X)
  test_y = scaler.fit_transform(test_y)
  train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
  test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
  model_rnn_60 = keras.Sequential()
  model_rnn_60.add(layers.SimpleRNN(128, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
  model_rnn_60.add(layers.Dense(train_y.shape[1]))   # Regression -> No Need for Activation
  model_rnn_60.summary()
  lr = 0.00001
  EPOCHS = 5
  model_rnn_60.compile(
              optimizer=Adam(learning_rate=lr, decay=lr/EPOCHS),
              loss='mse',
              metrics=['mae'])


  es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=500)
  mc = ModelCheckpoint(PyPluMA.prefix()+"/saved_model/rnn.h5", monitor='val_mae', mode='min', verbose=2, save_best_only=True)


  history = model_rnn_60.fit(train_X, train_y,
                    batch_size=512,
                    epochs=EPOCHS,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False,
                          callbacks=[es, mc])

  plt.rcParams["figure.figsize"] = (8, 6)
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('Epoch', fontsize=16)
  plt.ylabel('Loss', fontsize=16)
  plt.legend(fontsize=14)
  plt.title("Training loss vs Testing loss", fontsize=18)
  plt.show()

  from tensorflow.keras.models import load_model

  model_load = load_model(PyPluMA.prefix()+"/saved_model/rnn.h5")

  yhat = model_load.predict(test_X)
  inv_yhat = scaler.inverse_transform(yhat)
  inv_y = scaler.inverse_transform(test_y)

  inv_yhat = pd.DataFrame(inv_yhat)
  inv_y = pd.DataFrame(inv_y)
  print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)

  error_abs = abs(inv_yhat - inv_y)
  error = inv_yhat - inv_y
  error_19_20 = error.iloc[-17544:, :]
  inv_y.to_csv(outputfile+"/inv_y_rnn.csv")
  inv_yhat.to_csv(outputfile+"/inv_yhat_rnn.csv")
  error.to_csv(outputfile+"/error_rnn.csv")
  plt.rcParams["figure.figsize"] = (16, 4)
  months = ['Jan 2019', 'May 2019', 'Sep 2019', 'Jan 2020', 'May 2020', 'Sep 2020', 'Jan 2021']

  # 17544 = 24 hours * (365+366) days
  locations = ['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
  for i, col in enumerate([-4, -3, -2, -1]):
    plt.plot(error.iloc[-17544:, col], label='RNN')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    plt.ylim(-1.5, 1.5)
    plt.xticks(np.arange(1726, 19270, 2923), months, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.title('Absolute Error of {} in 2019 & 2020'.format(locations[i]), fontsize=16)
    # plt.savefig('figures/Test_Error_rnn_chop.png', dpi=300)
    plt.text(10000, -0.9, 'MAE = {}'.format(float("{:.3f}".format(sum(abs(error.iloc[-17544:, col]))/17544))), fontsize=12)
    plt.show()
    plt.close()

  error_24th_hour = error.iloc[:, [-4, -3, -2, -1]]  # last 4 columns --> t+12
  error_24th_hour

  error_24th_hour.columns = ['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']

  import seaborn as sns
  plt.rcParams["figure.figsize"] = (8, 5)
  # plt.grid()
  plt.xlabel('Station', fontsize=18)
  plt.ylabel('Error', fontsize=18)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.title('Distribution of Prediction Errors (RNN)', fontsize='18')
  ax = sns.violinplot(data=error_24th_hour[['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']])




