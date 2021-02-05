# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import pandas as pd
from kerastuner import BayesianOptimization
from tensorflow.keras import regularizers

#%%
def Normalise_Data(df):
    #interpolate
    df = df.interpolate( axis="rows", limit_direction = "both", order = 1)
    #fill remaining nan values with zero
    df = df.fillna(0)
    #using nl2 by default
    Normalise = preprocessing.Normalizer()
    x_scaled = Normalise.fit_transform(df)
    #convert to dataframe and return
    df = pd.DataFrame(x_scaled)
    return df

#%%
#load data
cnn = pd.read_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\CleanDataCNN.csv")

#%%
#pop labels
cnn_labels = cnn.pop('4608')

#%%
#randomize data rows and take a validation set of 20%
x_train, x_valid, y_train, y_valid = train_test_split(cnn, cnn_labels, test_size=0.2,shuffle=True)

#%%
def build_model_CNN(hp):
    model = keras.Sequential()
    #range of nodes in each layer changes between values of the range the hp is defined as
    model.add(layers.Dense(units=hp.Int("input_units",10,4000,1), input_shape=(4608,),
              activity_regularizer=regularizers.l2(0.1))) #l2 weight shrinkage
    
    model.add(layers.Dense(use_bias = True,units=hp.Int('units_1', 1, 4000, 1),
              activation=hp.Choice('acti_1', ['relu']),
              activity_regularizer=regularizers.l2(0.1)))
    model.add(layers.Dropout(hp.Float('drop_1', 0, 0.5, 0.1)))#dropout only on hidden layers
    
    model.add(layers.Dense(use_bias = True,units=hp.Int('units_2', 1, 4000, 1),
              activation=hp.Choice('acti_2', ['relu']),
              activity_regularizer=regularizers.l2(0.1)))
    
    model.add(layers.Dense(use_bias = True,units=hp.Int('units_3', 1, 4000, 1),
              activation=hp.Choice('acti_3', ['relu']),
              activity_regularizer=regularizers.l2(0.1)))
    model.add(layers.Dropout(hp.Float('drop_3', 0, 0.5, 0.1)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(
        lr = hp.Choice('learning_rate', [1e-4,1e-3,1e-2])))
    return model

#%%
#set optimiser
tuner = BayesianOptimization(
    build_model_CNN,
    objective='val_accuracy',
    metrics=['accuracy'],
    max_trials=10000,
    executions_per_trial=2)
#normalise data
x = np.array(Normalise_Data(x_train))
xv = np.array(Normalise_Data(x_valid))
#begin hp search
tuner.search(x=x,
             y=y_train,
             epochs=150,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)],
             batch_size=32,
             verbose=2,
             validation_data=(xv,y_valid))

#%%
#my method for extracting best model however look to pickle python script for loading a model
rf = tuner.get_best_models(1)[0]
test = pd.read_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\TEST.csv")
ID = test.pop('ID')
#test = test.loc[:, 'CNNs':'CNNs.4095':]
test = Normalise_Data(test)
pred = rf.predict(test)
pred = pd.DataFrame(pred)
pred.columns = ['prediction']
pred = np.around(pred, decimals=0)
pred = pred.astype(int)
pred = pd.concat([ID, pred], axis=1, sort=False)
pred.to_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\PREDICTION9.csv", index = False)




























