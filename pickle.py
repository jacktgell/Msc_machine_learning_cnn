from tensorflow import keras
from sklearn import preprocessing
import numpy as np
import pandas as pd

def Normalise_Data(df):
    df = df.interpolate( axis="rows", limit_direction = "both", order = 1)
    df = df.fillna(0)
    Normalise = preprocessing.Normalizer()
    x_scaled = Normalise.fit_transform(df)
    df = pd.DataFrame(x_scaled)
    return df

#load model
rf = keras.models.load_model('my_model.h5')
#set this to the test data location
test = pd.read_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\TEST.csv")
#remove id col
ID = test.pop('ID')
#normalise features
test = Normalise_Data(test)
#run through model
pred = rf.predict(test)
#convert to dataframe
pred = pd.DataFrame(pred)
#rename a colums
pred.columns = ['prediction']
#round to closest decimal place
pred = np.around(pred, decimals=0)
#kaggle doesnt take floating points
pred = pred.astype(int)
#append labels
pred = pd.concat([ID, pred], axis=1, sort=False)
#save results in a csv file in a given location
pred.to_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\PREDICTION9.csv", index = False)
























