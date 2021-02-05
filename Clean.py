import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

#%%
#function taht cound the ratio of true a false samples
def get_count(df):
    neg, pos = np.bincount(df['prediction'])
    print("zeros = ", neg, "   ones = ", pos)
    return neg

#%%
#load in data
df_data = pd.read_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\TRAINING.csv")
df_add = pd.read_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\ADDITIONAL.csv")

#remove id
df_data.pop("ID")
df_add.pop("ID")

#%%
#get counts
neg_data = get_count(df_data)
neg_add = get_count(df_add)

#%%
true_data = df_data.loc[df_data['prediction'] == 1]
false_data = df_data.loc[df_data['prediction'] == 0]
true_add = df_add.loc[df_add['prediction'] == 1]
false_add = df_add.loc[df_add['prediction'] == 0]

#create a dataset ratio to match test set
false = pd.concat([false_data,false_add], ignore_index=True)
true = true_add.sample(n=int((false.shape[0])*1.25)-true_data.shape[0])

#%%
true = pd.concat([true_data,true], ignore_index=True)
false = pd.concat([false,false], ignore_index=True)
data = pd.concat([false,true], ignore_index=True)

#%%
#print ratio
print(false.shape[0]/(false.shape[0]+true.shape[0]))
labels = data.pop('prediction')
data = pd.concat([data, labels], ignore_index=True, axis =1)
#%%
data.to_csv("C:\\Users\jackt\Desktop\machine_learning_coursework\CleanDataCNN.csv", index = False)

































