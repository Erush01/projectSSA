
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Silence tensorflow
import tensorflow as tf

import numpy as np
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from mmt import MiniMegaTortora
from collections import Counter
import seaborn as sns
import pandas as pd
from rich import print as print_rich
from rich.panel import Panel
from rich.table import Table
from ssa_utils import RichBarCallBack
from sklearn.utils import class_weight
#Tensorflow settings

physical_devices = tf.test.gpu_device_name()
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Configurations
satelliteNumber = 20 # Maximum number of satellites for each class in dataset
trackSize = 1000 # Maximum sample points for each track
epochs = 100 # Number of epochs for training
batchSize = 20 # batch size for training


mmt=MiniMegaTortora(satNumber=satelliteNumber,periodic=True)
print_rich(mmt.get_data_rich())
classes=[[x] for x in mmt.satelliteData]#Main classes
dataset=list()
#Add all satellites from each class to bigger set
for i in classes:
    for j in mmt.satelliteData[i[0]]:
        dataset.append(j)

#Shuffle dataset
np.random.shuffle(dataset)

x=list()
y=list()

#Parse dataset into tracks and classes
for i in dataset:
    for data in i['data']:
        y.append([i['class']])
        x.append(data[0:trackSize])
    
#Pad tracks to maximum TrackSize
x=[np.pad(x,((0,trackSize-len(x))),mode='constant') for x in x]

#Numpy array conversion        
x=np.array([np.array(val) for val in x])
y=np.array(y)

cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()

# Train-Val-Test split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               shuffle=True,
                                               test_size=0.2,
                                               stratify=y)


x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,
                                             shuffle=True,
                                             test_size=0.2,
                                             stratify=y_train)


# weights = class_weight.compute_class_weight('balanced',
#                                             classes=np.unique(cat.inverse_transform(y_train)), 
#                                             y=[x[0] for x in cat.inverse_transform(y)])                                            
# class_weights = dict(enumerate(weights))
# print(class_weights)


# Normalization
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)

#Expanding dimension to fit Convolutional layer
x_train=np.expand_dims(x_train,axis=-1)
x_val=np.expand_dims(x_val,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

train_table=Table.grid()
train_table.add_row(
            Panel.fit(
                f"[bold green]X: [white]{x_train.shape}\n"
                f"[bold green]Y: [white]{y_train.shape}\n",
                title="[b]Train", border_style="green", padding=(1, 1)),
            Panel.fit(
                f"[bold yellow]X : [white]{x_val.shape}\n"
                f"[bold yellow]Y : [white]{y_val.shape}\n",
                title="[b]Validation", border_style="yellow", padding=(1,1)),
        Panel.fit(f"[bold cyan]X : [white]{x_test.shape}\n"
                f"[bold cyan] Y : [white]{y_test.shape}\n",
                title="[b]Test",border_style="cyan", padding=(1, 1)))

print_rich(train_table)

model2=tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],x_train.shape[2])),
    tf.keras.layers.Conv1D(64,2,activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=(2)),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64,2,activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=(2)),
    tf.keras.layers.LSTM(64,return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3,activation='softmax')])


model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalFocalCrossentropy(),
              metrics=['acc'])

model2.summary()

# Logging metrics using tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#Training model
history=model2.fit(x=x_train,
          y=y_train,
          batch_size=batchSize,
          epochs=epochs,
          validation_data=(x_val,y_val),
          callbacks=[tensorboard_callback],
          verbose=2
            )

#Confusion matrix and F1 score
y_pred=model2.predict(x_test)
y_pred_str=cat.inverse_transform(y_pred)

y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

clf_report=classification_report(y_test,y_pred,target_names=np.unique(y_pred_str),output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,cmap='viridis')
cm = confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_pred_str))
disp.plot()
plt.show()
