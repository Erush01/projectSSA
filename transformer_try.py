#!env

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Silence tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
from sklearn.utils import class_weight
from tensorflow.keras import layers
from tensorflow.keras import Model
from io import BytesIO
from mmt import MiniMegaTortora
from ssaUtils import get_summary_str
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return Model(inputs, outputs)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


physical_devices = tf.test.gpu_device_name()
# print(physical_devices)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"gpus are: {gpus}")
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices, True)

#Configurations
satelliteNumber = [45,150,280] # Maximum number of satellites for each class in dataset
trackSize = 1000 # Maximum sample points for each track
epochs = 100 # Number of epochs for training
batchSize = 2 # batch size for training
buffer = BytesIO()
buffer2=BytesIO()

mmt=MiniMegaTortora(satNumber=satelliteNumber,periodic=True)
print_rich(mmt.get_data_rich())
classes=[[x] for x in mmt.satelliteData]#Main classes
print(classes)
dataset=list()
#Add all satellites from each class to bigger set
for i in classes:
    for j in mmt.satelliteData[i[0]]:
        dataset.append(j)

#Shuffle dataset
np.random.shuffle(dataset)

x=list()
y=list()
y_number_class = list()

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

print(y[0:5])

for item in y:
    if item == 'SATELLITE':
        item = y_number_class.append([0])
    if item == 'ROCKETBODY':
        item = y_number_class.append([1])
    if item == 'DEBRIS':
        item = y_number_class.append([2])

print(y_number_class[0:5])

y_number_class = np.array(y_number_class)

cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()



# Train-Val-Test split
x_train,x_test,y_train,y_test=train_test_split(x,y_number_class,
                                               shuffle=True,
                                               test_size=0.2,
                                               stratify=y_number_class)


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

log_dir = "logs/fit/" +"transformer"+datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=3))).strftime("%d %B %Y - %H:%M:%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir+"/train")

model = build_model(
    (trackSize,1),
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batchSize,
    callbacks=[tensorboard_callback,callbacks],
)

model.evaluate(x_test, y_test, verbose=1)

y_pred=model.predict(x_test)
y_pred_str=cat.inverse_transform(y_pred)

y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

fig,axs=plt.subplots(1,2)
clf_report=classification_report(y_test,y_pred,target_names=np.unique(y_pred_str),output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :4].T, annot=True,cmap='viridis',ax=axs[1])
cm = confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_pred_str))
disp.plot(ax=axs[0])
plt.gcf().set_size_inches(16, 9)



plt.savefig(buffer,format='png')
buffer.seek(0)
image = tf.image.decode_png(buffer.getvalue(), channels=4)
image = tf.expand_dims(image, 0)

with file_writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=0)
    tf.summary.text("Data", mmt.get_data(), step=0)
    tf.summary.text("Model",get_summary_str(model),step=0)
    
plt.show()
