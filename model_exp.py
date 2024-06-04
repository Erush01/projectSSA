
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Silence tensorflow
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from mmt import MiniMegaTortora
import seaborn as sns
import pandas as pd
from rich import print as print_rich
from rich.panel import Panel
from rich.table import Table
from keras.utils import plot_model
from io import BytesIO
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform,DelayedExponentialDecay,save_evaluated_lc_plots

#Tensorflow settings
physical_devices = tf.test.gpu_device_name()
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

#-------------Configurations---------------------------
satelliteNumber = [45,160,280] # Maximum number of satellites for each class in dataset[Satellite,Rocketbody,Debris]
# satelliteNumber=[1,1,1]
trackSize = 700      # Maximum sample points for each track
epochs = 100     # Number of epochs for training
batchSize = 32        # batch size for training
buffer = BytesIO()
buffer2=BytesIO()
acc_buffer=BytesIO()
loss_buffer=BytesIO()


#--------------------Learning Rate Scheduler-------------------------------

learning_rate=1e-4
final_learning_rate = 5e-6
start_epoch=60
learning_rate_decay_factor = (final_learning_rate /learning_rate)**(1/(epochs-start_epoch))

#-------------------------------------------------------

mmt=MiniMegaTortora(satNumber=satelliteNumber,periodic=True)

print_rich(mmt.get_data_rich())


classes=[[x] for x in mmt.satelliteData]#Main classes
dataset=list()

#Add all satellites from each class to bigger set

for i in classes:
    for j in mmt.satelliteData[i[0]]:
        dataset.append(j)

#Shuffle dataset
# np.random.shuffle(dataset)

x=list()
y=list()

#Parse dataset into tracks and classes
for i in dataset:
    for data in i['data']:
        y.append([i['class']])     
        x.append(data)

#Apply Discrete wavelet transform to dataset
DiscreteWaveletTransform(x)
#Pad tracks to maximum TrackSize
x=[a[0:trackSize] for a in x]
x=[np.pad(a,(0,trackSize-len(a)),mode='symmetric') for a in x]
# x=[np.split(y,4) for y in x] # Only use if using ConvLSTM


#Numpy array conversion        
x=np.array(x,dtype='object')
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


# Normalization
scaler=StandardScaler()

#Use only if you use ConvLSTM
# x_train=[scaler.fit_transform(x) for x in x_train]
# x_val=[scaler.fit_transform(x) for x in x_val]
# x_test=[scaler.fit_transform(x) for x in x_test]

x_train=scaler.fit_transform(x_train)
x_val=scaler.fit_transform(x_val)
x_test=scaler.fit_transform(x_test)

x_train=np.array(x_train)
x_val=np.array(x_val)
x_test=np.array(x_test)

#Expanding dimension to fit Convolutional layer

#Only use if you not use ConvLSTM
x_train=np.expand_dims(x_train,axis=-1)
x_val=np.expand_dims(x_val,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)

print_rich(train_table(x_train,y_train,x_val,y_val,x_test,y_test))


#%97 training accuracy
model=tf.keras.models.Sequential([
    
    tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],x_train.shape[2])),
    
    tf.keras.layers.Conv1D(64,kernel_size=3,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(64,kernel_size=3,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(128,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(128,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
                
    tf.keras.layers.Conv1D(128,kernel_size=5,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
        
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv1D(256,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(256,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(256,kernel_size=5,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256,return_sequences=True)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(512,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(512,kernel_size=5,strides=1,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv1D(512,kernel_size=5,strides=2,activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3,activation='softmax')])


# model=tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(x_train.shape[1],x_train.shape[2],1)),
#     tf.keras.layers.ConvLSTM1D(32,2,return_sequences=True,activation='relu'),
#     tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.ConvLSTM1D(32,2,return_sequences=True,activation='relu'),
#     tf.keras.layers.BatchNormalization(),
    
#     tf.keras.layers.Conv1D(32,2,activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2)),
    
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(64,activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(3,activation='softmax'),
# ])


model.summary()

f_name=datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=3))).strftime("%d %B %Y - %H:%M:%S")
log_dir = "logs/fit/" + f_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir+"/train")

delayed_lr_schedule = DelayedExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=1,
    decay_rate=learning_rate_decay_factor,
    start_epoch=start_epoch,
    log_dir=log_dir)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['acc'])
start_time = datetime.datetime.now()

# Training model
history=model.fit(x=x_train,
          y=y_train,
          batch_size=batchSize,
          epochs=epochs,
          validation_data=(x_val,y_val),
          callbacks=[tensorboard_callback,delayed_lr_schedule],
          verbose=1,
            )


model.save(f'checkpoints/{f_name}.keras')

end_time = datetime.datetime.now()
duration = end_time - start_time
hours, remainder = divmod(duration.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")

# Confusion matrix and F1 score
# model = tf.keras.models.load_model('checkpoints/01 June 2024 - 13:58:06.keras')

y_pred=model.predict(x_test)
y_pred_str=cat.inverse_transform(y_pred)
y_test_str=cat.inverse_transform(y_test)

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
plt.show()


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(acc_buffer,format='png')
plt.show()
acc_buffer.seek(0)
acc_img = tf.image.decode_png(acc_buffer.getvalue(), channels=4)
acc_img = tf.expand_dims(acc_img, 0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(loss_buffer,format='png')
plt.show()

loss_buffer.seek(0)
loss_img = tf.image.decode_png(loss_buffer.getvalue(), channels=4)
loss_img = tf.expand_dims(loss_img, 0)

save_evaluated_lc_plots(x_test,y_test_str,y_pred_str,buffer2)
buffer2.seek(0)
image2 = tf.image.decode_png(buffer2.getvalue(), channels=4)
image2 = tf.expand_dims(image2, 0)

with file_writer.as_default():
    tf.summary.image("Confusion Matrix", image, step=0)
    tf.summary.image("Evaluated Light Curve Plots", image2, step=0)
    tf.summary.image("Accuracy_legend", acc_img, step=0)
    tf.summary.image("Loss_legend", loss_img, step=0)
    tf.summary.text("Data", mmt.get_data(), step=0)
    tf.summary.text("Model",get_summary_str(model),step=0)
    tf.summary.text("Training Duration",f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds",step=0)

