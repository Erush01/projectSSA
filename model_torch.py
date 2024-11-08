import torch
import torch.nn as nn
import torch.nn.functional as F
from ssaModel import SSAModel
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from mmt import MiniMegaTortoraDataset
import seaborn as sns
import pandas as pd
from rich import print as print_rich
from io import BytesIO
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform,save_evaluated_lc_plots,pad_to_size_interpolate
from torch.utils.data import TensorDataset,DataLoader
import argparse
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


parser=argparse.ArgumentParser(
        prog="SSA Classifier",
)
parser.add_argument('-d','--debug',action="store_true")
args=parser.parse_args()

if(args.debug): 
    satelliteNumber=[1,1,5]
else:
    satelliteNumber=[40,80,300]

trackSize = 700      # Maximum sample points for each track
EPOCHS = 100    # Number of epochs for training
batchSize = 32        # batch size for training

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#--------------------Learning Rate Scheduler-------------------------------

mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)
classes=[[x] for x in mmt.satelliteData]

x,y=mmt.load_data()

#DiscreteWaveletTransform(x)

x=[pad_to_size_interpolate(array,trackSize) for array in x]


#Numpy array conversion        
x=np.array(x,dtype='object')
y=np.array(y)

cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()
y=torch.Tensor(y)

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

x_train=scaler.fit_transform(x_train)
x_val=scaler.fit_transform(x_val)
x_test=scaler.fit_transform(x_test)


#Only use if you not use ConvLSTM
x_train=np.expand_dims(x_train,axis=-1)
x_val=np.expand_dims(x_val,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)


tensor_train_x=torch.Tensor(x_train)
tensor_train_y=torch.Tensor(y_train)
tensor_train_x=torch.transpose(tensor_train_x,1,2)

tensor_val_x=torch.Tensor(x_val)
tensor_val_y=torch.Tensor(y_val)
tensor_val_x=torch.transpose(tensor_val_x,1,2)

tensor_test_x=torch.Tensor(x_test)
tensor_test_x=torch.transpose(tensor_test_x,1,2)

tensor_TrainDataset=TensorDataset(tensor_train_x,tensor_train_y)
tensor_ValDataset=TensorDataset(tensor_val_x,tensor_val_y)

train_dataloader=DataLoader(tensor_TrainDataset,batch_size=batchSize)
val_dataloader=DataLoader(tensor_ValDataset,batch_size=batchSize)

f_name=datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=3))).strftime("%d %B %Y - %H:%M:%S")
log_dir = "./runs/" + f_name
writer=SummaryWriter(log_dir)


model=SSAModel()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
model.compile(optimizer,loss_fn,device,writer)
model.to(device)
# summary(model.cuda(),input_size=(batchSize,tensor_train_x.shape[1],tensor_train_x.shape[2]))

model.trainModel(train_dataloader,val_dataloader,EPOCHS)
history=model.history


plt.plot(history['train_acc'])
plt.plot(history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history['train_loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# #Test

y_pred=model.predict(tensor_test_x[0:batchSize])
y_pred = y_pred.detach().cpu().numpy() # remove from computational graph to cpu and as numpy

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

plt.show()



