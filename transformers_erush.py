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
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform,save_evaluated_lc_plots
from torch.utils.data import TensorDataset,DataLoader
import argparse
from torchinfo import summary
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import math


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

DiscreteWaveletTransform(x)

x=[a[0:trackSize] for a in x]
x=[np.pad(a,(0,trackSize-len(a)),mode='symmetric') for a in x]

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
# tensor_train_x=torch.transpose(tensor_train_x,1,2)

tensor_val_x=torch.Tensor(x_val)
tensor_val_y=torch.Tensor(y_val)
# tensor_val_x=torch.transpose(tensor_val_x,1,2)

tensor_test_x=torch.Tensor(x_test)
tensor_test_y=torch.Tensor(y_test)
# tensor_test_x=torch.transpose(tensor_test_x,1,2)

# tensor_TrainDataset=TensorDataset(tensor_train_x,tensor_train_y)
# tensor_ValDataset=TensorDataset(tensor_val_x,tensor_val_y)

# train_dataloader=DataLoader(tensor_TrainDataset,batch_size=batchSize)
# val_dataloader=DataLoader(tensor_ValDataset,batch_size=batchSize)


# Define the Transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
seq_length = 700
input_dim = 1
num_classes = 3


# Convert to PyTorch tensors

# Initialize the model
model = TimeSeriesTransformer(input_dim, num_classes)
model.to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 10
batch_size = 1

for epoch in range(EPOCHS):
    model.train()
    for i in range(0, len(tensor_train_x), batch_size):
        batch_X = tensor_train_x[i:i+batch_size].to(device)
        batch_y = tensor_train_y[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(tensor_test_x.to(device))
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = (predicted==torch.max(tensor_test_y.to(device).data,1).indices).float().mean()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(tensor_test_x.to(device))
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted==torch.max(tensor_test_y.to(device).data,1).indices).float().mean()
    print(f'Final Test Accuracy: {accuracy.item():.4f}')