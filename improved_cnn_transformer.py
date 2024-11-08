import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from mmt import MiniMegaTortoraDataset
import seaborn as sns
import pandas as pd
from rich import print as print_rich
from io import BytesIO
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform,save_evaluated_lc_plots,pad_to_size_interpolate,trainingProgressBar
from torch.utils.data import TensorDataset,DataLoader
import argparse
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x):
        # Split into different frequency bands
        low_freq = x[:, :, ::4]  # Downsample by 4
        mid_freq = x[:, :, ::2]  # Downsample by 2
        high_freq = x
        
        # Process each frequency band
        low_freq = self.conv1(low_freq)
        mid_freq = self.conv2(mid_freq)
        high_freq = self.conv3(high_freq)
        
        # Upsample and combine
        low_freq = F.interpolate(low_freq, size=high_freq.shape[-1])
        mid_freq = F.interpolate(mid_freq, size=high_freq.shape[-1])
        
        return low_freq + mid_freq + high_freq

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, base_filters=32, kernel_sizes=[5, 3, 3]):
        super(CNNFeatureExtractor, self).__init__()
        
        self.cnn_layers = nn.ModuleList()
        current_channels = in_channels
        
        for i, k_size in enumerate(kernel_sizes):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=current_channels,
                        out_channels=base_filters * (2**i),
                        kernel_size=k_size,
                        padding='same'
                    ),
                    nn.BatchNorm1d(base_filters * (2**i)),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            )
            current_channels = base_filters * (2**i)
            
        self.output_channels = current_channels
        # Add FPN layer
        self.fpn = FeaturePyramidNetwork(current_channels, current_channels)
        
    def forward(self, x):
        # Expected input shape: (batch_size, sequence_length, channels)
        # Convert to (batch_size, channels, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Apply FPN
        x = self.fpn(x)
            
        # Convert back to (batch_size, sequence_length, channels)
        return x.transpose(1, 2)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )       
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            feature_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.feature_embedding = nn.Linear(feature_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
            for _ in range(num_layers)]
        )        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(self.feature_embedding(x) + self.position_embedding(positions))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size,
        num_classes,
        embed_size=128,
        num_layers=3,
        forward_expansion=2,
        heads=4,
        dropout=0.1,
        device="cuda",
        max_length=1000,
        cnn_base_filters=32,
        cnn_kernel_sizes=[7,5, 3, 3]
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        # Feature extractor with FPN
        self.feature_extractor = CNNFeatureExtractor(
            in_channels=feature_size,
            base_filters=cnn_base_filters,
            kernel_sizes=cnn_kernel_sizes
        )
        
        self.encoder = Encoder(
            feature_size=self.feature_extractor.output_channels,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length
        )
        
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, num_classes)
        )
        
    def make_mask(self, x):
        mask = torch.ones(x.shape[0], 1, 1, x.shape[1]).to(self.device)
        return mask
    
    def forward(self, x):
        # Extract features using CNN with FPN
        x = self.feature_extractor(x)
        
        # Apply transformer
        mask = self.make_mask(x)
        encoder_out = self.encoder(x, mask)
        
        # Classification
        out = self.classifier(encoder_out.mean(dim=1))
        return out

def train_epoch(model, dataloader, criterion, optimizer, device,progressBar):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        X, y = batch

        # y=y.type(torch.LongTensor)
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # _, predicted = torch.max(outputs.data, 1)
        _, predicted = torch.max(torch.sigmoid(outputs.data), 1)

        total += y.size(0)
        # correct += (predicted == y).sum().item()
        correct+=(predicted==torch.max(y.data,1).indices).sum().item()
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        progressBar.update(0,advance=1,train_acc=accuracy,train_loss=avg_loss)

    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device,progressBar):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            # y=y.type(torch.LongTensor)

            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(torch.sigmoid(outputs.data), 1)

            total += y.size(0)
            # correct += (predicted == y).sum().item()
            correct+=(predicted==torch.max(y.data,1).indices).sum().item()
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 *correct / total
            progressBar.update(0,val_acc=accuracy,val_loss=avg_loss)

    return avg_loss, accuracy
# Example usage
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    parser=argparse.ArgumentParser(
        prog="SSA Classifier",
    )
    parser.add_argument('-d','--debug',action="store_true")
    args=parser.parse_args()

    if(args.debug): 
        satelliteNumber=[1,1,5]
    else:
        satelliteNumber=[60,160,300]

    trackSize = 500      # Maximum sample points for each track
    EPOCHS = 100    # Number of epochs for training
    batchSize = 32        # batch size for training
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    #--------------------Learning Rate Scheduler-------------------------------

    mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)
    classes=[[x] for x in mmt.satelliteData]

    x,y=mmt.load_data()
    

    DiscreteWaveletTransform(x)
    
    x=[pad_to_size_interpolate(array,trackSize) for array in x]

    # x=[a[0:trackSize] for a in x]
    # x=[np.pad(a,(0,trackSize-len(a)),mode='symmetric') for a in x]

    #Numpy array conversion        
    x=np.array(x)
    
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
    x_val=scaler.transform(x_val)
    x_test=scaler.transform(x_test)
    

    #Only use if you not use ConvLSTM
    x_train=np.expand_dims(x_train,axis=-1)
    x_val=np.expand_dims(x_val,axis=-1)
    x_test=np.expand_dims(x_test,axis=-1)

    tensor_train_x=torch.Tensor(x_train)
    tensor_train_y=torch.Tensor(y_train)

    tensor_val_x=torch.Tensor(x_val)
    tensor_val_y=torch.Tensor(y_val)

    tensor_test_x=torch.Tensor(x_test)

    tensor_TrainDataset=TensorDataset(tensor_train_x,tensor_train_y)
    tensor_ValDataset=TensorDataset(tensor_val_x,tensor_val_y)

    train_dataloader=DataLoader(tensor_TrainDataset,batch_size=batchSize)
    val_dataloader=DataLoader(tensor_ValDataset,batch_size=batchSize)
    
    model = TimeSeriesTransformer(
        feature_size=1,
        num_classes=3,
        device=device,
        embed_size=128,
        num_layers=3,
        forward_expansion=1,
        heads=8,
        max_length=trackSize,
        cnn_base_filters=32,
        cnn_kernel_sizes=[7,5, 3, 3],
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    num_epochs = EPOCHS  # Increased from 10
    best_accuracy = 0
    best_train_acc=0
    start_time = datetime.datetime.now()

    for epoch in range(num_epochs):
        progressBar=trainingProgressBar(epoch+1,len(train_dataloader))
        with progressBar:
            train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, device,progressBar)
            test_loss, test_acc = evaluate(model, val_dataloader, criterion, device,progressBar)
            
            scheduler.step(test_loss)
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(test_loss)
            history["val_acc"].append(test_acc)
            # print(f"Epoch {epoch+1}/{num_epochs}")
            # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print(f"Best Train Acc: {best_train_acc:.4f}")
    print(f"Best Test Acc: {best_accuracy:.4f}")
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")    
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

    
    

    plt.show()
    # Load best model and perform final evaluation
    # model.load_state_dict(torch.load('best_model.pth'))
    # model.eval()
    # with torch.no_grad():
    #     test_output = model(tensor_test_x.to(device))
    #     _, predicted = torch.max(test_output, 1)
    #     final_accuracy = (predicted == torch.max(y.data,1).indices).float().mean()

    #     print(f"Final Test Accuracy: {final_accuracy:.4f}")

    # print("Model output shape:", test_output.shape)

    # y_pred=test_output
    # y_pred = y_pred.detach().cpu().numpy() # remove from computational graph to cpu and as numpy

    # y_pred_str=cat.inverse_transform(y_pred)
    # y_test_str=cat.inverse_transform(y_test)

    # y_pred=np.argmax(y_pred, axis=1)
    # y_test=np.argmax(y_test, axis=1)
    # fig,axs=plt.subplots(1,2)
    # clf_report=classification_report(y_test,y_pred,target_names=np.unique(y_pred_str),output_dict=True)
    # sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :4].T, annot=True,cmap='viridis',ax=axs[1])
    # cm = confusion_matrix(y_test,y_pred)
    # disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_pred_str))
    # disp.plot(ax=axs[0])
    # plt.gcf().set_size_inches(16, 9)
    # plt.show()
