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
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform1,DiscreteWaveletTransform2,WaveletFeatureExtractor,save_evaluated_lc_plots,pad_to_size_interpolate,trainingProgressBar
from torch.utils.data import TensorDataset,DataLoader
import argparse
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
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
        
    def forward(self, x):
        # Expected input shape: (batch_size, sequence_length, channels)
        # Convert to (batch_size, channels, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        
        for layer in self.cnn_layers:
            x = layer(x)
            
        # Convert back to (batch_size, sequence_length, channels)
        return x.transpose(1, 2)
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        
        # Single linear layer for all projections
        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embed_size, embed_size)
        self.output_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_size = x.size()
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)
        
        # Combine heads
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, seq_length, embed_size)
        
        return self.output_dropout(self.output(out))
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

class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion_factor=4, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * expansion_factor, embed_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1, expansion_factor=4):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, expansion_factor, dropout)

    def forward(self, x, mask=None):
        # Pre-LayerNorm architecture
        attended = self.attention(self.norm1(x), mask)
        x = x + attended
        
        forwarded = self.feed_forward(self.norm2(x))
        out = x + forwarded
        
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]    
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


class ImprovedTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_size,
        num_classes,
        embed_size=128,
        depth=3,
        heads=8,
        dropout=0.1,
        expansion_factor=2,
        max_length=1000,
        cnn_base_filters=32,
        cnn_kernel_sizes=[7, 5, 3, 3]
    ):
        super(ImprovedTimeSeriesTransformer, self).__init__()
        
        self.feature_extractor = CNNFeatureExtractor(
            in_channels=feature_size,
            base_filters=cnn_base_filters,
            kernel_sizes=cnn_kernel_sizes
        )
        
        # Project features to embedding dimension
        self.input_projection = nn.Linear(self.feature_extractor.output_channels, embed_size)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_size, max_length)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_size=embed_size,
                heads=heads,
                dropout=dropout,
                expansion_factor=expansion_factor
            ) for _ in range(depth)
        ])
        
        # Output head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, embed_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, num_classes)
        )
        
    def forward(self, x):
        # Extract features using CNN
        x = self.feature_extractor(x)
        
        # Project to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)

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

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)    
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
    EPOCHS = 200    # Number of epochs for training
    batchSize = 32        # batch size for training
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    #--------------------Learning Rate Scheduler-------------------------------

    mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)
    classes=[[x] for x in mmt.satelliteData]

    x,y=mmt.load_data()
    

    # DiscreteWaveletTransform1(x)
    
    x = DiscreteWaveletTransform2(x, wavelet='db4', level=3)


    x=[pad_to_size_interpolate(array,trackSize) for array in x]
    # x=[a[0:trackSize] for a in x]
    # x=[np.pad(a,(0,trackSize-len(a)),mode='symmetric') for a in x]

    #Numpy array conversion        
    x=np.array(x)
    print(x.shape)

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
    
    model = ImprovedTimeSeriesTransformer(
    feature_size=1,
    num_classes=3,
    embed_size=128,
    depth=3,
    heads=8,
    dropout=0.5,
    expansion_factor=2,
    max_length=trackSize,
    cnn_base_filters=32,
    cnn_kernel_sizes=[7,5,5, 3,3, 3]
    )  .to(device)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,  # Initial restart interval
    T_mult=2  # Multiply interval by 2 after each restart
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    model.apply(init_weights)



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
