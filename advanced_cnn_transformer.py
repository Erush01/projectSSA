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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(InceptionBlock1D, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, kernel_size=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )
        
        # 1x1 followed by 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, kernel_size=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )
        
        # 1x1 followed by 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, n_filters, kernel_size=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )
        
        # Max pooling followed by 1x1 convolution branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1),
            nn.BatchNorm1d(n_filters),
            nn.ReLU()
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection with 1x1 conv if needed
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        return F.relu(out)

class AttentionPool1d(nn.Module):
    def __init__(self, in_channels):
        super(AttentionPool1d, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        weights = self.attention(x)
        return torch.sum(x * weights, dim=2)

class AdvancedCNNFeatureExtractor(nn.Module):
    def __init__(
        self, 
        in_channels, 
        base_filters=32, 
        n_inception_blocks=2,
        n_residual_blocks=2,
        kernel_sizes=[7, 5, 3],
        use_dilated_conv=True
    ):
        super(AdvancedCNNFeatureExtractor, self).__init__()
        
        self.activation_maps = {}  # Store activations for visualization
        
        # Initial convolution with larger kernel for capturing longer patterns
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, padding='same'),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Inception blocks
        self.inception_blocks = nn.ModuleList([
            InceptionBlock1D(
                base_filters if i == 0 else base_filters * 4,
                base_filters
            ) for i in range(n_inception_blocks)
        ])
        
        # Residual blocks with dilated convolutions
        self.residual_blocks = nn.ModuleList()
        current_channels = base_filters * 4
        
        for i in range(n_residual_blocks):
            if use_dilated_conv:
                dilation = 2 ** i
                self.residual_blocks.append(
                    ResidualBlock1D(
                        current_channels,
                        current_channels * 2,
                        kernel_size=3
                    )
                )
                current_channels *= 2
            else:
                self.residual_blocks.append(
                    ResidualBlock1D(
                        current_channels,
                        current_channels,
                        kernel_size=3
                    )
                )
        
        self.output_channels = current_channels
        
        # Optional attention pooling
        self.attention_pool = AttentionPool1d(self.output_channels)
        
        # Hook for feature visualization
        self.hooks = []
        
    def _hook_function(self, name):
        def hook(module, input, output):
            self.activation_maps[name] = output.detach()
        return hook
        
    def register_visualization_hooks(self):
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Register new hooks
        self.hooks.append(self.input_conv[0].register_forward_hook(
            self._hook_function('input_conv')))
        
        for i, block in enumerate(self.inception_blocks):
            self.hooks.append(block.register_forward_hook(
                self._hook_function(f'inception_block_{i}')))
            
        for i, block in enumerate(self.residual_blocks):
            self.hooks.append(block.register_forward_hook(
                self._hook_function(f'residual_block_{i}')))
    
    def get_activation_maps(self):
        return self.activation_maps
        
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)  # Convert to (batch_size, channels, sequence_length)
        
        x = self.input_conv(x)
        
        # Apply inception blocks
        for inception_block in self.inception_blocks:
            x = inception_block(x)
        
        # Apply residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Convert back to (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)
        return x

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
        embed_size=128,  # Increased for more capacity
        num_layers=3,
        forward_expansion=2,
        heads=8,  # Increased for better attention
        dropout=0.2,  # Increased for better regularization
        device="cuda",
        max_length=1000,
        cnn_base_filters=32,
        n_inception_blocks=2,
        n_residual_blocks=2,
        use_dilated_conv=True
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        # Enhanced CNN feature extractor
        self.feature_extractor = AdvancedCNNFeatureExtractor(
            in_channels=feature_size,
            base_filters=cnn_base_filters,
            n_inception_blocks=n_inception_blocks,
            n_residual_blocks=n_residual_blocks,
            use_dilated_conv=use_dilated_conv
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
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, num_classes)
        )
        
    def make_mask(self, x):
        mask = torch.ones(x.shape[0], 1, 1, x.shape[1]).to(self.device)
        return mask
    
    def forward(self, x):
        x = self.feature_extractor(x)
        mask = self.make_mask(x)
        encoder_out = self.encoder(x, mask)
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
        accuracy = 100* correct / total
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
            accuracy = 100*correct / total
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
        dropout=0.5
    ).to(device)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    num_epochs = EPOCHS  # Increased from 10
    best_accuracy = 0
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
            
            # print(f"Epoch {epoch+1}/{num_epochs}")
            # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            # print(f"Best Test Acc: {best_accuracy:.4f}")
            # print()
    print(f"Best Test Acc: {best_accuracy:.4f}")
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds") 
    # Load best model and perform final evaluation
    # model.load_state_dict(torch.load('best_model.pth'))
    # model.eval()
    # with torch.no_grad():
    #     test_output = model(X_test.to(device))
    #     _, predicted = torch.max(test_output, 1)
    #     final_accuracy = (predicted == y_test.to(device)).float().mean()
    #     print(f"Final Test Accuracy: {final_accuracy:.4f}")

    # print("Model output shape:", test_output.shape)