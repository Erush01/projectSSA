import rich.progress
import rich.progress_bar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy
import rich
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn
from rich.live import Live
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
class SSAModel(nn.Module):
    def __init__(self):
        super(SSAModel,self).__init__()
        self.optimizer=None
        self.lossFn=None
        self.device=None
        self.Conv1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),

        )
        self.lstm1=nn.Sequential(
            nn.LSTM(input_size=64,hidden_size=64,batch_first=True,bidirectional=False))
        
        self.Conv2=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=5,stride=2,padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm2=nn.Sequential(
            nn.LSTM(input_size=128,hidden_size=128,batch_first=True,bidirectional=False))

        self.Conv3=nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=512,kernel_size=7,stride=1),
            nn.BatchNorm1d(512),
            nn.Conv1d(in_channels=512,out_channels=512,kernel_size=7,stride=1),
            nn.BatchNorm1d(512),
            nn.Conv1d(in_channels=512,out_channels=512,kernel_size=7,stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        # self.lstm3=nn.Sequential(
        #     nn.LSTM(input_size=23,hidden_size=256,num_layers=1))

        # self.Conv4=nn.Sequential(
        #     nn.Conv1d(in_channels=512,out_channels=1024,kernel_size=5,stride=1),
        #     nn.BatchNorm1d(1024),
        #     nn.Conv1d(in_channels=1024,out_channels=1024,kernel_size=5,stride=1),
        #     nn.BatchNorm1d(1024),
        #     nn.Conv1d(in_channels=1024,out_channels=1024,kernel_size=5,stride=2),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU()
        # )
        self.avgPool1=nn.AdaptiveAvgPool1d(1)

        self.FullyConnected=nn.Sequential(
            nn.Linear(in_features=22144,out_features=128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=128,out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=256,out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features=512,out_features=3),
        )
        
        self.history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
        self.writer=None
    def forward(self,x):

        x=self.Conv1(x)
        # x,_=self.lstm1(x.permute(0, 2, 1))
        # x=F.tanh(x)
        x=self.Conv2(x)
        # x,_=self.lstm2(x.permute(0, 2, 1))
        # x=self.Conv3(x)
        # x,_=self.lstm3(x)
        # x=F.tanh(x)
        # x=self.Conv4(x)
        x=torch.flatten(x,1)

        # x = x.view(-1, x.size(1) * x.size(2))x
        # print(f"Shape:{x.shape}")
        x=self.FullyConnected(x)

        return x
    
    def compile(self,optimizer,loss_fn,device,writer):
        self.optimizer=optimizer
        self.lossFn=loss_fn
        self.device=device
        self.writer=writer
        return True
    
    def __trainOneEpoch(self,trainDL,valDL,epoch_index):
        trainLoss=0.
        trainCorrect=0
        trainTotal=0
        valTotal=0
        valCorrect=0
        valAcc=0
        trainAcc=0
        runningTrainLoss=0
        progressBar = Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TextColumn("[yellow] train_loss: {task.fields[train_loss]} | [yellow]train_acc: {task.fields[train_acc]} | [blue] val_loss: {task.fields[val_loss]} | [blue]val_acc: {task.fields[val_acc]} "),
        TimeElapsedColumn())
        task=progressBar.add_task(f"Epoch: {epoch_index} | Batch:",total=len(trainDL),train_loss=0.00,train_acc=0.0000,val_loss=0.00,val_acc=0.0000)

        with progressBar:
            #Training
            self.train(True)
            for step,data in enumerate(trainDL):
                
                inputs,labels=data[0].to(self.device),data[1].to(self.device)
               
                self.optimizer.zero_grad()
                outputs=self(inputs)
                _,predicted = torch.max(outputs.data,1)
                trainTotal+=labels.size(0)
                trainCorrect+=(predicted==torch.max(labels.data,1).indices).sum().item()
                
                loss=self.lossFn(outputs,labels)
                loss.backward()
                self.optimizer.step()
                runningTrainLoss+=loss.item()*inputs.size(0)
                trainLoss=runningTrainLoss/len(trainDL.dataset)
                trainAcc=100 * trainCorrect / trainTotal 
                progressBar.update(task,advance=1,train_acc=f"{trainAcc:.4f}",train_loss=f"{trainLoss:.2f}")

            #Validation
            runningvalLoss = 0.0
            valLoss=0.0
            self.eval()
            with torch.no_grad():
                for vdata in valDL:
                    vinputs, vlabels = vdata[0].to(self.device),vdata[1].to(self.device)
                    voutputs = self(vinputs)
                    _,valPredicted = torch.max(voutputs.data, 1)
                    valTotal+=vlabels.size(0)
                    valCorrect+=(valPredicted==torch.max(vlabels.data,1).indices).sum().item()
                    vloss = self.lossFn(voutputs, vlabels)
                    runningvalLoss += vloss.item()*vlabels.size(0)

            valLoss=runningvalLoss/len(valDL.dataset)
            valAcc=100*valCorrect/valTotal
            self.history["train_loss"].append(trainLoss)
            self.history["train_acc"].append(trainAcc)
            self.history["val_loss"].append(valLoss)
            self.history["val_acc"].append(valAcc)
            progressBar.update(task,train_acc=f"{trainAcc:.4f}",train_loss=f"{trainLoss:.2f}",val_loss=f"{valLoss:.2f}",val_acc=f"{valAcc:.4f}")
            self.writer.add_scalar("acc/train",trainAcc,epoch_index)
            self.writer.add_scalar("acc/val",valAcc,epoch_index)
            self.writer.add_scalar("loss/train",trainLoss,epoch_index)
            self.writer.add_scalar("loss/val",valLoss,epoch_index)

            return trainLoss,trainAcc
    

    def trainModel(self,trainDL,valDL,numEpochs):
        TotalBar = Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TextColumn("[yellow] train_loss: {task.fields[train_loss]} | [yellow]train_acc: {task.fields[train_acc]} | [blue] val_loss: {task.fields[val_loss]} | [blue]val_acc: {task.fields[val_acc]} "),
        TimeElapsedColumn())
        for epoch in range(numEpochs):
            self.__trainOneEpoch(trainDL,valDL,epoch+1)
            self.writer.flush()

    def predict(self,testSet):
        self.eval()
        predictions=[]
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
                testSet=testSet.to(self.device)
                pred = self(testSet)
                # _,predicted = torch.max(pred.data, 1)
        return pred
    