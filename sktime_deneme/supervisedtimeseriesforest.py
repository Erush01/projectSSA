from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.datasets import load_unit_test
from sklearn.metrics import accuracy_score
import sys
sys.path.append("../")
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.sklearn import RotationForest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mmt import MiniMegaTortoraDataset
import seaborn as sns
import pandas as pd
from rich import print as print_rich
from io import BytesIO
from ssaUtils import get_summary_str,train_table,DiscreteWaveletTransform,DiscreteWaveletTransform1,save_evaluated_lc_plots,pad_to_size_interpolate,trainingProgressBar
import argparse
from torchinfo import summary
import datetime

    
    
parser=argparse.ArgumentParser(
    prog="SSA Classifier",
)
parser.add_argument('-d','--debug',action="store_true")
args=parser.parse_args()

if(args.debug): 
    satelliteNumber=[5,5,100]
else:
    satelliteNumber=[60,160,300]

trackSize = 500      # Maximum sample points for each track
EPOCHS = 100    # Number of epochs for training
batchSize = 32        # batch size for training
history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#--------------------Learning Rate Scheduler-------------------------------

mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)
classes=[[x] for x in mmt.satelliteData]

x,y=mmt.load_data()


DiscreteWaveletTransform1(x)

# x = DiscreteWaveletTransform(x, wavelet='db4', level=2)

x=[pad_to_size_interpolate(array,trackSize) for array in x]

#Numpy array conversion        
x=np.array(x)

y=np.array(y)

cat=preprocessing.OneHotEncoder().fit(classes)
y=cat.transform(y).toarray()
y=np.argmax(y,1)

# Train-Val-Test split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                            shuffle=True,
                                            test_size=0.2,
                                            random_state=42,
                                            stratify=y)


x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,
                                            shuffle=True,
                                            test_size=0.2,
                                            random_state=42,
                                            stratify=y_train)

# Normalization
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)

start_time = datetime.datetime.now()

clf = SupervisedTimeSeriesForest(n_estimators=5)
clf.fit(x_train, y_train)
end_time = datetime.datetime.now()
duration = end_time - start_time
hours, remainder = divmod(duration.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds") 
y_pred = clf.predict(x_test)


correct=(y_pred==y_test).sum().item()
accuracy = 100 *correct / len(x_test)
print(accuracy)
fig,axs=plt.subplots(1,2)
fig.suptitle("SupervisedTimeseriesForest")

clf_report=classification_report(y_test,y_pred,target_names=["SATELLITE","ROCKETBODY","DEBRIS"],output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :4].T, annot=True,cmap='viridis',ax=axs[1])
cm = confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["SATELLITE","ROCKETBODY","DEBRIS"])
disp.plot(ax=axs[0])
plt.gcf().set_size_inches(16, 9)
plt.show()
print(accuracy_score(y_test, y_pred))