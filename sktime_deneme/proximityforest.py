import sys
sys.path.append("../")
from sktime.classification.distance_based import ProximityForest
from sktime.datasets import load_unit_test
from sklearn.metrics import accuracy_score
import argparse
from mmt import MiniMegaTortoraDataset
from ssaUtils import DiscreteWaveletTransform,pad_to_size_interpolate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import datetime
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

#--------------------Learning Rate Scheduler-------------------------------

mmt=MiniMegaTortoraDataset(satNumber=satelliteNumber,periodic=True)
classes=[[x] for x in mmt.satelliteData]

x,y=mmt.load_data()


# DiscreteWaveletTransform(x)

x = DiscreteWaveletTransform(x, wavelet='db4', level=2)

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
                                            stratify=y)


x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,
                                            shuffle=True,
                                            test_size=0.2,
                                            stratify=y_train)

# Normalization
scaler=preprocessing.StandardScaler()

x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)


clf = ProximityForest(n_estimators=2, max_depth=2, n_stump_evaluations=1)
start_time = datetime.datetime.now()

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
clf_report=classification_report(y_test,y_pred,target_names=["DEBRIS","ROCKETBODY","SATELLITE"],output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :4].T, annot=True,cmap='viridis',ax=axs[1])
cm = confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["DEBRIS","ROCKETBODY","SATELLITE"])
disp.plot(ax=axs[0])
plt.gcf().set_size_inches(16, 9)
plt.show()
print(accuracy_score(y_test, y_pred))