import sys
sys.path.append("../")
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.sklearn import RotationForest
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from mmt import MiniMegaTortoraDataset
from rich import print as print_rich
from ssaUtils import DiscreteWaveletTransform,pad_to_size_interpolate
import argparse
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

x = DiscreteWaveletTransform(x, wavelet='db4', level=3)

x=[pad_to_size_interpolate(array,trackSize) for array in x]

#Numpy array conversion        
x=np.array(x)

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

x_train=scaler.fit_transform(x_train)
x_val=scaler.transform(x_val)
x_test=scaler.transform(x_test)




clf = HIVECOTEV2(
    stc_params={
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 100,
        "max_shapelets": 10,
        "batch_size": 20,
    },
    drcif_params={"n_estimators": 2, "n_intervals": 2, "att_subsample_size": 2},
    arsenal_params={"num_kernels": 50, "n_estimators": 3},
    tde_params={
        "n_parameter_samples": 10,
        "max_ensemble_size": 3,
        "randomly_selected_params": 5,
    },
)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))