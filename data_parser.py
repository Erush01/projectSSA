import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from constants import DATASET_FOLDER


def data_splitter(satellite):
    data_array=[]
    track_numbers=[]
    all_data=[[],[],[]]
    labels=[[],[],[]]
    with open(satellite) as f:
        class_name=satellite.split("/")[5].split('_')[1].split('.')[0]
        lines=f.readlines()[7:]
        for i in lines:
            data_array.append(i.split(' '))
            track_numbers.append(i.split(' ')[9].replace('\n',''))

    track_numbers=set(track_numbers)
    all_data=[[]*x for x in range(len(track_numbers))]
    labels=[[]*x for x in range(len(track_numbers))]

    for data in data_array:
        for idx,track in enumerate(track_numbers):
            if track==data[9].replace('\n',''):
                all_data[idx].append(float(data[3]))

            labels[idx]=class_name


    return labels,all_data


def data_plotter(magnitude_array):

    plt.plot(magnitude_array, marker='o', linestyle='-')
    plt.title('Mag Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.xticks(rotation=45)
    plt.xticks(range(0,len(magnitude_array),10))
    plt.tight_layout()
    plt.show()

def dataset_creator(satellite_datas):
    dataset=list()
    labels=list()
    for idx,satellite in enumerate(satellite_datas):
        file_path=os.path.join(DATA_FOLDER,satellite)
        temp_label,temp_data=data_splitter(file_path)
        dataset.append(temp_data)
        labels.append(temp_label)
        if idx==10:
            break
    
    return labels,dataset

if __name__=='__main__':
    DATA_FOLDER=DATASET_FOLDER
    satellite_datas=os.listdir(DATA_FOLDER)

    dataset_creator(satellite_datas)
    # dataset=list()
    # labels=list()
    # file_path=os.path.join(DATA_FOLDER,satellite_datas[7])
    # labels,dataset=data_splitter(file_path)

