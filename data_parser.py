import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from constants import DATASET_FOLDER
import random


def satellite_info(satellite):
    with open(satellite) as f:
        class_name=satellite.split("/")[6].split('_')[1].split('.')[0]
        satellite_name=f.readlines()[:7][0].split("/")[0].split(":")[1]

    labels,data,track_numbers=data_splitter(satellite)
    return class_name,satellite_name,labels,data,track_numbers

def data_splitter(satellite):
    data_array=[]
    track_numbers=[]
    all_data=[[],[],[]]
    labels=[[],[],[]]
    with open(satellite) as f:
        class_name=satellite.split("/")[6].split('_')[1].split('.')[0]
        lines=f.readlines()[7:]
        for i in lines:
            data_array.append(i.split(' '))
            track_numbers.append(i.split(' ')[9].replace('\n',''))

    track_numbers=set(track_numbers)
    all_data=[[]*x for x in range(len(track_numbers))]
    labels=[[]*x for x in range(len(track_numbers))]
    all_track_numbers=[[]*x for x in range(len(track_numbers))]
    for data in data_array:
        for idx,track in enumerate(track_numbers):
            if track==data[9].replace('\n',''):
                all_data[idx].append(float(data[3]))
                all_track_numbers[idx]=track
            labels[idx]=class_name


    return labels,all_data,all_track_numbers


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
    
    return labels,dataset

def dataPlotter(satellite):
    class_name,satellite_name,labels,data,track_numbers=satellite_info(satellite)
    # idx=random.randint(0,len(data))
    idx=0
    plt.plot(data[idx],marker='o',linewidth=1.0,markersize=3,fillstyle='right')
    plt.title(f"{satellite_name}-{class_name}-{track_numbers[idx]} Mag Values")
    plt.xlabel("Sample #")
    plt.ylabel("Magnitude")
    plt.xticks(rotation=45)
    plt.xlim(0,500)
    plt.tight_layout()
    plt.savefig(f'denemera.png',bbox_inches='tight', dpi=200)   # save the figure to file
    plt.show()



if __name__=='__main__':
    DATA_FOLDER=DATASET_FOLDER
    satellite_folder=sorted(os.listdir(DATA_FOLDER))
    print(satellite_folder)
    folder=satellite_folder[7]
    folder_path=os.path.join(DATA_FOLDER,folder)
    satellite=os.path.join(folder_path,os.listdir(folder_path)[0])    
    # satellite1="/media/erush/data/SSADataset/ACT/satellite-1319_ACT.txt"
    # print(satellite)
    dataPlotter(satellite)

