import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from data_parser import data_splitter,satellite_info
from constants import DATASET_FOLDER
#from scipy.signal import spectrogram,ShortTimeFFT
import numpy as np
import random
from scipy.fft import fft, fftfreq
from kymatio.numpy import Scattering1D,Scattering2D
from kymatio.datasets import fetch_fsdd
import pywt

# folder="/home/erush/projectSSA/lca_comparison"

def all_sat_plotter(satellite):
    
    class_name,satellite_name,labels,data,track_numbers=satellite_info(satellite)
    
    for idx,d in enumerate(data):
        plt.plot(d,marker='o',linewidth=1.4,markersize=2,label=track_numbers[idx])

    plt.title(f"{satellite_name}-{class_name} Mag Values")
    plt.xlabel("Sample #")
    plt.ylabel("Magnitude")
    plt.xticks(rotation=45)
    plt.tight_layout()    
    plt.gcf().set_size_inches(16, 9)
    plt.savefig(f'rapor_plots/{satellite_name} {class_name}.png',bbox_inches='tight', dpi=200)   # save the figure to file
    plt.close()

def class_comparison_plot(classNames,satellites):
    all_data=[[]*x for x in range(len(satellites))]
    sat_names=[[]*x for x in range(len(satellites))]
    class_names=[[]*x for x in range(len(satellites))]
    for idx,i in enumerate(satellites):
        folder=classNames[idx]
        file=i[0]
        print(file)
        class_name,satellite_name,labels,data,track_numbers=satellite_info(os.path.join(DATASET_FOLDER,f"{classNames[idx]}/{file}"))
        all_data[idx]=(data[0])
        sat_names[idx]=satellite_name
        class_names[idx]=class_name
        
    for idx0,d in enumerate(all_data):
        plt.plot(d,marker='o',linewidth=1.4,markersize=2,label=f"{sat_names[idx0]}-{class_names[idx0]}")
        plt.xlim(0,500)

    plt.title("Class based satellite plot Mag Values")
    plt.xlabel("Sample #")
    plt.ylabel("Magnitude")
    plt.xticks(rotation=45)
    plt.tight_layout()    
    plt.gcf().set_size_inches(16, 9)
    
    plt.legend()
    plt.savefig(f'rapor_plots/AllClasses{idx0}raaa.png',bbox_inches='tight', dpi=200)   # save the figure to file
    
def main_class_comparison_plot():
    main_classes=['SATELLITE','ROCKETBODY','DEBRIS']
    colors=['red','blue','green']
    fig, axs = plt.subplots(3)
    for idx,cls in enumerate(main_classes):
        folder_path=os.path.join(DATASET_FOLDER,cls)
        folder=sorted(os.listdir(folder_path))
        color=colors[idx]
        axs[0].plot([],[],color=color,label=cls)
        axs[0].set_title('Original')
        axs[1].plot([],[],color=color,label=cls)
        axs[1].set_title('cA')
        axs[2].plot([],[],color=color,label=cls)
        axs[2].set_title('cD')
        for i in range(min(7,len(folder))):
            class_name,satellite_name,labels,data,track_numbers=satellite_info(
                os.path.join(folder_path,folder[i]))
            for j in range(min(3,len(data))):
                tempData=data[j]
                axs[0].set_xlim(0,50)
                axs[1].set_xlim(0,50)
                axs[2].set_xlim(0,50)
                cA,cD=wwt_transform(tempData)
                axs[0].scatter(tempData,color=color)
                axs[1].scatter(cA,color=color)            
                axs[2].scatter(cD,color=color)
                
    fig.suptitle("Discrete Wavelet Transform")
    # fig.xlabel("Sample #")
    # fig.set_ylabel("Magnitude")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig(f'MainClassesWWTPlots-bior3.3-symmetric.png',bbox_inches='tight', dpi=200)   # save the figure to file
    plt.show()
        

def spectogram_analysis(satellite):
    class_name,satellite_name,labels,data,track_numbers=satellite_info(satellite)
    Fs = 1# Sample rate 

    f,t,Sxx=spectrogram(np.array(data[0]),Fs)
    plt.pcolormesh(t,f,Sxx,shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
def fft_analysis():
    main_classes=['SATELLITE','ROCKETBODY','DEBRIS']
    colors=['red','blue','green']
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    SAMPLE_RATE=10
    for idx,cls in enumerate(main_classes):
        folder_path=os.path.join(DATASET_FOLDER,cls)
        folder=sorted(os.listdir(folder_path))
        color=colors[idx]
        ax.plot([],[],color=color,label=cls)
        for i in range(min(10,len(folder))):
            class_name,satellite_name,labels,data,track_numbers=satellite_info(
                os.path.join(folder_path,folder[i]))
            yf=fft(data[0])
            xf=fftfreq(len(data[0]),1/SAMPLE_RATE)
            ax.plot(xf,np.abs(yf),color=color)

    ax.set_title("FFT_Analysis")
    ax.set_xlabel("Sample #")
    ax.set_ylabel("Magnitude")
    plt.legend()
    plt.gcf().set_size_inches(16, 9)
    plt.savefig(f'FFT_Analylsis-sr10,.png',bbox_inches='tight', dpi=200)   # save the figure to file

    # plt.show()

def all_sat_spectogram_plotter(satellite):
    
    Fs = 1# Sample rate 

    class_name,satellite_name,labels,data,track_numbers=satellite_info(satellite)
    for idx,d in enumerate(data):
        f,t,Sxx=spectrogram(np.array(d),Fs)
        plt.pcolormesh(t,f,Sxx,shading='gouraud')
        
    plt.title(f"{satellite_name}-{class_name} Mag Values")
    plt.xlabel("Sample #")
    plt.ylabel("Magnitude")
    plt.xticks(rotation=45)
    plt.tight_layout()    
    plt.gcf().set_size_inches(16, 9)
    plt.savefig(f'rapor_plots/spectogram_plots/{satellite_name}-{class_name}-{Fs}-{track_numbers[idx]}.png',bbox_inches='tight', dpi=200)   # save the figure to file
    plt.close()


def wavelet_transform_plot(satellite):
    class_name,satellite_name,labels,data,track_numbers=satellite_info(satellite)
    x=np.array(data[0])
    x=x/np.max(abs(x))
    T=x.shape[-1]
    print(T)
    J=6
    Q=(32,2)
    scattering=Scattering1D(J,T,Q)
    Sx=scattering(x)
    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    fig, axs = plt.subplots(4)
    fig.suptitle('Scattering and Original')
    axs[0].plot(Sx[order0][0],label="Sc0")
    axs[1].plot(Sx[order1],label="Sc1")
    axs[2].plot(Sx[order2],label="Sc2")
    axs[3].plot(x)
    plt.show()

def wavelet_transform(data):
    x=np.array(data)
    x=x/np.max(abs(x))
    T=x.shape[-1]
    
    J=6
    Q=16
    scattering=Scattering1D(J,T,Q)
    Sx=scattering(x)
    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)
    return Sx[order0][0]

def wwt_transform(data):
    
    w=pywt.Wavelet('bior3.3')
    cA,cD=pywt.dwt(data,wavelet=w,mode='symmetric')
    # fig, axs = plt.subplots(3)
    # axs[0].plot(cA)
    # axs[0].set_title('cA')
    # axs[1].plot(cD)
    # axs[1].set_title('cD')
    # axs[2].plot(data[0])
    # axs[2].set_title('Original')
    return cA,cD
    plt.show()

    
if __name__ =='__main__':
    plot_save_folder='rapor_plots'
    classNames=sorted(os.listdir(DATASET_FOLDER))
    classNames.remove('MainClasses')
    act_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[0]))
    fdeb_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[1]))
    inact_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[2]))
    mdeb_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[3]))
    rb_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[4]))
    usat_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[5]))
    unident_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[6]))
    satellites=[act_sat,fdeb_sat,inact_sat,mdeb_sat,rb_sat,usat_sat,unident_sat]
    
    # wwt_transform(os.path.join(DATASET_FOLDER,f"{classNames[2]}/{satellites[2][0]}"))
    main_class_comparison_plot()
    # fft_analysis()
    
    
    # spectogram_analysis(os.path.join(DATASET_FOLDER,f"{classNames[2]}/{satellites[2][0]}"))
    # for idx,i in enumerate(satellites):
    #     for j in i:
    #         all_sat_spectogram_plotter(os.path.join(DATASET_FOLDER,f"{classNames[idx]}/{j}")),
            
    # class_comparison_plot(classNames=classNames,satellites=satellites)
                
                        
    
