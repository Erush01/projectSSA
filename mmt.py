import os
import matplotlib.pyplot as plt
from constants import DATASET_FOLDER,PERIODIC_FOLDER
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram,periodogram

from kymatio.numpy import Scattering1D,Scattering2D
from kymatio.datasets import fetch_fsdd
import pywt
from tqdm import tqdm
import lightkurve as lk

import librosa
import librosa.display


class MiniMegaTortora():
    
    #MMT class for easy plot analysis for dataset
    
    def __init__(self,satNumber=1):
        self.mainClasses = ['SATELLITE','ROCKETBODY','DEBRIS']
        self.satellites={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.satelliteData={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.satelliteNumber=satNumber
        self.DATASET_FOLDER=PERIODIC_FOLDER
        self.db_info()
        self.read_satellites()
        self.sample=self.satelliteData["DEBRIS"][0]
        
    def __repr__(self):
        return (f"SATELLITE Number:{len(self.satellites['SATELLITE'])}\n"
                f"ROCKETBODY Number: {len(self.satellites['ROCKETBODY'])}\n"
                f"DEBRIS Number: {len(self.satellites['DEBRIS'])}")
                
    def db_info(self):
        for cls in self.mainClasses:
            folder=os.path.join(self.DATASET_FOLDER,cls)
            for file in sorted(os.listdir(folder)):
                self.satellites[cls].append(file)
    
                
    def read_satellites(self):
        satNum=self.satelliteNumber
        pbar = tqdm(self.mainClasses,colour="green")
        for cls in pbar: #SAT-R/B-DB
            pbar.set_description(f'Processing << {cls} >>' ,refresh=True)
            folder=os.path.join(self.DATASET_FOLDER,cls)
            for idx in range(min(len(self.satellites[cls]),satNum)):
                name=self.satellites[cls][idx]
                class_name=name.split('_')[-1].split('.')[0]
                
                with open(os.path.join(folder,name)) as f:
                    allFile=f.readlines()
                    satellite_name=allFile[:7][0].split("/")[0].split(":")[1]
                    dataPart=allFile[7:]
                    trackNums=sorted(set([x.split(' ')[9].replace('\n','') for x in dataPart]))
                    all_data=[[]*x for x in range(len(trackNums))]

                    for i in dataPart:
                        apperentMag=i.split(' ')[3] #Magnitude, #[2] for standard mag
                        trackNum=i.split(' ')[9].replace('\n','') #TrackNumber
                        for idx,track in enumerate(trackNums):
                            if track==trackNum:
                                all_data[idx].append(float(apperentMag))
                self.satelliteData[cls].append({"name":satellite_name,"class":class_name,"data":all_data})
                                  
    def plot_single_track(self,cls,idx=0):
        
        #Plots one single track from one satellite
        
        sat=self.satelliteData[cls][idx]
        name=sat["name"]
        label=sat["class"]
        data=sat["data"]
        plt.plot(data[0],marker='o',linewidth=1.0,markersize=3)
        plt.title(f"{name}-{label}")
        plt.xlabel("Sample #")
        plt.ylabel("Magnitude")
        plt.xticks(rotation=45)
        plt.xlim(0,500)
        plt.tight_layout()    
        plt.gcf().set_size_inches(16, 9)
        # plt.savefig(f'rapor_plots/spectogram_plots/{satellite_name}-{class_name}-{Fs}-{track_numbers[idx]}.png',bbox_inches='tight', dpi=200)   # save the figure to file
        plt.show()
        plt.close()
        
    def plot_tracks(self,cls,idx=0,nTracks=1,plotAll=False):
        
        #Plots nTracks from one satellite

        sat=self.satelliteData[cls][idx]
        name=sat["name"]
        label=sat["class"]
        data=sat["data"]
        if plotAll:nTracks=len(data)
        
        for i in range(min(nTracks,len(data))):
            plt.plot(data[i],marker='o',linewidth=1.0,markersize=2,label=i)

        plt.title(f"{name}-{label}")
        plt.xlabel("Sample #")
        plt.ylabel("Magnitude")
        plt.xticks(rotation=45)
        plt.xlim(0,500)
        plt.tight_layout()    
        plt.gcf().set_size_inches(16, 9)
    
        # plt.savefig(f'rapor_plots/spectogram_plots/{satellite_name}-{class_name}-{Fs}-{track_numbers[idx]}.png',bbox_inches='tight', dpi=200)   # save the figure to file
        plt.legend()    
        plt.show()
        plt.close()
     
    def plot_class_comparison(self,nTracks=1,nSats=1):
        
        #Plots nTracks from nSats from each class
        
        colors=['red','blue','green']

        for idx,cls in enumerate(self.mainClasses):
            color=colors[idx]
            plt.plot([],[],label=cls,color=color)
            for j in range(min(nSats,len(self.satelliteData[cls]))):
                sat=self.satelliteData[cls][j]
                for i in range(min(nTracks,len(sat["data"]))):
                    plt.plot(sat["data"][i],color=color)
                    plt.xlim(0,500)
    
        plt.xlabel("Sample #")
        plt.ylabel("Magnitude")
        plt.xticks(rotation=45)
        plt.tight_layout()    
        plt.gcf().set_size_inches(16, 9)    
        # plt.savefig(f'rapor_plots/spectogram_plots/{satellite_name}-{class_name}-{Fs}-{track_numbers[idx]}.png',bbox_inches='tight', dpi=200)   # save the figure to file
        plt.legend()    
        plt.show()
        plt.close()
        
    
    def spectogram_analysis_1(self,cls):
        sat=self.sample
        name=sat["name"]
        label=sat["class"]
        data=np.array(sat["data"][0])
        
        data.reshape(-1)
        sR=10
        stft=librosa.stft(data,n_fft=256)
        spectrogram=np.abs(stft)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sR, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.show()
        
    
    def light_kurwa(self):
        sat=self.sample
        name=sat["name"]
        label=sat["class"]
        data=np.array(sat["data"][0])
        
        lc=lk.LightCurve(time=range(len(data)),flux=data).to_periodogram('ls').plot()

        # lc=lc.flatten(window_length=401)
        plt.show()
    
    def periodogram_analysis(self):
        sat=self.sample
        name=sat["name"]
        label=sat["class"]
        data=np.array(sat["data"][0])
        
        f, Pxx_den=periodogram(data,fs=10)
        plt.semilogy(f, Pxx_den)
        plt.show()
        
if __name__ == "__main__":
    mmt=MiniMegaTortora(satNumber=1)
    # mmt.plot_tracks("ROCKETBODY",plotAll=True)
    # mmt.plot_class_comparison(nTracks=5,nSats=3)
    # mmt.light_kurwa()
    # mmt.periodogram_analysis()
