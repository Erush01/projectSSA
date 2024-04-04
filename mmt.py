import os
import matplotlib.pyplot as plt
from constants import DATASET_FOLDER,PERIODIC_FOLDER,NONPERIODIC_FOLDER,PERIODIC_FOLDER_FAST
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram,periodogram

from kymatio.numpy import Scattering1D,Scattering2D
from kymatio.datasets import fetch_fsdd
import pywt
from tqdm import tqdm

# import librosa
# import librosa.display

# from astropy.timeseries import LombScargle



class MiniMegaTortora():
    
    #MMT class for easy plot analysis for dataset
    
    def __init__(self,satNumber=1,periodic=True):
        self.mainClasses = ['SATELLITE','ROCKETBODY','DEBRIS']
        self.satellites={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.satelliteData={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.satelliteNumber=satNumber
        if periodic:self.DATASET_FOLDER=PERIODIC_FOLDER_FAST
        else:self.DATASET_FOLDER=NONPERIODIC_FOLDER
        self.db_info()
        self.read_satellites()
        self.sample=None
    def __repr__(self):
        return (f"SATELLITE Number:{len(self.satellites['SATELLITE'])}\n"
                f"{[x['name'] for x in self.satelliteData['SATELLITE']]}\n"
                f"ROCKETBODY Number: {len(self.satellites['ROCKETBODY'])}\n"
                f"{[x['name'] for x in self.satelliteData['ROCKETBODY']]}\n"
                f"DEBRIS Number: {len(self.satellites['DEBRIS'])}\n"
                f"{[x['name'] for x in self.satelliteData['DEBRIS']]}")

                
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
                self.satelliteData[cls].append({"name":satellite_name,"class":cls,"data":all_data})
                                  
    def plot_single_track(self,cls=None,idx=0,trackIdx=0):
        
        #Plots one single track from one satellite
        if cls==None:
            sat=self.sample
        else:
            sat=self.satelliteData[cls][idx]
        name=sat["name"]
        label=sat["class"]
        data=sat["data"]
        plt.plot(data[trackIdx],marker='o',linewidth=1.0,markersize=3)
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
        
    def plot_tracks(self,cls=None,idx=0,nTracks=1,plotAll=False):
        
        #Plots nTracks from one satellite
        
        if cls==None:
            sat=self.sample
        else:
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
                    plt.plot(sat["data"][i],color=color,marker='o',linewidth=1.0,markersize=3)
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
        
        
    def getSatellitebyName(self,cls,name):
        found=False
        for sat in self.satelliteData[cls]:
            if sat['name']==name:
                self.sample=sat
                found=True
                return sat
        if not found:
            print(f"Could not find the data in name {name}")
    
    def class_comparison_subplot(self,satIdx=0,trackIdx=0):
        fig, axs = plt.subplots(3)
        # sat1=self.getSatellitebyName("SATELLITE"," 42058 USA 274 (NOSS 3-8 (A)) ")
        # sat2=self.getSatellitebyName("ROCKETBODY"," 8331 ATLAS CENTAUR R")
        # sat3=self.getSatellitebyName("DEBRIS"," 25746 USA 144 DEB ")
        
        for idx,cls in enumerate(self.mainClasses):
            sat=self.satelliteData[cls][satIdx]
            axs[idx].set_title(f"{sat['name']}-{sat['class']}")
            axs[idx].set_xlim(0,500)
            axs[idx].plot(sat["data"][trackIdx],marker='o',linewidth=1.0,markersize=3)

        plt.xticks(rotation=45)

        plt.gcf().set_size_inches(16, 9)    
        
        plt.show()
        
    def class_comparison_subplot_name(self,trackId=0):
        fig, axs = plt.subplots(3)
        sat1=self.getSatellitebyName("SATELLITE"," 24871 IRIDIUM 920 ")
        sat2=self.getSatellitebyName("ROCKETBODY"," 41858 CZ-2D R")
        sat3=self.getSatellitebyName("DEBRIS"," 8311 THORAD DELTA 1 DEB ")

        # sat1=self.getSatellitebyName("SATELLITE"," 51870 STARLINK-3591 ")
        # sat2=self.getSatellitebyName("ROCKETBODY"," 7688 TITAN 34B R")
        # sat3=self.getSatellitebyName("DEBRIS"," 7091 DELTA 1 DEB ")
        sats=[sat1,sat2,sat3]
        for idx,sat in enumerate(sats):
            axs[idx].set_title(f"{sat['name']}-{sat['class']}")
            axs[idx].set_xlim(0,500)
            axs[idx].plot(sat["data"][trackId],marker='o',linewidth=1.0,markersize=3)
        
        plt.xticks(rotation=45)

        plt.gcf().set_size_inches(16, 9)    
        
        plt.show()

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
        
    def periodogram_analysis_2(self):
        sat=self.sample
        name=sat["name"]
        label=sat["class"]
        data=np.array(sat["data"][0])
        
        t=range(len(data))
        frequency,power = LombScargle(t,data).autopower()
        
        plt.plot(frequency,power)
        plt.show()
    
    
    def discreteWaveletTransform(self,cls,satName):
        sat=self.getSatellitebyName(cls,satName)
        name=sat["name"]
        label=sat["class"]
        data=np.array(sat["data"][0])
        w=pywt.Wavelet('dmey')
        cA,cD= pywt.dwt(data,w,'constant')
        fig, axs = plt.subplots(3)
        axs[0].plot(data,marker='o',linewidth=1.2,markersize=3)
        axs[0].set_title(f'Original-{name}-{label}')
        axs[1].plot(cA,marker='o',linewidth=1.2,markersize=3)
        axs[1].set_title('cA')
        axs[2].plot(cD,marker='o',linewidth=1.2,markersize=3)
        axs[2].set_title('cD')
        plt.xticks(rotation=45)

        plt.gcf().set_size_inches(16, 9)    
        
        plt.show()
        
    
if __name__ == "__main__":
    mmt=MiniMegaTortora(satNumber=10,periodic=False)
    print(mmt)
    # mmt.getSatellitebyName('SATELLITE'," 37165 YAOGAN 11 ")
    # mmt.plot_single_track(trackIdx=4)
    # mmt.plot_class_comparison(nTracks=2,nSats=1),
    mmt.discreteWaveletTransform("SATELLITE"," 51870 STARLINK-3591 ")
    # mmt.class_comparison_subplot()
    # mmt.plot_tracks("SATELLITE",1,5)
    
    # mmt.class_comparison_subplot_name(trackId=3)
    
