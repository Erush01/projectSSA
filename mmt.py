import concurrent.futures
import os
import matplotlib.pyplot as plt
from constants import DATASET_FOLDER,PERIODIC_FOLDER,NONPERIODIC_FOLDER,PERIODIC_FOLDER_FAST
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram,periodogram
from scipy.interpolate import interp1d
import pandas as pd

import pywt
from tqdm import tqdm
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn
from rich.table import Table
import rich
import multiprocessing
import concurrent
import time
import threading
import random
from rich.console import Console

# import librosa
# import librosa.display

# from astropy.timeseries import LombScargle

job_progress = Progress(
"{task.description}",
SpinnerColumn('aesthetic',speed=0.4,style=rich.style.Style(color='yellow')),
BarColumn(),
TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
MofNCompleteColumn()
)

class MiniMegaTortora():
    
    #MMT class for easy plot analysis for dataset
    
    def __init__(self,satNumber=[1,1,1],periodic=True):
        job1 = job_progress.add_task("[bold green]SATELLITE",total=satNumber[0])
        job2 = job_progress.add_task("[bold yellow]ROCKETBODY", total=satNumber[1])
        job3 = job_progress.add_task("[bold cyan]DEBRIS", total=satNumber[2])
        # job4 = job_progress.add_task("[bold magenta]NONROTARY", total=satNumber[3])

        total = sum(task.total for task in job_progress.tasks)
        self.overall_progress = Progress(TextColumn("[progress.description]{task.description}"),
                                         BarColumn(),
                                         TaskProgressColumn(),
                                         MofNCompleteColumn(),
                                         TimeElapsedColumn())
        self.overall_task = self.overall_progress.add_task("[b]All RSO's", total=int(total))
        self.progress_table = Table.grid()
        self.progress_table.add_row(
            Panel.fit(
                self.overall_progress, title="Overall Progress", border_style="green", padding=(2, 2)
            ),
            Panel.fit(job_progress, title="[b]Resident Space Objects", border_style="red", padding=(1, 2)),
        )
        
        self.mainClasses = ['SATELLITE','ROCKETBODY','DEBRIS']
        self.satellites={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.satelliteData={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.trackNumbers={"SATELLITE":0,"ROCKETBODY":0,"DEBRIS":0}
        
        
        # self.mainClasses = ['SATELLITE','ROCKETBODY','DEBRIS','NONROTARY']

        # self.satellites={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[],"NONROTARY":[]}

        # self.satelliteData={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[],"NONROTARY":[]}

        # self.trackNumbers={"SATELLITE":0,"ROCKETBODY":0,"DEBRIS":0,"NONROTARY":0}
        

        self.satelliteNumber=satNumber
        self.column=['Date','Time','StdMag','Mag','Filter','Penumbra','Distance','Phase','Channel','Track']

        if periodic:self.DATASET_FOLDER=PERIODIC_FOLDER_FAST
        else:self.DATASET_FOLDER=NONPERIODIC_FOLDER
        self.db_info()
        self.read_satellites_multiprocess()
        self.get_track_numbers()
        self.sample=None        
        
    def __repr__(self):
        return (f"SATELLITE Number:{len(self.satelliteData['SATELLITE'])}\n"
                f"SATELLITE track number: {self.trackNumbers['SATELLITE']}\n"
                f"ROCKETBODY Number: {len(self.satelliteData['ROCKETBODY'])}\n"
                f"ROCKETBODY track number: {self.trackNumbers['ROCKETBODY']}\n"
                f"DEBRIS Number: {len(self.satelliteData['DEBRIS'])}\n"
                f"DEBRIS track number: {self.trackNumbers['DEBRIS']}\n")
        
    def get_data(self):
        return (f"SATELLITE Number:{len(self.satelliteData['SATELLITE'])}\n"
                f"SATELLITE track number: {self.trackNumbers['SATELLITE']}\n"
                f"ROCKETBODY Number: {len(self.satelliteData['ROCKETBODY'])}\n"
                f"ROCKETBODY track number: {self.trackNumbers['ROCKETBODY']}\n"
                f"DEBRIS Number: {len(self.satelliteData['DEBRIS'])}\n"
                f"DEBRIS track number: {self.trackNumbers['DEBRIS']}\n")
                # f"NONROTARY Number: {len(self.satelliteData['NONROTARY'])}\n"
                # f"NONROTARY track number: {self.trackNumbers['NONROTARY']}\n")
        
    def get_data_rich(self):
        progress_table = Table.grid()
        progress_table.add_row(
            Panel.fit(
                f"[bold green]Object Number: [white]{len(self.satelliteData['SATELLITE'])}\n"
                f"[bold green]Track number: [white]{self.trackNumbers['SATELLITE']}\n",
                title="[b]Satellite", border_style="green", padding=(1, 1)
            ),
            Panel.fit(
                f"[bold yellow]Object Number: [white]{len(self.satelliteData['ROCKETBODY'])}\n"
                f"[bold yellow]Track number: [white]{self.trackNumbers['ROCKETBODY']}\n",
                title="[b]Rocketbody", border_style="yellow", padding=(1,1)),
        Panel.fit(f"[bold cyan]ObjectNumber: [white]{len(self.satelliteData['DEBRIS'])}\n"
                f"[bold cyan]Track number: [white]{self.trackNumbers['DEBRIS']}\n",
                title="[b]Debris",border_style="cyan", padding=(1, 1))),
        
        # Panel.fit(f"[bold magenta]ObjectNumber: [white]{len(self.satelliteData['NONROTARY'])}\n"
        #         f"[bold magenta]Track number: [white]{self.trackNumbers['NONROTARY']}\n",
        #         title="[b]NONROTARY",border_style="magenta", padding=(1, 1)))
        return progress_table
      
    def db_info(self):
        for cls in self.mainClasses:
            folder=os.path.join(self.DATASET_FOLDER,cls)
            folders=os.listdir(folder)
            sorted(folders)
            # np.random.shuffle(folders)
            # sorted(folders)
            for file in folders:
                self.satellites[cls].append(file)
    
    def get_track_numbers(self):
        for i in self.satelliteData:
            for sat in self.satelliteData[i]:
                self.trackNumbers[i]+=len(sat['data'])
                
                
    def read_satellites(self):
        with Live(self.progress_table, refresh_per_second=10):
            satNum=self.satelliteNumber
            for mid,cls in enumerate(self.mainClasses): #SAT-R/B-DB
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
                    job_progress.advance(task_id=mid)
                    self.overall_progress.update(self.overall_task,advance=1)
                    self.satelliteData[cls].append({"name":satellite_name,"class":cls,"data":all_data})


    def read_satellites_multiprocess(self): 
        with Live(self.progress_table, refresh_per_second=10):
            with concurrent.futures.ThreadPoolExecutor(16) as exe:
                exe.map(self.multiprocess_read,self.mainClasses)
              
                          
    def multiprocess_read(self,cls):
        mid=self.mainClasses.index(cls)
        satNum=self.satelliteNumber
        folder=os.path.join(self.DATASET_FOLDER,cls)
        for idx in range(min(len(self.satellites[cls]),satNum[mid])):
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
            job_progress.advance(task_id=mid)
            self.overall_progress.update(self.overall_task,advance=1)
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
        plt.tight_layout()    
        plt.gcf().set_size_inches(5, 5)
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
            plt.plot(data[i],marker='o',linewidth=1.0,markersize=1,label=i)

        plt.title(f"{name}-{label}")
        plt.xlabel("Sample #")
        plt.ylabel("Apperent Magnitude")
        plt.xticks(rotation=45)
        plt.tight_layout()    
        plt.gcf().set_size_inches(8, 3)
    
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
    
    
    def discreteWaveletTransform(self,cls):
        sat=self.satelliteData[cls][0]
        name=sat["name"]
        label=sat["class"]
        data=np.array(sat["data"][1])
        w=pywt.Wavelet('haar')
        cA,cD= pywt.dwt(data,w,'constant')
        fig, axs = plt.subplots(2)
        axs[0].plot(data,marker='o',linewidth=1.2,markersize=3)
        axs[0].set_title(f'Original Light Curve')
        axs[1].plot(cA,marker='o',linewidth=1.2,markersize=3)
        axs[1].set_title('Discrete Wavelet Transformed')
        plt.xticks(rotation=45)
        fig.suptitle(f"{name}-{label}")
        plt.gcf().set_size_inches(16, 9)    
        plt.tight_layout()    

        plt.show()
        
    def basicDWT(self,track):
        w=pywt.Wavelet('haar')
        cA,cD= pywt.dwt(track,w,'constant')
        return cA
    
    def interpolation(self,cls):
        sat=self.satelliteData[cls][0]
        name=sat["name"]
        label=sat["class"]
        data=np.array(sat["data"][0])
        plt.plot(data)
        if len(data)<1000:
            data=interp1d(np.arange(1000),data,'linear')
            plt.plot(data)
    
    def get_average_track_length(self):
        counter=0
        track_length=[0,0,0]
        track_number=[0,0,0]
        for idx,cls in enumerate(self.mainClasses):
            for sat in self.satelliteData[cls]:
                for track in sat['data']:
                    track_length[idx]+=len(track)
                    track_number[idx]+=1
        print(track_length)
        print(track_number)
        print([track_length[idx]/track_number[idx] for idx in range(0,3)])
            
if __name__ == "__main__":
  
    
    mmt=MiniMegaTortora(periodic=True,satNumber=[45,160,280])
    # mmt.discreteWaveletTransform("SATELLITE")
    mmt.get_average_track_length()
    # mmt.getSatellitebyName('SATELLITE'," 37165 YAOGAN 11 ")
    # mmt.plot_single_track("SATELLITE",trackIdx=0)
    # mmt.plot_class_comparison(nTracks=2,nSats=1),
    # mmt.class_comparison_subplot()

    # mmt.class_comparison_subplot_name(trackId=3)
    
