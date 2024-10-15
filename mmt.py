import concurrent.futures
import os
import matplotlib.pyplot as plt
from constants import DATASET_FOLDER,PERIODIC_FOLDER,NONPERIODIC_FOLDER,PERIODIC_FOLDER_FAST
import numpy as np
import pywt
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import concurrent
from ssaUtils import RSO,LightCurveReadProgressBar,LightCurve

class MiniMegaTortoraDataset():
    
    """MMT dataset class"""
    
    def __init__(self,satNumber,periodic=True):
        self.progressBar=LightCurveReadProgressBar(satNumber)
        
        self.mainClasses = ['SATELLITE','ROCKETBODY','DEBRIS']
        self.satellites={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.satelliteData={"SATELLITE":[],"ROCKETBODY":[],"DEBRIS":[]}
        self.trackNumbers={"SATELLITE":0,"ROCKETBODY":0,"DEBRIS":0}

        self.satelliteNumber=satNumber

        if periodic:self.DATASET_FOLDER=PERIODIC_FOLDER_FAST
        else:self.DATASET_FOLDER=NONPERIODIC_FOLDER
        self.db_info()
        # self.read_satellites_multiprocess()
        self.multiprocess_read()
        self.get_track_numbers()
        
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
        return progress_table
      
    def db_info(self):
        for cls in self.mainClasses:
            folder=os.path.join(self.DATASET_FOLDER,cls)
            folders=os.listdir(folder)
            sorted(folders)

            for file in folders:
                self.satellites[cls].append(file)
    
    def get_track_numbers(self):
        for i in self.satelliteData:
            for sat in self.satelliteData[i]:
                self.trackNumbers[i]+=len(sat.lightCurves)
            
    def read_satellites_multiprocess(self): 
       
            with concurrent.futures.ThreadPoolExecutor(16) as exe:
                exe.map(self.multiprocess_read,self.mainClasses)
                                      
    def multiprocess_read(self):
        with Live(self.progressBar.progress_table, refresh_per_second=10):
            for cls in self.mainClasses:
                mid=self.mainClasses.index(cls)
                satNum=self.satelliteNumber
                folder=os.path.join(self.DATASET_FOLDER,cls)
                for idx in range(min(len(self.satellites[cls]),satNum[mid])):
                    name=self.satellites[cls][idx]
                    with open(os.path.join(folder,name)) as f:
                        allFile=f.readlines()
                        satellite_name=allFile[:7][0].split("/")[0].split(":")[1]
                        dataPart=allFile[8:]
                        trackNums=sorted(set([x.split(' ')[9].replace('\n','') for x in dataPart]))
                        lightCurves=[LightCurve(rso=satellite_name,trackID=x,track=[]) for x in trackNums]

                        for i in dataPart:
                            apperentMag=i.split(' ')[3] #Magnitude, #[2] for standard mag
                            trackNum=i.split(' ')[9].replace('\n','') #TrackNumber
                            for lc in lightCurves:
                                if lc.trackID==trackNum:
                                    lc.track.append(float(apperentMag))
                                
                            
                    self.progressBar.job_progress.advance(task_id=mid)
                    self.progressBar.overall_progress.update(self.progressBar.overall_task,advance=1)
                    self.satelliteData[cls].append(RSO(satellite_name,cls,lightCurves))

    def discreteWaveletTransformPlot(self,cls):
        sat=self.satelliteData[cls][0]
        name=sat.name
        label=sat.type
        print(len(sat.lightCurves))
        data=np.array(sat.lightCurves[0].track)
        w=pywt.Wavelet('haar')
        cA,cD= pywt.dwt(data,w,'constant')
        fig, axs = plt.subplots(2)
        axs[0].plot(data,marker='o',linewidth=1.2,markersize=3)
        axs[0].set_title(f'Original Light Curve')
        axs[1].plot(cA,marker='o',linewidth=1.2,markersize=3)
        axs[1].set_title('Discrete Wavelet Transformed')
        plt.xticks(rotation=45)
        fig.suptitle(f"{name}-{label}-{sat.lightCurves[0].trackID}")
        plt.gcf().set_size_inches(16, 9)    
        plt.tight_layout()    

        plt.show()
        
    

if __name__ == "__main__":
    mmt=MiniMegaTortoraDataset(periodic=True,satNumber=[1,1,1])
    # print(len(mmt.satellites["ROCKETBODY"]))
    # mmt.satelliteData["DEBRIS"][0].lightCurves[1].plot()

    
