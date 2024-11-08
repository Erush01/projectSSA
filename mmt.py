import concurrent.futures
import os
import matplotlib.pyplot as plt
from constants import PICKLE_FOLDER
import numpy as np
import pywt
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
import concurrent
from ssaUtils import RSO,LightCurveReadProgressBar,LightCurve
from rich import print as print_rich
import pickle
class MiniMegaTortoraDataset():
    
    """MMT dataset class"""
    
    def __init__(self,satNumber,periodic=True):
        self.progressBar=LightCurveReadProgressBar(satNumber)

        self.satelliteNumber=satNumber
             
    def multiprocess_read(self):
        picklefile=open("ssaDataset60-160-300.pickle","wb")
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
                                
                    rso=RSO(satellite_name,cls,lightCurves)
                    self.progressBar.job_progress.advance(task_id=mid)
                    self.progressBar.overall_progress.update(self.progressBar.overall_task,advance=1)
                    pickle.dump(rso, picklefile)

    def printData(self,y):
        arr=np.unique(y,return_counts=True)
        names=arr[0]
        counts=arr[1]
        progress_table = Table.grid()
        progress_table.add_row(
            Panel.fit(
                f"[bold green]Object Number: [white]{self.satelliteNumber[0]}\n"
                f"[bold green]Track number: [white]{counts[0]}\n",
                title=f"[b]{names[0]}", border_style="green", padding=(1, 1)
            ),
            Panel.fit(
                f"[bold yellow]Object Number: [white]{self.satelliteNumber[1]}\n"
                f"[bold yellow]Track number: [white]{counts[1]}\n",
                title=f"[b]{names[1]}", border_style="yellow", padding=(1, 1)
            ),
        Panel.fit(
                f"[bold cyan]Object Number: [white]{self.satelliteNumber[2]}\n"
                f"[bold cyan]Track number: [white]{counts[2]}\n",
                title=f"[b]{names[2]}", border_style="cyan", padding=(1, 1)
            ))
        print_rich(progress_table)
        
    def load_data_new(self):
        x=list()
        y=list()
        satnumbers=self.satelliteNumber
        pickle_files=["ssaDataset-SATELLITE-100.pickle","ssaDataset-ROCKETBODY-200.pickle","ssaDataset-DEBRIS-300.pickle"]
        with Live(self.progressBar.progress_table, refresh_per_second=10):
            for idx,file in enumerate(pickle_files):    
                with open(os.path.join(PICKLE_FOLDER,file), "rb") as f:
                        while True:
                            try:
                                if(satnumbers[idx]>0):
                                    rso=pickle.load(f)
                                    for lightcurve in rso.lightCurves:
                                        y.append([rso.type])     
                                        x.append(lightcurve.track)
                                    self.progressBar.job_progress.advance(task_id=idx)
                                    self.progressBar.overall_progress.update(self.progressBar.overall_task,advance=1)
                                    satnumbers[idx]-=1    
                                else:
                                    break
                            except EOFError:
                                break
            return x,y

if __name__ == "__main__":
    mmt=MiniMegaTortoraDataset(periodic=True,satNumber=[10,10,50])
    x,y=mmt.load_data_new()
    mmt.printData(y)