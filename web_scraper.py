import requests
from bs4 import BeautifulSoup
import os
import urllib.request
from pathlib import Path
import os
from constants import DATASET_FOLDER
from rich.progress import Progress

from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn
from rich.table import Table
from rich.console import Console
from ssaUtils import downloadProgressBar

def class_based_scrapper(soup,data_folder,cls,number,progressBar,periodicFlag=True):
    global satNumberCounter
    sat_cls=["U/SAT","ACT","INACT"]
    deb_cls=["DEB","M/DEB","F/DEB"]
    rb_cls=["R/B"]
    cmp_list=None
    if cls=="SATELLITE":cmp_list=sat_cls
    if cls=="DEBRIS":cmp_list=deb_cls
    if cls=="ROCKETBODY":cmp_list=rb_cls
    links=[]
    names=[]
    f_folder=cls
    object_classes=[]
    folder_ls=os.listdir(f"{data_folder}/{f_folder}")
    for tag in soup.find_all('a',title="Downoad all tracks"):
        if periodicFlag:
            periodicity=tag.parent.parent.find('span',class_='text-default')
            if periodicity:
                    if 'Period' in periodicity.text:        
                        link=tag['href']
                        parent=tag.parent.parent.find_all('span')
                        for i in parent:
                            if i.text in cmp_list:
                                name=link.split('/')[2]
                                links.append(main_url+link)
                                names.append(name)
                                object_classes.append(i.text.replace("/","-"))
        else:
            link=tag['href']
            parent=tag.parent.parent.find_all('span')
            for i in parent:
                if i.text in cmp_list:
                    name=link.split('/')[2]
                    links.append(main_url+link)
                    names.append(name)
                    object_classes.append(i.text.replace("/","-"))

    for idx,link in enumerate(links):
        if satNumberCounter[cls]<number:
            response =requests.get(link,allow_redirects=True,stream=True)
            r.raise_for_status()
            if(f"satellite-{names[idx]}_{object_classes[idx]}.txt") in folder_ls:
                pass
            else:
                with open(f'{data_folder}/{f_folder}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb+') as f:
                    progressBar.process.update(progressBar.processingBar,description=f"[bold yellow]Processing: satellite-{names[idx]}_{object_classes[idx]}")
                    f.write(response.content)
                    
                response.close()

            progressBar.job_progress.advance(task_id=main_classes.index(cls))
            progressBar.updateOverall()    
            satNumberCounter[cls]+=1
        else:
            print(f"{number} reached. Terminating.")
            break


if __name__ =='__main__':
    satNumber=[350,350,350]
    satNumberCounter={"SATELLITE":0,"ROCKETBODY":0,"DEBRIS":0}
    periodic=True
    progressBar=downloadProgressBar(satNumber)
    main_url="http://mmt.favor2.info"
    url='http://mmt.favor2.info/satellites?page='
    if periodic:
        data_folder=os.path.join(DATASET_FOLDER,'Periodic')
    else:
        data_folder=os.path.join(DATASET_FOLDER,'Aperiodic')
    main_classes=["SATELLITE","ROCKETBODY",'DEBRIS']
    database_pages=1223
    with Live(progressBar.progress_table, console=progressBar.console,refresh_per_second=10) as live:
        live.console.print(f"[bold yellow]Folder:{data_folder}")
        for i in range(1,database_pages+1):
            live.console.log(f"Working on page #:{i}")

            for index,cls in enumerate(main_classes[2:3]):
                if satNumberCounter[cls]<satNumber[index]:
                    r=requests.get(url+str(i))
                    soup=BeautifulSoup(r.text,'lxml')
                    class_based_scrapper(soup=soup,data_folder=data_folder,cls=cls,number=satNumber[index],progressBar=progressBar,periodicFlag=periodic)
                else:
                    continue

 