import requests
from bs4 import BeautifulSoup
import os
import urllib.request
from tqdm import trange,tqdm
from pathlib import Path
import os
from constants import DATASET_FOLDER
from rich.progress import Progress

from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn
from rich.table import Table

def data_size(data_folder):
    total=0
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            path = Path(root) / file
            total += path.stat().st_size
    return total

def get_data_from_page(soup,data_folder):
    links=[]
    names=[]
    object_classes=[]

    for tag in soup.find_all('a',title="Downoad all tracks"):
        link=tag['href']
        parent=tag.parent.parent.find_all('span')

        links.append(main_url+link)
        names.append(link.split('/')[2])
        for i in parent:
            if i.text in classes:
                object_classes.append(i.text.replace("/","-"))
                break

    pbar = tqdm(links,colour="green")
    for idx,link in enumerate(pbar):
        with open(f'{data_folder}/{object_classes[idx]}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb') as f:
            data_size()
            pbar.set_description(f'Processing << satellite-{names[idx]}_{object_classes[idx]} >> Total Size > {data_size()/1024**2:.2f} MB',refresh=True)
            # pbar.set_postfix_str(f'Processing satellite_{names[idx]}.txt')
            f.write(requests.get(link,allow_redirects=True).content)


def main_class_scraper(soup,data_folder):
    main_classes=['SATELLITE','ROCKETBODY','DEBRIS']
    links=[]
    names=[]
    object_classes=[]

    for tag in soup.find_all('a',title="Download all tracks"):
        
        link=tag['href']
        parent=tag.parent.parent.find_all('span')

        links.append(main_url+link)
        names.append(link.split('/')[2])
        
        for i in parent:
            if i.text in classes:
                object_classes.append(i.text.replace("/","-"))
                

    pbar = tqdm(links,colour="green")
    for idx,link in enumerate(pbar):
        object_cls=object_classes[idx]
        if object_cls=='U-SAT' or  object_cls=='ACT' or  object_cls=='INACT':
            f_folder=main_classes[0]
        if object_cls=='DEB' or  object_cls=='M-DEB' or  object_cls=='F-DEB':
            f_folder=main_classes[2]
        if  object_cls=='R-B':
            f_folder=main_classes[1]
        with open(f'{data_folder}/{f_folder}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb') as f:
            data_size()
            pbar.set_description(f'Processing << satellite-{names[idx]}_{object_classes[idx]} >> Total Size > {data_size(data_folder)/1024**2:.2f} MB',refresh=True)
            # pbar.set_postfix_str(f'Processing satellite_{names[idx]}.txt')
            f.write(requests.get(link,allow_redirects=True).content)


def main_class_periodic_scrapper(soup,data_folder):
    main_classes=['SATELLITE','ROCKETBODY','DEBRIS']
    links=[]
    names=[]
    object_classes=[]

    for tag in soup.find_all('a',title="Downoad all tracks"):
        periodicity=tag.parent.parent.find('span',class_='text-default')
        
        if periodicity:
            if 'Period' in periodicity.text:        
                link=tag['href']
                parent=tag.parent.parent.find_all('span')

                links.append(main_url+link)
                names.append(link.split('/')[2])
                
                for i in parent:
                    if i.text in classes:
                        object_classes.append(i.text.replace("/","-"))
                
    pbar = tqdm(links,colour="green")
    for idx,link in enumerate(pbar):
        object_cls=object_classes[idx]
        f_folder=None
        if object_cls=='U-SAT' or  object_cls=='ACT' or  object_cls=='INACT':
            f_folder=main_classes[0]
        if object_cls=='DEB' or  object_cls=='M-DEB' or  object_cls=='F-DEB':
            f_folder=main_classes[2]
        if  object_cls=='R-B':
            f_folder=main_classes[1]
        if object_cls=='UNIDENT':
            continue    
        with open(f'{data_folder}/{f_folder}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb') as f:
            pbar.set_description(f'Processing << satellite-{names[idx]}_{object_classes[idx]} >> Total Size > {data_size(data_folder)/1024**2:.2f} MB',refresh=True)

            f.write(requests.get(link,allow_redirects=True).content)
   
   
def main_class_aperiodic_scrapper(soup,data_folder):
    main_classes=['SATELLITE','ROCKETBODY','DEBRIS']
    links=[]
    names=[]
    object_classes=[]

    for tag in soup.find_all('a',title="Downoad all tracks"):
        periodicity=tag.parent.parent.find('span',class_='text-default')
        
        if periodicity:
            if 'Aperiodic' in periodicity.text:        
                link=tag['href']
                parent=tag.parent.parent.find_all('span')

                links.append(main_url+link)
                names.append(link.split('/')[2])
                
                for i in parent:
                    if i.text in classes:
                        object_classes.append(i.text.replace("/","-"))
                
    pbar = tqdm(links,colour="green")
    for idx,link in enumerate(pbar):
        object_cls=object_classes[idx]
        f_folder=None
        if object_cls=='U-SAT' or  object_cls=='ACT' or  object_cls=='INACT':
            f_folder=main_classes[0]
        if object_cls=='DEB' or  object_cls=='M-DEB' or  object_cls=='F-DEB':
            f_folder=main_classes[2]
        if  object_cls=='R-B':
            f_folder=main_classes[1]
        if object_cls=='UNIDENT':
            continue    
        with open(f'{data_folder}/{f_folder}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb') as f:
            pbar.set_description(f'Processing << satellite-{names[idx]}_{object_classes[idx]} >> Total Size > {data_size(data_folder)/1024**2:.2f} MB',refresh=True)

            f.write(requests.get(link,allow_redirects=True).content)
      
   
            
def main_class_nonvariable_scrapper(soup,data_folder):
    main_classes=['SATELLITE','ROCKETBODY','DEBRIS']
    links=[]
    names=[]
    object_classes=[]

    for tag in soup.find_all('a',title="Downoad all tracks"):
        periodicity=tag.parent.parent.find('span',class_='text-default')
        if periodicity==None:
            link=tag['href']
            parent=tag.parent.parent.find_all('span')

            links.append(main_url+link)
            names.append(link.split('/')[2])
            
            for i in parent:
                if i.text in classes:
                    object_classes.append(i.text.replace("/","-"))
                
    pbar = tqdm(links,colour="green")
    for idx,link in enumerate(pbar):
        object_cls=object_classes[idx]
        f_folder=None
        if object_cls=='U-SAT' or  object_cls=='ACT' or  object_cls=='INACT':
            f_folder=main_classes[0]
        if object_cls=='DEB' or  object_cls=='M-DEB' or  object_cls=='F-DEB':
            f_folder=main_classes[2]
        if  object_cls=='R-B':
            f_folder=main_classes[1]
        if object_cls=='UNIDENT':
            continue    
        with open(f'{data_folder}/{f_folder}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb') as f:
            pbar.set_description(f'Processing << satellite-{names[idx]}_{object_classes[idx]} >> Total Size > {data_size(data_folder)/1024**2:.2f} MB',refresh=True)

            f.write(requests.get(link,allow_redirects=True).content)
     
def class_based_periodic_scrapper(soup,data_folder,cls="SATELLITE",number=20):
    global data_counter
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
    for tag in soup.find_all('a',title="Downoad all tracks"):
        periodicity=tag.parent.parent.find('span',class_='text-default')
        
        if periodicity:
            if 'Aperiodic' in periodicity.text:        
                link=tag['href']
                parent=tag.parent.parent.find_all('span')

                for i in parent:
                    if i.text in cmp_list:
                        name=link.split('/')[2]
                        links.append(main_url+link)
                        names.append(name)
                        
                        object_classes.append(i.text.replace("/","-"))
                
    for idx,link in enumerate(links):
        if data_counter<number:
            response =requests.get(link,allow_redirects=True,stream=True)
            r.raise_for_status()
            with open(f'{data_folder}/{f_folder}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb+') as f:
                process.update(process1,description=f"[bold yellow]Processing: satellite-{names[idx]}_{object_classes[idx]}")
                f.write(response.content)
                print(main_classes.index(cls))
                job_progress.advance(task_id=main_classes.index(cls))
                overall_progress.update(overall_task,advance=1)
                data_counter+=1
            response.close()

        else:
            print(f"{number} reached. Terminating.")
            
if __name__ =='__main__':
    satNumber=2
    job_progress = Progress(
    "{task.description}",
    SpinnerColumn(),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    MofNCompleteColumn()
    )
    process=Progress(
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        TimeElapsedColumn()
    )
    
    job1 = job_progress.add_task("[bold green]SATELLITE",total=satNumber)
    job2 = job_progress.add_task("[bold yellow]ROCKETBODY", total=satNumber)
    job3 = job_progress.add_task("[bold cyan]DEBRIS",start=True, total=satNumber)
    
    process1=process.add_task("[bold yellow]Processing",start=True,total=None)
    
    total = sum(task.total for task in job_progress.tasks)
    overall_progress = Progress(TextColumn("[progress.description]{task.description}"),
                                
                                         BarColumn(),
                                         TaskProgressColumn(),
                                         MofNCompleteColumn(),
                                         TimeElapsedColumn())
    overall_task = overall_progress.add_task("[b]All RSO's", total=int(total))
    progress_table = Table.grid()
    progress_table.add_row(
        Panel.fit(
            overall_progress, title="Overall Progress", border_style="green", padding=(2, 2)
        ),
        Panel.fit(job_progress, title="[b]Resident Space Objects", border_style="red", padding=(1, 2)),
        Panel.fit(process,title="Processing",border_style='yellow',padding=(2, 2))
    )
    
    main_url="http://mmt.favor2.info"
    url='http://mmt.favor2.info/satellites?page='
    data_folder=os.path.join(DATASET_FOLDER,'Aperiodic')
    main_classes=["SATELLITE","ROCKETBODY",'DEBRIS']

    classes=['U/SAT','ACT','INACT','R/B','DEB','M/DEB','F/DEB','UNIDENT']
    database_pages=1077
    
    with Live(progress_table, refresh_per_second=10):
        for cls in main_classes:
            data_counter=0
            for i in range(1,database_pages+1):
                if data_counter<satNumber:
                    r=requests.get(url+str(i))
                    soup=BeautifulSoup(r.text,'lxml')
                    class_based_periodic_scrapper(soup=soup,data_folder=data_folder,cls=cls,number=satNumber)
                else:
                    break
                
            overall_progress.update(overall_task,advance=1)

