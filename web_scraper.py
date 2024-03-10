import requests
from bs4 import BeautifulSoup
import os
import urllib.request
from tqdm import trange,tqdm
from pathlib import Path
import os
from constants import DATASET_FOLDER
def data_size():
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
        with open(f'{data_folder}/satellite-{names[idx]}_{object_classes[idx]}.txt','wb') as f:
            data_size()
            pbar.set_description(f'Processing << satellite-{names[idx]}_{object_classes[idx]} >> Total Size > {data_size()/1024**2:.2f} MB',refresh=True)
            # pbar.set_postfix_str(f'Processing satellite_{names[idx]}.txt')
            f.write(requests.get(link,allow_redirects=True).content)
          
            
if __name__ =='__main__':

    main_url="http://mmt.favor2.info"
    url='http://mmt.favor2.info/satellites?page='
    data_folder=DATASET_FOLDER

    classes=['U/SAT','ACT','INACT','R/B','DEB','M/DEB','F/DEB','UNIDENT']
    database_pages=1077
    pbar=trange(1,database_pages+1,colour='red')
    for i in pbar:
        r=requests.get(url+str(i))
        soup=BeautifulSoup(r.text,'lxml')
        pbar.set_description(f'Page-{i}',refresh=True)
        get_data_from_page(soup=soup,data_folder=data_folder)