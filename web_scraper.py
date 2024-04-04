import requests
from bs4 import BeautifulSoup
import os
import urllib.request
from tqdm import trange,tqdm
from pathlib import Path
import os
from constants import DATASET_FOLDER

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
            # pbar.set_postfix_str(f'Processing satellite_{names[idx]}.txt')
            # print(type(requests.get(link,allow_redirects=True).content))
            f.write(requests.get(link,allow_redirects=True).content)
    
if __name__ =='__main__':

    main_url="http://mmt.favor2.info"
    url='http://mmt.favor2.info/satellites?page='
    data_folder=os.path.join(DATASET_FOLDER,'Periodic')

    classes=['U/SAT','ACT','INACT','R/B','DEB','M/DEB','F/DEB','UNIDENT']
    database_pages=1077
    pbar=trange(1,database_pages+1,colour='red')
    for i in pbar:
        r=requests.get(url+str(i))
        soup=BeautifulSoup(r.text,'lxml')
        pbar.set_description(f'Page-{i}',refresh=True)
        main_class_periodic_scrapper(soup=soup,data_folder=data_folder)