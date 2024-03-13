import os
import matplotlib.pyplot as plt
from data_parser import data_splitter,satellite_info
from constants import DATASET_FOLDER
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
    # plt.savefig(f'rapor_plots/{satellite_name} {class_name}.png',bbox_inches='tight', dpi=200)   # save the figure to file
    plt.show()

if __name__ =='__main__':
    folder=DATASET_FOLDER
    plot_save_folder='rapor_plots'
    classNames=sorted(os.listdir(folder))
    act_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[0]))
    fdeb_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[1]))
    inact_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[2]))
    mdeb_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[3]))
    rb_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[4]))
    usat_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[5]))
    unident_sat=os.listdir(os.path.join(DATASET_FOLDER,classNames[6]))
    satellites=[act_sat,fdeb_sat,inact_sat,mdeb_sat,rb_sat,usat_sat,unident_sat]
    for idx,i in enumerate(satellites):
        for j in i:
            all_sat_plotter(os.path.join(DATASET_FOLDER,f"{classNames[idx]}/{j}"))
                
                        
    
