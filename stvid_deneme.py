
import pandas as pd
import matplotlib.pyplot as plt

with open("stvid_files/2022-07-18T21_30_05.202_stars.cat",'r+') as f:
    # print(allfile)
    for i in range(6):next(f)
    data=[[float(j) for j in " ".join(x.split()).split(" ")] for x in f.readlines()]
    print(data)
    column=['X_IMAGE','Y_IMAGE','MAG_BEST','FLUX_BEST','FLUX_MAX','BACKGROUND']
    df=pd.DataFrame(data,columns=column)
    values=df.get('MAG_BEST')
    plt.plot(values)
    plt.show()
