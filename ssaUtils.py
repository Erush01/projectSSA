# import tensorflow as tf
from PIL import ImageFont
from collections import defaultdict
from rich.panel import Panel
from rich.table import Table
import pywt
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
from rich.progress import Progress
import rich
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn
from rich.table import Table
from rich.console import Console

def get_summary_str(model):
    lines = []
    model.summary(print_fn=lines.append)
    # Add initial spaces to avoid markdown formatting in TensorBoard
    return '    ' + '\n    '.join(lines)

def train_table(x_train,y_train,x_val,y_val,x_test,y_test):
    train_table=Table.grid()
    train_table.add_row(
                Panel.fit(
                    f"[bold green]X: [white]{x_train.shape}\n"
                    f"[bold green]Y: [white]{y_train.shape}\n",
                    title="[b]Train", border_style="green", padding=(1, 1)),
                Panel.fit(
                    f"[bold yellow]X : [white]{x_val.shape}\n"
                    f"[bold yellow]Y : [white]{y_val.shape}\n",
                    title="[b]Validation", border_style="yellow", padding=(1,1)),
            Panel.fit(f"[bold cyan]X : [white]{x_test.shape}\n"
                    f"[bold cyan] Y : [white]{y_test.shape}\n",
                    title="[b]Test",border_style="cyan", padding=(1, 1)))
    return train_table


def DiscreteWaveletTransform(trackSeries):

    # tempSeries=list()
    for idx,track in enumerate(trackSeries):
        w=pywt.Wavelet('haar')
        cA,cD= pywt.dwt(track,w,'constant')
        
        trackSeries[idx]=(cA)
    # return tempSeries  

 

# class DelayedExponentialDecay(tf.keras.callbacks.Callback):
#     def __init__(self, initial_learning_rate, decay_steps, decay_rate, start_epoch,log_dir, **kwargs):
#         super().__init__(**kwargs)
#         self.initial_learning_rate = initial_learning_rate
#         self.decay_steps = decay_steps
#         self.decay_rate = decay_rate
#         self.start_epoch = start_epoch
#         self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             initial_learning_rate=self.initial_learning_rate,
#             decay_steps=self.decay_steps,
#             decay_rate=self.decay_rate,
#             staircase=True
#         )
#         self.log_dir = log_dir+"/train"
#         self.writer = tf.summary.create_file_writer(self.log_dir)

#     def on_epoch_begin(self, epoch, logs=None):
#         if epoch >= self.start_epoch:
#             # Adjust the learning rate using the schedule
#             lr = self.lr_schedule(epoch - self.start_epoch)
#             tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         lr = tf.keras.backend.get_value(self.model.optimizer.lr)
#         logs.update({'lr': lr})
#         with self.writer.as_default():
#             # Log the learning rate for TensorBoard
#             tf.summary.scalar('learning_rate', lr, step=epoch)
#             self.writer.flush()
            
            
def save_evaluated_lc_plots(data,labels,predictions,buffer):
    total_plots = 20
    start_index = random.randint(0,data.shape[0]-total_plots)
    end_index = start_index + total_plots
    data=data[start_index:end_index]
    labels=labels[start_index:end_index]
    predictions=predictions[start_index:end_index]
    cols = total_plots // 4
    rows = total_plots // cols
    if total_plots % cols != 0:
        rows += 1
    pos = range(1, total_plots + 1)
    fig = plt.figure(figsize=(20, 10))
    for i, (plot_data, og_label, pred_label) in enumerate(
        zip(data, labels, predictions)
    ):
        plt.subplot(rows, cols, pos[i])
        plt.plot(plot_data)
        plt.title(f"Actual Label : {og_label}\nPredicted Label : {pred_label}",fontdict={'fontsize':10})
        fig.subplots_adjust(hspace=0.5)
    plt.savefig(buffer,format='png')

class downloadProgressBar():
    def __init__(self,satNumber):
        self.job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn()
        )
        self.process=Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            BarColumn(),
            TimeElapsedColumn()
        )
        self.console=Console()
        self.job_progress.add_task("[bold green]SATELLITE",total=satNumber[0])
        self.job_progress.add_task("[bold yellow]ROCKETBODY", total=satNumber[1])
        self.job_progress.add_task("[bold cyan]DEBRIS",start=True, total=satNumber[2])
        
        self.processingBar=self.process.add_task("[bold yellow]Processing",start=True,total=None)
        
        self.total = sum(task.total for task in self.job_progress.tasks)
        self.overall_progress = Progress(TextColumn("[progress.description]{task.description}"),
                                            BarColumn(),
                                            TaskProgressColumn(),
                                            MofNCompleteColumn(),
                                            TimeElapsedColumn())
        self.overall_task = self.overall_progress.add_task("[b]All RSO's", total=int(self.total))
        self.progress_table = Table.grid()
        self.progress_table.add_row(
            Panel.fit(
                self.overall_progress, title="Overall Progress", border_style="green", padding=(2, 2)
            ),
            Panel.fit(self.job_progress, title="[b]Resident Space Objects", border_style="red", padding=(1, 2)),
            Panel.fit(self.process,title="Processing",border_style='yellow',padding=(2, 2))
        )

    def updateOverall(self):
        self.overall_progress.update(self.overall_task,advance=1)

class LightCurveReadProgressBar():
    def __init__(self,satNumber):
        self.job_progress = Progress(
            "{task.description}",
            SpinnerColumn('aesthetic',speed=0.4,style=rich.style.Style(color='yellow')),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn()
            )
        self.job_progress.add_task("[bold green]SATELLITE",total=satNumber[0])
        self.job_progress.add_task("[bold yellow]ROCKETBODY", total=satNumber[1])
        self.job_progress.add_task("[bold cyan]DEBRIS", total=satNumber[2])
        # job4 = job_progress.add_task("[bold magenta]NONROTARY", total=satNumber[3])

        total = sum(task.total for task in self.job_progress.tasks)
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
            Panel.fit(self.job_progress, title="[b]Resident Space Objects", border_style="red", padding=(1, 2)),
        )


class RSO:
    def __init__(self,name,type,lcArray):
        self.name=name
        self.type=type
        self.lightCurves=lcArray




class LightCurve:
    def __init__(self,rso,trackID,track=[]):
        self.rso=rso
        self.trackID=trackID
        self.track=track
        self.nPoints=len(track)

    def plot(self):
        plt.plot(self.track,marker='o',linewidth=1.0,markersize=3)
        plt.title(f"{self.rso}-{self.trackID}")
        plt.xlabel("Sample #")
        plt.ylabel("Magnitude")
        plt.xticks(rotation=45)
        plt.tight_layout()    
        plt.gcf().set_size_inches(5, 5)
        plt.show()
        plt.close()

    def __repr__(self):
        return (f"Resident Space Object:{self.rso}\n"
                f"Track ID:{self.trackID}\n"
                f"Number of points:{self.nPoints}")

