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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn,ProgressColumn
from rich.table import Table
from rich.console import Console
from rich.text import Text
from scipy import interpolate
from scipy import stats
import tensorflow as tf
from typing import Union, List, Tuple, Dict
import warnings

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


def DiscreteWaveletTransform1(trackSeries):

    # tempSeries=list()
    transformedSeries = []
    for track in trackSeries:
        w=pywt.Wavelet('db4')
        cA,cD= pywt.dwt(track,w,'constant')
        combined = np.concatenate([cA, cD])  # Flattened vector
        transformedSeries.append(combined)
    return transformedSeries  

def DiscreteWaveletTransform(trackSeries, wavelet='db4', level=3):

    for idx, track in enumerate(trackSeries):
        # # Ensure minimum length for decomposition
        # if len(track) < 2**level:
        #     pad_length = 2**level - len(track)
        #     track = np.pad(track, (0, pad_length), mode='symmetric')
    
        # Perform multi-level wavelet decomposition
        coeffs = pywt.wavedec(track, wavelet)
        
        # Concatenate all coefficients (both approximation and details)
        features = np.concatenate(coeffs)
        
        # Update the track with concatenated coefficients
        trackSeries[idx] = features
        
    return trackSeries


def discrete_wavelet_transform(
    track_series: Union[np.ndarray, List[np.ndarray]], 
    wavelet: str = 'db4', 
    level: int = None,
    pad_mode: str = 'symmetric'
) -> np.ndarray:
    """
    Apply multi-level discrete wavelet transform to time series data with robust handling
    
    Parameters:
    -----------
    track_series : numpy array or list of numpy arrays
        Input time series tracks to transform
    wavelet : str, optional (default='db4')
        Wavelet basis to use
    level : int, optional (default=None)
        Number of decomposition levels
    pad_mode : str, optional (default='symmetric')
        Padding mode for handling edge effects
    
    Returns:
    --------
    transformed_series : numpy array
        Transformed time series with uniform shape
    """
    # Convert input to list if it's a numpy array
    if isinstance(track_series, np.ndarray):
        track_series = list(track_series)
    
    # Determine optimal decomposition level if not specified
    if level is None:
        level = int(np.floor(np.log2(min(len(track) for track in track_series))))
    
    def transform_single_track(track):
        # Ensure minimum length for decomposition
        min_length = 2**level
        if len(track) < min_length:
            pad_length = min_length - len(track)
            track = np.pad(track, (0, pad_length), mode=pad_mode)
        
        # Perform multi-level wavelet decomposition
        coeffs = pywt.wavedec(track, wavelet, level=level)
        
        # Flatten and concatenate coefficients
        return np.concatenate(coeffs)
    
    # Transform each track
    transformed_tracks = [transform_single_track(track) for track in track_series]
    
    # Find the minimum length among transformed tracks
    min_length = min(len(track) for track in transformed_tracks)
    
    # Truncate or pad tracks to ensure uniform shape
    uniform_transformed_tracks = [
        track[:min_length] if len(track) > min_length 
        else np.pad(track, (0, min_length - len(track)), mode='constant')
        for track in transformed_tracks
    ]
    
    return np.array(uniform_transformed_tracks)





def pad_to_size_interpolate(array, target_size):
    """
    Pad or sample a 1D array to a target size using cubic interpolation.

    Parameters:
    -----------
    array : numpy.ndarray
        1D input array
    target_size : int
        Desired length of the output array
        
    Returns:
    --------
    numpy.ndarray
        Array padded or sampled to target_size using interpolation
    """
    array = np.asarray(array)
    if array.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")

    current_size = len(array)

    # If array is already the target size, return it
    if current_size == target_size:
        return array.copy()

    # Create new x coordinates for interpolation
    old_x = np.linspace(0, 1, current_size)
    new_x = np.linspace(0, 1, target_size)

    # Use cubic interpolation for smooth results
    f = interpolate.interp1d(old_x, array, kind='cubic', fill_value='extrapolate')

    # Sample the interpolated function at the new x coordinates
    return f(new_x)

class DelayedExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, start_epoch,log_dir, **kwargs):
        super().__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.start_epoch = start_epoch
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True
        )
        self.log_dir = log_dir+"/train"
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            # Adjust the learning rate using the schedule
            lr = self.lr_schedule(epoch - self.start_epoch)
            self.model.optimizer.learning_rate.assign(lr)

            
            
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

    def __repr__(self):
        return (f"Name:{self.name}\n"
                f"type ID:{self.type}\n"
                f"Track number :{len(self.lightCurves)}")

class MetricsColumn(ProgressColumn):
    def __init__(self, getter_func):
        super().__init__()
        self.getter_func = getter_func

    def render(self, task):
        # Call the getter function to retrieve the metric value
        value = self.getter_func(task.fields)
        
        # Assign colors based on metric type
        if 'Loss' in value:
            return Text(value, style="red")  # Loss in red
        elif 'Acc' in value:
            accuracy = float(value.split(": ")[1].strip('%'))  # Extract accuracy value
            if accuracy >= 90:
                color="bold green"
            elif accuracy >=70:
                color="bold blue"
            else:
                color="bold yellow"
            # color = "bold blue" if accuracy >= 70 else "bold yellow"  # High accuracy is green, lower is yellow
            return Text(value, style=color)
        else:
            return Text(value)
    
def trainingProgressBar(epoch,batch):
    console=Console()
    progressBar = Progress(
    "[bold]Epoch: {task.fields[epoch]}",
    "[green]Batch:",
    MofNCompleteColumn(),
    BarColumn(),
    MetricsColumn(lambda fields: f"Loss: {fields['train_loss']:.4f}"),
    MetricsColumn(lambda fields: f"Train Acc: {fields['train_acc']:.2f}%"),
    MetricsColumn(lambda fields: f"Val Loss: {fields['val_loss']:.4f}"),
    MetricsColumn(lambda fields: f"Val Acc: {fields['val_acc']:.2f}%"),
        "| {task.percentage:>3.1f}% ",
    TimeElapsedColumn(),console=console)
    task=progressBar.add_task("Training",epoch=epoch,total=batch,train_loss=0,train_acc=0,val_loss=0,val_acc=0)
    
    return progressBar



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

