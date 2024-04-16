import tensorflow as tf
from tensorflow import keras
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn,MofNCompleteColumn,TimeElapsedColumn,TaskProgressColumn
from rich.table import Table
import rich

class RichBarCallBack(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.progress=Progress(
            "{task.description}",
            SpinnerColumn('aesthetic',speed=0.4,style=rich.style.Style(color='yellow')),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn()
            )
        with self.progress:
            self.epoch=self.progress.add_task("Epoch")
    def on_epoch_end(self, epoch, logs=None):
        with Live(self.progress,refresh_per_second=10):
            keys = list(logs.keys())
            print("End epoch {} of training; got log keys: {}".format(epoch, keys))
            self.progress.update(self.epoch,advance=1)
        