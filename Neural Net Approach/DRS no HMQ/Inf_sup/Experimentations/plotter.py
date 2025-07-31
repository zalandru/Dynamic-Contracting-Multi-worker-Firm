# plotter.py
import matplotlib.pyplot as plt
import matplotlib
class LossPlotter:
    def __init__(self, loss_names, pause=0.0001, update_interval=1, ma_window=1,
                 show_raw=True, show_ma=True, background=True):
        """
        loss_names: list of str, keys you’ll use in update()
        pause: float, seconds for plt.pause() to let GUI refresh
        update_interval: int, update/redraw the plot every N iterations
        ma_window: int, window size for moving average (>=1). If 1, MA equals actual.
        show_raw: bool, whether to plot raw loss curves
        show_ma: bool, whether to plot moving average curves
        """
        self.pause = pause
        self.update_interval = max(1, update_interval)
        self.loss_names = loss_names
        self.ma_window = max(1, ma_window)
        self.show_raw = show_raw
        self.show_ma = show_ma
        self.background = background

        plt.ion()
        self.fig, self.ax = plt.subplots()

        if self.background:
            manager = plt.get_current_fig_manager()
            backend = matplotlib.get_backend()
            try:
                if 'Qt' in backend:
                    from PyQt5 import QtCore
                    flags = manager.window.windowFlags()
                    manager.window.setWindowFlags(flags | QtCore.Qt.WindowStaysOnBottomHint)
                    manager.window.show()
                elif backend.startswith('Tk'):
                    manager.window.attributes('-topmost', False)
                else:
                    manager.window.lower()
            except Exception:
                pass


        self.lines = {}
        self.ma_lines = {}
        self.data = {'iter': []}
        self.ma_data = {name: [] for name in loss_names}

        # Initialize line objects based on flags
        for name in loss_names:
            self.data[name] = []
            if self.show_raw:
                line, = self.ax.plot([], [], label=f"{name} (raw)")
                self.lines[name] = line
            if self.show_ma:
                ma_line, = self.ax.plot([], [], linestyle='--',
                                         label=f"{name} (MA{self.ma_window})")
                self.ma_lines[name] = ma_line

        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.fig.tight_layout()

    def update(self, iteration, losses):
        """
        iteration: int, current iteration count
        losses: dict {name: scalar}, keys must match loss_names
        """
        # Record iteration
        self.data['iter'].append(iteration)

        # Append losses and compute MA
        for name, val in losses.items():
            if name not in self.loss_names:
                continue
            arr = self.data[name]
            arr.append(val)

            # update raw line data
            if self.show_raw and name in self.lines:
                self.lines[name].set_data(self.data['iter'], arr)

            # compute and update MA data
            if self.show_ma and name in self.ma_lines:
                window = arr[-self.ma_window:]
                ma_val = sum(window) / len(window)
                self.ma_data[name].append(ma_val)
                self.ma_lines[name].set_data(self.data['iter'], self.ma_data[name])

        # Redraw at interval
        if iteration % self.update_interval == 0:
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.legend()
            self.fig.canvas.draw()
            plt.pause(self.pause)
    
