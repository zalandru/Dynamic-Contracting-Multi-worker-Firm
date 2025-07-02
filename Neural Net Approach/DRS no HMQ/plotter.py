# plotter.py
import matplotlib.pyplot as plt

class LossPlotter:
    def __init__(self, loss_names, pause=0.001, update_interval=5):
        """
        loss_names: list of str, keys you’ll use in update()
        pause: float, seconds for plt.pause() to let GUI refresh
        """
        self.pause = pause
        self.loss_names = loss_names
        # ensure update_interval is at least 1
        self.update_interval = max(1, update_interval)
        plt.ion()  # interactive mode on
        self.fig, self.ax = plt.subplots()
        self.lines = {}
        self.data = {'iter': []}

        # initialize one line & data buffer per loss
        for name in loss_names:
            self.data[name] = []
            line, = self.ax.plot([], [], label=name.replace('_', ' ').title())
            self.lines[name] = line

        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.fig.tight_layout()

    def update(self, iteration, losses):
        """
        iteration: int, current episode/step count
        losses: dict {name: scalar}, keys must match loss_names
        """
        self.data['iter'].append(iteration)

        # append new losses and update the corresponding Line2D
        for name, val in losses.items():
            if name not in self.lines:
                continue
            self.data[name].append(val)
            self.lines[name].set_data(self.data['iter'], self.data[name])
        # redraw only at the specified interval
        if iteration % self.update_interval == 0:
            # rescale & redraw
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            plt.pause(self.pause)