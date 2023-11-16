"""
===============
Multiprocessing
===============

Demo of using multiprocessing for generating data in one process and
plotting in another.

Written by Robert Cimrman
"""
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(19680801)

class ProcessPlotter:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.x = []
        self.y = []

    def terminate(self):
        if False:
            print('Hello World!')
        plt.close('all')

    def call_back(self):
        if False:
            while True:
                i = 10
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.x.append(command[0])
                self.y.append(command[1])
                self.ax.plot(self.x, self.y, 'ro')
        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        if False:
            for i in range(10):
                print('nop')
        print('starting plotter...')
        self.pipe = pipe
        (self.fig, self.ax) = plt.subplots()
        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()
        print('...done')
        plt.show()

class NBPlot:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        (self.plot_pipe, plotter_pipe) = mp.Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, finished=False):
        if False:
            for i in range(10):
                print('nop')
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            data = np.random.random(2)
            send(data)

def main():
    if False:
        for i in range(10):
            print('nop')
    pl = NBPlot()
    for _ in range(10):
        pl.plot()
        time.sleep(0.5)
    pl.plot(finished=True)
if __name__ == '__main__':
    if plt.get_backend() == 'MacOSX':
        mp.set_start_method('forkserver')
    main()