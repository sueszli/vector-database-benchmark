from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
from threading import Thread
import time

class LiveGraph:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        (self.x_data, self.y_data) = ([], [])
        self.figure = plt.figure()
        (self.line,) = plt.plot(self.x_data, self.y_data)
        self.animation = FuncAnimation(self.figure, self.update, interval=1000)
        self.th = Thread(target=self.thread_f, name='LiveGraph', daemon=True)
        self.th.start()

    def update(self, frame):
        if False:
            for i in range(10):
                print('nop')
        self.line.set_data(self.x_data, self.y_data)
        self.figure.gca().relim()
        self.figure.gca().autoscale_view()
        return (self.line,)

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        plt.show()

    def thread_f(self):
        if False:
            while True:
                i = 10
        x = 0
        while True:
            self.x_data.append(x)
            x += 1
            self.y_data.append(randrange(0, 100))
            time.sleep(1)