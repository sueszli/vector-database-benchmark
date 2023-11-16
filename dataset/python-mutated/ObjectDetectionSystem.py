import tkinter as tk
from tkinter import font as tkfont
from tkinter import Frame
from typing import List, Tuple
from ObjectDetector import ObjectDetector
from tkinter import filedialog
from Model import Model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Arial Nova', size=18, weight='bold')
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky='nsew')
        self.show_frame('StartPage')

    def show_frame(self, page_name):
        if False:
            for i in range(10):
                print('nop')
        'Show a frame for the given page name'
        frame = self.frames[page_name]
        frame.tkraise()

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        if False:
            return 10
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text='Wellcome to object detection system', font=controller.title_font)
        label.pack(side='top', fill='x', pady=10)
        f = Frame(self, height=3, width=1000, bg='white')
        f.pack()
        button1 = tk.Button(self, text='Camera Detection', command=lambda : self.openPageCamDetection(), width=60, padx=20, pady=10)
        button2 = tk.Button(self, text='Image Detection', command=lambda : controller.show_frame('PageTwo'), width=60, padx=20, pady=10)
        button1.pack()
        button2.pack()
        self.model = Model.getInstance()
        f = Frame(self, height=3, width=1000, bg='white')
        f.pack()
        v = tk.IntVar()
        v.set(1)
        trainedModels: List[Tuple[str, int]] = [('faster_rcnn_inception_v2_coco_2018_01_28', 1), ('ssd_inception_v2_coco_2018_01_28', 2), ('ssd_mobilenet_v2_coco_2018_03_29', 3), ('mask_rcnn_inception_v2_coco_2018_01_28', 4), ('intis_Model', 5)]

        def showChoice():
            if False:
                return 10
            self.model.set_name(trainedModels[v.get()][0])
            if v.get() == 4:
                self.model.set_bool_custom_trained(True)
            else:
                self.model.set_bool_custom_trained(False)
        f.pack()
        tk.Label(self, text='Choose model that will be used\n        for detection:', justify=tk.LEFT, padx=20, pady=5).pack()
        for (index, name) in enumerate(trainedModels):
            tk.Radiobutton(self, text=name, indicatoron=0, width=40, padx=30, pady=10, variable=v, command=showChoice, value=index).pack(anchor=tk.CENTER)
        f.pack()

    def openPageCamDetection(self):
        if False:
            return 10
        self.controller.show_frame('PageOne')
        PageOne(self, self.controller).startDetecting()

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        if False:
            for i in range(10):
                print('nop')
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text='Camera Detection', font=controller.title_font)
        label.pack(side='top', fill='x', padx=30, pady=20)
        f = Frame(self, height=3, width=1000, bg='white')
        f.pack()
        button = tk.Button(self, text='Return to main menu', command=lambda : controller.show_frame('StartPage'), padx=20, pady=10)
        button.pack()
        f = Frame(self, height=3, width=1000, bg='white')
        f.pack()
        self.od = ObjectDetector()

    def startDetecting(self):
        if False:
            i = 10
            return i + 15
        self.od.detectOcjectsFromCamera()

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        if False:
            for i in range(10):
                print('nop')
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text='Image Detection', font=controller.title_font)
        label.pack(side='top', fill='x', padx=30, pady=20)
        f = Frame(self, height=3, width=1000, bg='white')
        f.pack()
        button = tk.Button(self, text='Return to main menu', command=lambda : controller.show_frame('StartPage'), padx=20, pady=10)
        button.pack()
        button2 = tk.Button(self, text='New image', command=lambda : self.detectObjectsImage(), padx=20, pady=10)
        button2.pack()
        f = Frame(self, height=3, width=1000, bg='white')
        f.pack()
        self.od = ObjectDetector()
        self.od.detectOcjectsFromImagesSetup()

    def detectObjectsImage(self):
        if False:
            for i in range(10):
                print('nop')
        self.od = ObjectDetector()
        self.od.detectOcjectsFromImagesSetup()
        tk.Tk().withdraw()
        self.file_path = filedialog.askopenfilename()
        if self.file_path == '':
            return
        image = Image.open(self.file_path)
        image_np = self.od.load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = self.od.run_inference_for_single_image(image_np_expanded, self.od.detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(image_np, output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'], self.od.category_index, instance_masks=output_dict.get('detection_masks'), use_normalized_coordinates=True, line_thickness=8)
        plt.cla()
        plt.clf()
        plt.close(fig='all')
        plt.close(plt.gcf())
        plt.figure(clear=True)
        fig = plt.figure(figsize=self.od.IMAGE_SIZE)
        im = plt.imshow(image_np)
        plt.minorticks_off()
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        try:
            self.canvas.get_tk_widget().pack_forget()
        except AttributeError:
            pass
        f = Figure(figsize=(5, 1))
        aplt = f.add_subplot(111)
        aplt.plot([1, 2, 3, 4])
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
if __name__ == '__main__':
    app = SampleApp()
    app.mainloop()