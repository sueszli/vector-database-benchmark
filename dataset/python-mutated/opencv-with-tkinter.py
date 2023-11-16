"""
opencv-with-tkinter.py:
https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/

不需要
pip install image

"""
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2

def select_image():
    if False:
        while True:
            i = 10
    global panelA, panelB
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
        if panelA is None or panelB is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side='left', padx=10, pady=10)
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side='right', padx=10, pady=10)
        else:
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged
root = Tk()
panelA = None
panelB = None
btn = Button(root, text='Select an image', command=select_image)
btn.pack(side='bottom', fill='both', expand='yes', padx='10', pady='10')
root.mainloop()