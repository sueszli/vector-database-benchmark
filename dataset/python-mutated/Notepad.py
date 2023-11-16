from tkinter import *
from tkinter.messagebox import showinfo
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os

def newFile():
    if False:
        return 10
    global file
    root.title('Untitled - Notepad')
    file = None
    TextArea.delete(1.0, END)

def openFile():
    if False:
        for i in range(10):
            print('nop')
    global file
    file = askopenfilename(defaultextension='.txt', filetypes=[('All Files', '*.*'), ('Text Documents', '*.txt')])
    if file == '':
        file = None
    else:
        root.title(os.path.basename(file) + ' - Notepad')
        TextArea.delete(1.0, END)
        f = open(file, 'r')
        TextArea.insert(1.0, f.read())
        f.close()

def saveFile():
    if False:
        i = 10
        return i + 15
    global file
    if file == None:
        file = asksaveasfilename(initialfile='Untitled.txt', defaultextension='.txt', filetypes=[('All Files', '*.*'), ('Text Documents', '*.txt')])
        if file == '':
            file = None
        else:
            f = open(file, 'w')
            f.write(TextArea.get(1.0, END))
            f.close()
            root.title(os.path.basename(file) + ' - Notepad')
            print('File Saved')
    else:
        f = open(file, 'w')
        f.write(TextArea.get(1.0, END))
        f.close()

def quitApp():
    if False:
        for i in range(10):
            print('nop')
    root.destroy()

def cut():
    if False:
        print('Hello World!')
    TextArea.event_generate('<Cut>')

def copy():
    if False:
        return 10
    TextArea.event_generate('<Copy>')

def paste():
    if False:
        while True:
            i = 10
    TextArea.event_generate('<Paste>')

def about():
    if False:
        for i in range(10):
            print('nop')
    showinfo('Notepad', 'Notepad by Akash Singh')
if __name__ == '__main__':
    root = Tk()
    root.title('Untitled - Notepad')
    root.wm_iconbitmap('note.ico')
    root.geometry('644x688')
    TextArea = Text(root, font='lucida 13')
    file = None
    TextArea.pack(expand=True, fill=BOTH)
    MenuBar = Menu(root)
    FileMenu = Menu(MenuBar, tearoff=0)
    FileMenu.add_command(label='New', command=newFile)
    FileMenu.add_command(label='Open', command=openFile)
    FileMenu.add_command(label='Save', command=saveFile)
    FileMenu.add_separator()
    FileMenu.add_command(label='Exit', command=quitApp)
    MenuBar.add_cascade(label='File', menu=FileMenu)
    EditMenu = Menu(MenuBar, tearoff=0)
    EditMenu.add_command(label='Cut', command=cut)
    EditMenu.add_command(label='Copy', command=copy)
    EditMenu.add_command(label='Paste', command=paste)
    MenuBar.add_cascade(label='Edit', menu=EditMenu)
    HelpMenu = Menu(MenuBar, tearoff=0)
    HelpMenu.add_command(label='About Creater', command=about)
    MenuBar.add_cascade(label='Help', menu=HelpMenu)
    root.config(menu=MenuBar)
    Scroll = Scrollbar(TextArea)
    Scroll.pack(side=RIGHT, fill=Y)
    Scroll.config(command=TextArea.yview)
    TextArea.config(yscrollcommand=Scroll.set)
    root.mainloop()
import tkinter
import os
from tkinter import *
from tkinter.messagebox import *
from tkinter.filedialog import *

class Notepad:
    __root = Tk()
    __thisWidth = 300
    __thisHeight = 300
    __thisTextArea = Text(__root)
    __thisMenuBar = Menu(__root)
    __thisFileMenu = Menu(__thisMenuBar, tearoff=0)
    __thisEditMenu = Menu(__thisMenuBar, tearoff=0)
    __thisHelpMenu = Menu(__thisMenuBar, tearoff=0)
    __thisScrollBar = Scrollbar(__thisTextArea)
    __file = None

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            self.__root.wm_iconbitmap('Notepad.ico')
        except:
            pass
        try:
            self.__thisWidth = kwargs['width']
        except KeyError:
            pass
        try:
            self.__thisHeight = kwargs['height']
        except KeyError:
            pass
        self.__root.title('Untitled - Notepad')
        screenWidth = self.__root.winfo_screenwidth()
        screenHeight = self.__root.winfo_screenheight()
        left = screenWidth / 2 - self.__thisWidth / 2
        top = screenHeight / 2 - self.__thisHeight / 2
        self.__root.geometry('%dx%d+%d+%d' % (self.__thisWidth, self.__thisHeight, left, top))
        self.__root.grid_rowconfigure(0, weight=1)
        self.__root.grid_columnconfigure(0, weight=1)
        self.__thisTextArea.grid(sticky=N + E + S + W)
        self.__thisFileMenu.add_command(label='New', command=self.__newFile)
        self.__thisFileMenu.add_command(label='Open', command=self.__openFile)
        self.__thisFileMenu.add_command(label='Save', command=self.__saveFile)
        self.__thisFileMenu.add_separator()
        self.__thisFileMenu.add_command(label='Exit', command=self.__quitApplication)
        self.__thisMenuBar.add_cascade(label='File', menu=self.__thisFileMenu)
        self.__thisEditMenu.add_command(label='Cut', command=self.__cut)
        self.__thisEditMenu.add_command(label='Copy', command=self.__copy)
        self.__thisEditMenu.add_command(label='Paste', command=self.__paste)
        self.__thisMenuBar.add_cascade(label='Edit', menu=self.__thisEditMenu)
        self.__thisHelpMenu.add_command(label='About Creater', command=self.__showAbout)
        self.__thisMenuBar.add_cascade(label='Help', menu=self.__thisHelpMenu)
        self.__root.config(menu=self.__thisMenuBar)
        self.__thisScrollBar.pack(side=RIGHT, fill=Y)
        self.__thisScrollBar.config(command=self.__thisTextArea.yview)
        self.__thisTextArea.config(yscrollcommand=self.__thisScrollBar.set)

    def __quitApplication(self):
        if False:
            i = 10
            return i + 15
        self.__root.destroy()

    def __showAbout(self):
        if False:
            i = 10
            return i + 15
        showinfo('Notepad', 'Mrinal Verma')

    def __openFile(self):
        if False:
            return 10
        self.__file = askopenfilename(defaultextension='.txt', filetypes=[('All Files', '*.*'), ('Text Documents', '*.txt')])
        if self.__file == '':
            self.__file = None
        else:
            self.__root.title(os.path.basename(self.__file) + ' - Notepad')
            self.__thisTextArea.delete(1.0, END)
            file = open(self.__file, 'r')
            self.__thisTextArea.insert(1.0, file.read())
            file.close()

    def __newFile(self):
        if False:
            return 10
        self.__root.title('Untitled - Notepad')
        self.__file = None
        self.__thisTextArea.delete(1.0, END)

    def __saveFile(self):
        if False:
            return 10
        if self.__file == None:
            self.__file = asksaveasfilename(initialfile='Untitled.txt', defaultextension='.txt', filetypes=[('All Files', '*.*'), ('Text Documents', '*.txt')])
            if self.__file == '':
                self.__file = None
            else:
                file = open(self.__file, 'w')
                file.write(self.__thisTextArea.get(1.0, END))
                file.close()
                self.__root.title(os.path.basename(self.__file) + ' - Notepad')
        else:
            file = open(self.__file, 'w')
            file.write(self.__thisTextArea.get(1.0, END))
            file.close()

    def __cut(self):
        if False:
            while True:
                i = 10
        self.__thisTextArea.event_generate('<<Cut>>')

    def __copy(self):
        if False:
            print('Hello World!')
        self.__thisTextArea.event_generate('<<Copy>>')

    def __paste(self):
        if False:
            i = 10
            return i + 15
        self.__thisTextArea.event_generate('<<Paste>>')

    def run(self):
        if False:
            print('Hello World!')
        self.__root.mainloop()
notepad = Notepad(width=600, height=400)
notepad.run()