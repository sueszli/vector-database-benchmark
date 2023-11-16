from tkinter import *
import tkinter as tk
import tkinter.messagebox as mbox
from pil import ImageTk, Image
import time
import psutil
import socket
window1 = tk.Tk()
window1.title('Network Usage Tracker')
window1.geometry('1000x700')
start1 = tk.Label(text='NETWORK USAGE\nTRACKER', font=('Arial', 55, 'underline'), fg='magenta')
start1.place(x=150, y=10)

def start_fun():
    if False:
        for i in range(10):
            print('nop')
    window1.destroy()
startb = Button(window1, text='START', command=start_fun, font=('Arial', 25), bg='orange', fg='blue', borderwidth=3, relief='raised')
startb.place(x=130, y=590)
path = 'Images/front.png'
img1 = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window1, image=img1)
panel.place(x=320, y=200)

def exit_win():
    if False:
        for i in range(10):
            print('nop')
    if mbox.askokcancel('Exit', 'Do you want to exit?'):
        window1.destroy()
exitb = Button(window1, text='EXIT', command=exit_win, font=('Arial', 25), bg='red', fg='blue', borderwidth=3, relief='raised')
exitb.place(x=730, y=590)
window1.protocol('WM_DELETE_WINDOW', exit_win)
window1.mainloop()
window = Tk()
window.title('Network Usage Tracker')
window.geometry('1000x700')
top1 = Label(window, text='NETWORK USAGE\nTRACKER', font=('Arial', 50, 'underline'), fg='magenta')
top1.place(x=190, y=10)
top1 = Label(window, text='MAX LIMIT  :  1 MB/sec', font=('Arial', 50), fg='green')
top1.place(x=130, y=180)
path_text = Text(window, height=1, width=24, font=('Arial', 50), bg='white', fg='blue', borderwidth=2, relief='solid')
path_text.place(x=50, y=300)
top1 = Label(window, text='Connection Status :', font=('Arial', 50), fg='green')
top1.place(x=200, y=450)
l2 = Label(window, fg='blue', font=('Arial', 30))
l2.place(x=200, y=530)

def convert_to_gbit(value):
    if False:
        print('Hello World!')
    return value / 1024.0 / 1024.0 / 1024.0 * 8
old_value = 0

def update_label():
    if False:
        for i in range(10):
            print('nop')
    global old_value
    new_value = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
    x = '{0:.3f}'.format(new_value - old_value)
    path_text.delete('1.0', 'end')
    path_text.insert(END, 'Usage : ' + str(x) + ' bytes/sec')
    IPaddress = socket.gethostbyname(socket.gethostname())
    if IPaddress == '127.0.0.1':
        l2.configure(text='No internet, your localhost is\n' + IPaddress)
    else:
        l2.configure(text='Connected, with the IP address\n' + IPaddress)
    if new_value - old_value > 1000000:
        mbox.showinfo('Exceed Status', 'Max Limit Usage Exceeded.')
    old_value = new_value
    time.sleep(0.5)
    window.after(1, update_label)
update_label()

def exit_win():
    if False:
        i = 10
        return i + 15
    if mbox.askokcancel('Exit', 'Do you want to exit?'):
        window.destroy()
window.protocol('WM_DELETE_WINDOW', exit_win)
window.mainloop()