import tkinter as tk
window = tk.Tk()
window.title('my window')
window.geometry('200x200')
e = tk.Entry(window, show='1')
e.pack()

def insert_point():
    if False:
        print('Hello World!')
    var = e.get()
    t.insert('insert', var)

def insert_end():
    if False:
        while True:
            i = 10
    var = e.get()
    t.insert(2.2, var)
b1 = tk.Button(window, text='insert point', width=15, height=2, command=insert_point)
b1.pack()
b2 = tk.Button(window, text='insert end', command=insert_end)
b2.pack()
t = tk.Text(window, height=2)
t.pack()
window.mainloop()