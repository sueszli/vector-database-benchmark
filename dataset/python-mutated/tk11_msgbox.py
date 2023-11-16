import tkinter as tk
import tkinter.messagebox
window = tk.Tk()
window.title('my window')
window.geometry('200x200')

def hit_me():
    if False:
        i = 10
        return i + 15
    print(tk.messagebox.asktrycancel(title='Hi', message='hahahaha'))
    print(tk.messagebox.askokcancel(title='Hi', message='hahahaha'))
    print(tk.messagebox.askyesnocancel(title='Hi', message='haha'))
tk.Button(window, text='hit me', command=hit_me).pack()
window.mainloop()