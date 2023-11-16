import customtkinter
import tkinter.ttk as ttk
app = customtkinter.CTk()
app.geometry('400x600')
p1 = customtkinter.CTkProgressBar(app)
p1.pack(pady=20)
p2 = ttk.Progressbar(app)
p2.pack(pady=20)
s1 = customtkinter.CTkSlider(app, command=p1.set)
s1.pack(pady=20)

def switch_func():
    if False:
        print('Hello World!')
    if sw1.get() == 1:
        p1.configure(mode='indeterminate')
        p2.configure(mode='indeterminate')
    else:
        p1.configure(mode='determinate')
        p2.configure(mode='determinate')

def start():
    if False:
        for i in range(10):
            print('nop')
    p1.start()
    p2.start()

def stop():
    if False:
        for i in range(10):
            print('nop')
    p1.stop()
    p2.stop()

def step():
    if False:
        print('Hello World!')
    p1.step()
    p2.step(10)
sw1 = customtkinter.CTkSwitch(app, text='intermediate mode', command=switch_func)
sw1.pack(pady=20)
b1 = customtkinter.CTkButton(app, text='start', command=start)
b1.pack(pady=20)
b2 = customtkinter.CTkButton(app, text='stop', command=stop)
b2.pack(pady=20)
b3 = customtkinter.CTkButton(app, text='step', command=step)
b3.pack(pady=20)
app.mainloop()