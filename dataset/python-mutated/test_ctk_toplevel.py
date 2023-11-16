import customtkinter
customtkinter.set_appearance_mode('dark')

class ToplevelWindow(customtkinter.CTkToplevel):

    def __init__(self, *args, closing_event=None, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.protocol('WM_DELETE_WINDOW', self.closing)
        self.geometry('500x300')
        self.resizable(False, False)
        self.closing_event = closing_event
        self.label = customtkinter.CTkLabel(self, text='ToplevelWindow')
        self.label.pack(padx=20, pady=20)
        self.button_1 = customtkinter.CTkButton(self, text='set dark', command=lambda : customtkinter.set_appearance_mode('dark'))
        self.button_1.pack(side='top', padx=40, pady=40)

    def closing(self):
        if False:
            for i in range(10):
                print('nop')
        self.destroy()
        if self.closing_event is not None:
            self.closing_event()

class App(customtkinter.CTk):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.geometry('500x400')
        self.resizable(False, False)
        self.button_1 = customtkinter.CTkButton(self, text='Open CTkToplevel', command=self.open_toplevel)
        self.button_1.pack(side='top', padx=40, pady=40)
        self.button_2 = customtkinter.CTkButton(self, text='iconify toplevel', command=lambda : self.toplevel_window.iconify())
        self.button_2.pack(side='top', padx=40, pady=40)
        self.button_3 = customtkinter.CTkButton(self, text='set light', command=lambda : customtkinter.set_appearance_mode('light'))
        self.button_3.pack(side='top', padx=40, pady=40)
        self.toplevel_window = None

    def open_toplevel(self):
        if False:
            return 10
        if self.toplevel_window is None:
            self.toplevel_window = ToplevelWindow(self, closing_event=self.toplevel_close_event)

    def toplevel_close_event(self):
        if False:
            print('Hello World!')
        self.toplevel_window = None
if __name__ == '__main__':
    app = App()
    app.mainloop()