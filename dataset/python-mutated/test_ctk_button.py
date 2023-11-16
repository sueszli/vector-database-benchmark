import time
import customtkinter

class TestCTkButton:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.root_ctk = customtkinter.CTk()
        self.ctk_button = customtkinter.CTkButton(self.root_ctk)
        self.ctk_button.pack(padx=20, pady=20)
        self.root_ctk.title(self.__class__.__name__)

    def clean(self):
        if False:
            while True:
                i = 10
        self.root_ctk.quit()
        self.root_ctk.withdraw()

    def main(self):
        if False:
            while True:
                i = 10
        self.execute_tests()
        self.root_ctk.mainloop()

    def execute_tests(self):
        if False:
            print('Hello World!')
        print(f'\n{self.__class__.__name__} started:')
        start_time = 0
        self.root_ctk.after(start_time, self.test_iconify)
        start_time += 1500
        self.root_ctk.after(start_time, self.clean)

    def test_iconify(self):
        if False:
            print('Hello World!')
        print(' -> test_iconify: ', end='')
        self.root_ctk.iconify()
        self.root_ctk.after(100, self.root_ctk.deiconify)
        print('successful')
if __name__ == '__main__':
    TestCTkButton().main()