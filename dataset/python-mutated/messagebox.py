import time
try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

def retryskipabort(message, timeout=20):
    if False:
        for i in range(10):
            print('nop')
    '\n    Show dialog of RETRY,SKIP,ABORT\n    Returns:\n        one of "retry", "skip", "abort"\n    '
    root = tk.Tk()
    root.geometry('400x200')
    root.title('Exception handle')
    root.eval('tk::PlaceWindow %s center' % root.winfo_pathname(root.winfo_id()))
    root.attributes('-topmost', True)
    _kvs = {'result': 'abort'}

    def cancel_timer(*args):
        if False:
            return 10
        root.after_cancel(_kvs['root'])
        root.title('Manual')

    def update_prompt():
        if False:
            i = 10
            return i + 15
        cancel_timer()

    def f(result):
        if False:
            i = 10
            return i + 15

        def _inner():
            if False:
                print('Hello World!')
            _kvs['result'] = result
            root.destroy()
        return _inner
    tk.Label(root, text=message).pack(side=tk.TOP, fill=tk.X, pady=10)
    frmbtns = tk.Frame(root)
    tk.Button(frmbtns, text='Skip', command=f('skip')).pack(side=tk.LEFT)
    tk.Button(frmbtns, text='Retry', command=f('retry')).pack(side=tk.LEFT)
    tk.Button(frmbtns, text='ABORT', command=f('abort')).pack(side=tk.LEFT)
    frmbtns.pack(side=tk.BOTTOM)
    prompt = tk.StringVar()
    label1 = tk.Label(root, textvariable=prompt)
    label1.pack()
    deadline = time.time() + timeout

    def _refresh_timer():
        if False:
            print('Hello World!')
        leftseconds = deadline - time.time()
        if leftseconds <= 0:
            root.destroy()
            return
        root.title('Test will stop after ' + str(int(leftseconds)) + ' s')
        _kvs['root'] = root.after(500, _refresh_timer)
    _kvs['root'] = root.after(0, _refresh_timer)
    root.bind('<Button-1>', cancel_timer)
    root.mainloop()
    return _kvs['result']
if __name__ == '__main__':
    print(retryskipabort('LKJSDF\nlkjj\\what?lkjsdlfjaskdfjlasdkjflnice'))