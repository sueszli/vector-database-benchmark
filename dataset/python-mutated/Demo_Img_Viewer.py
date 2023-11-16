import PySimpleGUI as sg
import os
from PIL import Image, ImageTk
import io
'\nSimple Image Browser based on PySimpleGUI\n--------------------------------------------\nThere are some improvements compared to the PNG browser of the repository:\n1. Paging is cyclic, i.e. automatically wraps around if file index is outside\n2. Supports all file types that are valid PIL images\n3. Limits the maximum form size to the physical screen\n4. When selecting an image from the listbox, subsequent paging uses its index\n5. Paging performance improved significantly because of using PIL\n\nDependecies\n------------\nPython3\nPIL\n'
folder = sg.popup_get_folder('Image folder to open', default_path='')
if not folder:
    sg.popup_cancel('Cancelling')
    raise SystemExit()
img_types = ('.png', '.jpg', 'jpeg', '.tiff', '.bmp')
flist0 = os.listdir(folder)
fnames = [f for f in flist0 if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(img_types)]
num_files = len(fnames)
if num_files == 0:
    sg.popup('No files in folder')
    raise SystemExit()
del flist0

def get_img_data(f, maxsize=(1200, 850), first=False):
    if False:
        for i in range(10):
            print('nop')
    'Generate image data using PIL\n    '
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)
filename = os.path.join(folder, fnames[0])
image_elem = sg.Image(data=get_img_data(filename, first=True))
filename_display_elem = sg.Text(filename, size=(80, 3))
file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15, 1))
col = [[filename_display_elem], [image_elem]]
col_files = [[sg.Listbox(values=fnames, change_submits=True, size=(60, 30), key='listbox')], [sg.Button('Next', size=(8, 2)), sg.Button('Prev', size=(8, 2)), file_num_display_elem]]
layout = [[sg.Column(col_files), sg.Column(col)]]
window = sg.Window('Image Browser', layout, return_keyboard_events=True, location=(0, 0), use_default_focus=False)
i = 0
while True:
    (event, values) = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED:
        break
    elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34'):
        i += 1
        if i >= num_files:
            i -= num_files
        filename = os.path.join(folder, fnames[i])
    elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
        i -= 1
        if i < 0:
            i = num_files + i
        filename = os.path.join(folder, fnames[i])
    elif event == 'listbox':
        f = values['listbox'][0]
        filename = os.path.join(folder, f)
        i = fnames.index(f)
    else:
        filename = os.path.join(folder, fnames[i])
    image_elem.update(data=get_img_data(filename, first=True))
    filename_display_elem.update(filename)
    file_num_display_elem.update('File {} of {}'.format(i + 1, num_files))
window.close()