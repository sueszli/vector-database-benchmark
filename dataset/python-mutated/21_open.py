def open_file():
    if False:
        while True:
            i = 10
    f = open('photo.jpg', 'r+')
    jpgdata = f.read()
    f.close()

def open_file_right():
    if False:
        while True:
            i = 10
    with open('photo.jpg', 'r+') as f:
        jpgdata = f.read()