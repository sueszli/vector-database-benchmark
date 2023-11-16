import os

def modify_search_index(app, exception):
    if False:
        return 10
    if exception is None:
        filename = os.path.join(app.outdir, 'searchindex.js')
        if os.path.exists(filename):
            searchfile = open(filename)
            data1 = searchfile.read()
            searchfile.close()
            p1 = data1.find('filenames:[')
            p2 = data1.find(']', p1)
            s = data1[p1:p2].replace('.txt', '')
            data2 = data1[:p1]
            data2 += s
            data2 += data1[p2:]
            searchfile = open(filename, 'w')
            searchfile.write(data2)
            searchfile.close()

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.connect('build-finished', modify_search_index)