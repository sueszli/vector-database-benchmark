import shutil

def cleanFake():
    if False:
        print('Hello World!')
    try:
        shutil.rmtree('templates/fake')
    except:
        pass