from random import random
from panda3d.core import Point3
from direct.task.TaskManagerGlobal import taskMgr
from direct.gui import DirectGuiGlobals
from direct.gui.DirectGui import DirectButton, DirectDialog, DirectEntry, DirectFrame, YesNoDialog

def test_DirectGui(base):
    if False:
        for i in range(10):
            print('nop')
    smiley = base.loader.loadModel('models/misc/smiley')

    def dummyCmd(index):
        if False:
            print('Hello World!')
        print('Button %d POW!!!!' % index)

    def shrink(db):
        if False:
            return 10
        db['text2_text'] = 'Hi!'
        taskMgr.remove('shrink')
        taskMgr.remove('expand')
        rolloverSmiley = db.component('geom2')
        rolloverSmiley.setScale(db.component('geom0').getScale()[0])
        rolloverSmiley.lerpScale(0.1, 0.1, 0.1, 1.0, blendType='easeInOut', task='shrink')

    def expand(db):
        if False:
            print('Hello World!')
        db['text0_text'] = 'Bye!'
        taskMgr.remove('shrink')
        taskMgr.remove('expand')
        db.component('geom0').setScale(db.component('geom2').getScale()[0])
        db.component('geom0').lerpScale(1, 1, 1, 1, blendType='easeInOut', task='expand')
        db.component('geom2').clearColor()

    def ouch(db):
        if False:
            print('Hello World!')
        taskMgr.remove('shrink')
        taskMgr.remove('expand')
        taskMgr.remove('runAway')
        db.component('geom0').setScale(db.component('geom2').getScale()[0])
        db.component('geom1').setScale(db.component('geom2').getScale()[0])
        db['text2_text'] = 'Ouch!'
        db['geom2_color'] = (1, 0, 0, 1)
        newX = -1.0 + random() * 2.0
        newZ = -1.0 + random() * 2.0
        db.lerpPos(Point3(newX, 0, newZ), 1.0, task='runAway', blendType='easeOut')
    dl = DirectFrame(image='models/maps/noise.rgb')
    dl.setScale(0.5)
    dbArray = []
    for i in range(10):
        db = DirectButton(parent=dl, image='models/maps/noise.rgb', geom=smiley, text=('Hi!', 'Ouch!', 'Bye!', 'ZZZZ!'), scale=0.15, relief='raised', geom1_color=(1, 0, 0, 1), text_pos=(0.6, -0.8), clickSound=DirectGuiGlobals.getDefaultClickSound(), rolloverSound=DirectGuiGlobals.getDefaultRolloverSound())
        db['text_scale'] = 0.5
        db['command'] = lambda i=i: dummyCmd(i)
        db.bind(DirectGuiGlobals.ENTER, lambda x, db=db: shrink(db))
        db.bind(DirectGuiGlobals.EXIT, lambda x, db=db: expand(db))
        db.bind(DirectGuiGlobals.B1PRESS, lambda x, db=db: ouch(db))
        db.bind(DirectGuiGlobals.B3PRESS, lambda x, db=db: db.place())
        dbArray.append(db)

    def printEntryText(text):
        if False:
            i = 10
            return i + 15
        print('Text: %s' % text)
    de1 = DirectEntry(initialText='Hello, how are you?', image='models/maps/noise.rgb', image_pos=(4.55, 0, -2.55), image_scale=(5.5, 1, 4), command=printEntryText, pos=(-1.1875, 0, 0.879167), scale=0.0707855, cursorKeys=1)

    def printDialogValue(value):
        if False:
            for i in range(10):
                print('nop')
        print('Value: %s' % value)
    simpleDialog = YesNoDialog(text='Simple', command=printDialogValue)
    customValues = YesNoDialog(text='Not Quite So Simple', buttonValueList=['Yes', 'No'], command=printDialogValue)
    fancyDialog = YesNoDialog(text='Testing Direct Dialog', geom=smiley, geom_scale=0.1, geom_pos=(-0.3, 0, 0), command=printDialogValue)
    customDialog = DirectDialog(text='Pick a number', buttonTextList=[str(i) for i in range(10)], buttonValueList=range(10), command=printDialogValue)
    print('BOUNDS: %s' % de1.getBounds())
    print('WIDTH: %s' % de1.getWidth())
    print('HEIGHT: %s' % de1.getHeight())
    print('CENTER: %s' % (de1.getCenter(),))