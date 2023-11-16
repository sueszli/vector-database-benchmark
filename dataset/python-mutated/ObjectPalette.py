"""
This is just a sample code.

LevelEditor, ObjectHandler, ObjectPalette should be rewritten
to be game specific.

You can define object template class inheriting ObjectBase
to define properties shared by multiple object types.
When you are defining properties
you should specify their name, UI type, data type,
update function, default value, and value range.

Then you need implement ObjectPalette class inheriting ObjectPaletteBase,
and in the populate function you can define ObjectPalette tree structure.
"""
from . import ObjectGlobals as OG
from .ObjectPaletteBase import ObjectBase, ObjectPaletteBase

class ObjectProp(ObjectBase):

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        ObjectBase.__init__(self, *args, **kw)
        self.properties['Abc'] = [OG.PROP_UI_RADIO, OG.PROP_STR, None, 'a', ['a', 'b', 'c']]

class ObjectSmiley(ObjectProp):

    def __init__(self, *args, **kw):
        if False:
            print('Hello World!')
        ObjectProp.__init__(self, *args, **kw)
        self.properties['123'] = [OG.PROP_UI_COMBO, OG.PROP_INT, None, 1, [1, 2, 3]]

class ObjectDoubleSmileys(ObjectProp):

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        ObjectProp.__init__(self, *args, **kw)
        self.properties['Distance'] = [OG.PROP_UI_SLIDE, OG.PROP_FLOAT, ('.updateDoubleSmiley', {'val': OG.ARG_VAL, 'obj': OG.ARG_OBJ}), 1.0, [0, 10, 0.1]]

class ObjectPalette(ObjectPaletteBase):

    def __init__(self):
        if False:
            print('Hello World!')
        ObjectPaletteBase.__init__(self)

    def populate(self):
        if False:
            i = 10
            return i + 15
        self.add('Prop')
        self.add('Double Smileys', 'Prop')
        self.add(ObjectSmiley(name='Smiley', model='models/smiley.egg', models=['models/smiley.egg', 'models/frowney.egg', 'models/jack.egg'], properties={'Happy': [OG.PROP_UI_CHECK, OG.PROP_BOOL, None, True], 'Number': [OG.PROP_UI_SPIN, OG.PROP_INT, ('.updateSmiley', {'val': OG.ARG_VAL, 'obj': OG.ARG_OBJ}), 1, [1, 10]]}), 'Prop')
        self.add(ObjectDoubleSmileys(name='H Double Smiley', createFunction=('.createDoubleSmiley', {})), 'Double Smileys')
        self.add(ObjectDoubleSmileys(name='V Double Smiley', createFunction=('.createDoubleSmiley', {'horizontal': False})), 'Double Smileys')
        self.add('Animal')
        self.add(ObjectBase(name='Panda', createFunction=('.createPanda', {}), anims=['models/panda-walk4.egg'], properties={}), 'Animal')
        self.add('BG')
        self.add(ObjectBase(name='Grass', createFunction=('.createGrass', {}), properties={}), 'BG')