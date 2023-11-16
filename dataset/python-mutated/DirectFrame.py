"""A DirectFrame is a basic DirectGUI component that acts as the base
class for various other components, and can also serve as a basic
container to hold other DirectGUI components.

A DirectFrame can have:

* A background texture (pass in path to image, or Texture Card)
* A midground geometry item (pass in geometry)
* A foreground text Node (pass in text string or OnscreenText)

Each of these has 1 or more states.  The same object can be used for
all states or each state can have a different text/geom/image (for
radio button and check button indicators, for example).

See the :ref:`directframe` page in the programming manual for a more in-depth
explanation and an example of how to use this class.
"""
__all__ = ['DirectFrame']
from panda3d.core import Filename, NodePath, PGItem, Texture
from . import DirectGuiGlobals as DGG
from .DirectGuiBase import DirectGuiWidget
from .OnscreenImage import OnscreenImage
from .OnscreenGeom import OnscreenGeom
from .OnscreenText import OnscreenText

class DirectFrame(DirectGuiWidget):
    DefDynGroups = ('text', 'geom', 'image')

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        optiondefs = (('pgFunc', PGItem, None), ('numStates', 1, None), ('state', self.inactiveInitState, None), ('image', None, self.setImage), ('geom', None, self.setGeom), ('text', None, self.setText), ('textMayChange', 1, None))
        self.defineoptions(kw, optiondefs, dynamicGroups=DirectFrame.DefDynGroups)
        DirectGuiWidget.__init__(self, parent)
        self.initialiseoptions(DirectFrame)

    def __reinitComponent(self, name, component_class, states, **kwargs):
        if False:
            return 10
        'Recreates the given component using the given keyword args.'
        assert name in ('geom', 'image', 'text')
        for c in range(self['numStates']):
            component_name = name + str(c)
            try:
                state = states[c]
            except IndexError:
                state = states[-1]
            if self.hascomponent(component_name):
                if state is None:
                    self.destroycomponent(component_name)
                else:
                    self[component_name + '_' + name] = state
            else:
                if state is None:
                    return
                kwargs[name] = state
                self.createcomponent(component_name, (), name, component_class, (), parent=self.stateNodePath[c], **kwargs)

    def clearText(self):
        if False:
            while True:
                i = 10
        self['text'] = None
        self.setText()

    def setText(self, text=None):
        if False:
            print('Hello World!')
        if text is not None:
            self['text'] = text
        text = self['text']
        if text is None or isinstance(text, str):
            text_list = (text,) * self['numStates']
        else:
            text_list = text
        self.__reinitComponent('text', OnscreenText, text_list, scale=1, mayChange=self['textMayChange'], sort=DGG.TEXT_SORT_INDEX)

    def clearGeom(self):
        if False:
            print('Hello World!')
        self['geom'] = None
        self.setGeom()

    def setGeom(self, geom=None):
        if False:
            print('Hello World!')
        if geom is not None:
            self['geom'] = geom
        geom = self['geom']
        if geom is None or isinstance(geom, NodePath) or isinstance(geom, str):
            geom_list = (geom,) * self['numStates']
        else:
            geom_list = geom
        self.__reinitComponent('geom', OnscreenGeom, geom_list, scale=1, sort=DGG.GEOM_SORT_INDEX)

    def clearImage(self):
        if False:
            return 10
        self['image'] = None
        self.setImage()

    def setImage(self, image=None):
        if False:
            return 10
        if image is not None:
            self['image'] = image
        image = self['image']
        if image is None or isinstance(image, NodePath) or isinstance(image, Texture) or isinstance(image, str) or isinstance(image, Filename) or (len(image) == 2 and isinstance(image[0], str) and isinstance(image[1], str)):
            image_list = (image,) * self['numStates']
        else:
            image_list = image
        self.__reinitComponent('image', OnscreenImage, image_list, scale=1, sort=DGG.IMAGE_SORT_INDEX)