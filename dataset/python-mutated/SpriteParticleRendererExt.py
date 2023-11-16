from panda3d.core import ConfigVariableString
from panda3d.physics import SpriteParticleRenderer

class SpriteParticleRendererExt(SpriteParticleRenderer):
    """
    Contains methods to extend functionality
    of the SpriteParticleRenderer class
    """
    sourceTextureName = None
    sourceFileName = None
    sourceNodeName = None

    def getSourceTextureName(self):
        if False:
            print('Hello World!')
        if self.sourceTextureName is None:
            SpriteParticleRendererExt.sourceTextureName = ConfigVariableString('particle-sprite-texture', 'maps/lightbulb.rgb').value
        return self.sourceTextureName

    def setSourceTextureName(self, name):
        if False:
            return 10
        self.sourceTextureName = name

    def setTextureFromFile(self, fileName=None):
        if False:
            i = 10
            return i + 15
        if fileName is None:
            fileName = self.getSourceTextureName()
        t = base.loader.loadTexture(fileName)
        if t is not None:
            self.setTexture(t, t.getYSize())
            self.setSourceTextureName(fileName)
            return True
        else:
            print("Couldn't find rendererSpriteTexture file: %s" % fileName)
            return False

    def addTextureFromFile(self, fileName=None):
        if False:
            i = 10
            return i + 15
        if self.getNumAnims() == 0:
            return self.setTextureFromFile(fileName)
        if fileName is None:
            fileName = self.getSourceTextureName()
        t = base.loader.loadTexture(fileName)
        if t is not None:
            self.addTexture(t, t.getYSize())
            return True
        else:
            print("Couldn't find rendererSpriteTexture file: %s" % fileName)
            return False

    def getSourceFileName(self):
        if False:
            return 10
        if self.sourceFileName is None:
            SpriteParticleRendererExt.sourceFileName = ConfigVariableString('particle-sprite-model', 'models/misc/smiley').value
        return self.sourceFileName

    def setSourceFileName(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.sourceFileName = name

    def getSourceNodeName(self):
        if False:
            print('Hello World!')
        if self.sourceNodeName is None:
            SpriteParticleRendererExt.sourceNodeName = ConfigVariableString('particle-sprite-node', '**/*').value
        return self.sourceNodeName

    def setSourceNodeName(self, name):
        if False:
            i = 10
            return i + 15
        self.sourceNodeName = name

    def setTextureFromNode(self, modelName=None, nodeName=None, sizeFromTexels=False):
        if False:
            i = 10
            return i + 15
        if modelName is None:
            modelName = self.getSourceFileName()
            if nodeName is None:
                nodeName = self.getSourceNodeName()
        m = base.loader.loadModel(modelName)
        if m is None:
            print("SpriteParticleRendererExt: Couldn't find model: %s!" % modelName)
            return False
        np = m.find(nodeName)
        if np.isEmpty():
            print("SpriteParticleRendererExt: Couldn't find node: %s!" % nodeName)
            m.removeNode()
            return False
        self.setFromNode(np, modelName, nodeName, sizeFromTexels)
        self.setSourceFileName(modelName)
        self.setSourceNodeName(nodeName)
        m.removeNode()
        return True

    def addTextureFromNode(self, modelName=None, nodeName=None, sizeFromTexels=False):
        if False:
            print('Hello World!')
        if self.getNumAnims() == 0:
            return self.setTextureFromNode(modelName, nodeName, sizeFromTexels)
        if modelName is None:
            modelName = self.getSourceFileName()
            if nodeName is None:
                nodeName = self.getSourceNodeName()
        m = base.loader.loadModel(modelName)
        if m is None:
            print("SpriteParticleRendererExt: Couldn't find model: %s!" % modelName)
            return False
        np = m.find(nodeName)
        if np.isEmpty():
            print("SpriteParticleRendererExt: Couldn't find node: %s!" % nodeName)
            m.removeNode()
            return False
        self.addFromNode(np, modelName, nodeName, sizeFromTexels)
        m.removeNode()
        return True