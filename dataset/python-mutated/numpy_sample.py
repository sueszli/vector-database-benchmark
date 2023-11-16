import Ogre
import Ogre.Bites
import Ogre.RTShader
import numpy as np
from matplotlib import pyplot

def main():
    if False:
        return 10
    app = Ogre.Bites.ApplicationContext('PySample')
    app.initApp()
    root = app.getRoot()
    scn_mgr = root.createSceneManager()
    shadergen = Ogre.RTShader.ShaderGenerator.getSingleton()
    shadergen.addSceneManager(scn_mgr)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 1] = np.mgrid[0:256, 0:256][1]
    ogre_img = Ogre.Image()
    ogre_img.loadDynamicImage(arr, 256, 256, Ogre.PF_BYTE_RGB)
    Ogre.TextureManager.getSingleton().loadImage('gradient', 'General', ogre_img)
    mat = Ogre.MaterialManager.getSingleton().create('gradient_mat', 'General')
    rpass = mat.getTechniques()[0].getPasses()[0]
    rpass.setLightingEnabled(False)
    rpass.createTextureUnitState('gradient')
    rect = scn_mgr.createScreenSpaceRect(True)
    rect.setCorners(-0.5, 0.5, 0.5, -0.5)
    rect.setMaterial(mat)
    scn_mgr.getRootSceneNode().createChildSceneNode().attachObject(rect)
    cam = scn_mgr.createCamera('myCam')
    win = app.getRenderWindow()
    vp = win.addViewport(cam)
    gray = np.array([0.3, 0.3, 0.3])
    vp.setBackgroundColour(gray)
    root.startRendering()
    mem = np.empty((win.getHeight(), win.getWidth(), 3), dtype=np.uint8)
    pb = Ogre.PixelBox(win.getWidth(), win.getHeight(), 1, Ogre.PF_BYTE_RGB, mem)
    win.copyContentsToMemory(pb, pb)
    pyplot.imsave('screenshot.png', mem)
if __name__ == '__main__':
    main()