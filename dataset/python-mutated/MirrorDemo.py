"""This file demonstrates one way to create a mirror effect in Panda.
Call :func:`setupMirror()` to create a mirror in the world that reflects
everything in front of it.

The approach taken here is to create an offscreen buffer with its own
camera that renders its view into a texture, which is then applied to
the mirror geometry.  The mirror's camera is repositioned each frame
with a task to keep it always on the opposite side of the mirror from
the main camera.

This demonstrates the basic interface for offscreen
render-to-a-texture in Panda.  Similar approaches can be used for
related effects, such as a remote spy camera presenting its view onto
a closed-circuit television screen.

In this example the mirror itself is always perfectly flat--it's just
a single polygon, after all--but small distortions of the mirror
surface are possible, like a funhouse mirror.  However, the reflection
itself is always basically planar; for more accurate convex
reflections, you will need to use a sphere map or a cube map."""
__all__ = ['setupMirror', 'showFrustum']
from panda3d.core import Camera, CardMaker, CullFaceAttrib, GeomNode, Lens, NodePath, PerspectiveLens, Plane, PlaneNode, Point3, Vec3
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr

def setupMirror(name, width, height, rootCamera=None, bufferSize=256, clearColor=None):
    if False:
        for i in range(10):
            print('nop')
    if rootCamera is None:
        rootCamera = base.camera
    root = render.attachNewNode(name)
    cm = CardMaker('mirror')
    cm.setFrame(width / 2.0, -width / 2.0, -height / 2.0, height / 2.0)
    cm.setHasUvs(1)
    card = root.attachNewNode(cm.generate())
    plane = Plane(Vec3(0, 1, 0), Point3(0, 0, 0))
    planeNode = PlaneNode('mirrorPlane')
    planeNode.setPlane(plane)
    planeNP = root.attachNewNode(planeNode)
    buffer = base.win.makeTextureBuffer(name, bufferSize, bufferSize)
    if clearColor is None:
        buffer.setClearColor(base.win.getClearColor())
    else:
        buffer.setClearColor(clearColor)
    dr = buffer.makeDisplayRegion()
    camera = Camera('mirrorCamera')
    lens = PerspectiveLens()
    lens.setFilmSize(width, height)
    camera.setLens(lens)
    cameraNP = planeNP.attachNewNode(camera)
    dr.setCamera(cameraNP)
    dummy = NodePath('dummy')
    dummy.setAttrib(CullFaceAttrib.makeReverse())
    dummy.setClipPlane(planeNP)
    camera.setInitialState(dummy.getState())

    def moveCamera(task, cameraNP=cameraNP, plane=plane, planeNP=planeNP, card=card, lens=lens, width=width, height=height, rootCamera=rootCamera):
        if False:
            while True:
                i = 10
        cameraNP.setMat(rootCamera.getMat(planeNP) * plane.getReflectionMat())
        cameraNP.setR(planeNP.getR() - 180)
        ul = cameraNP.getRelativePoint(card, Point3(-width / 2.0, 0, height / 2.0))
        ur = cameraNP.getRelativePoint(card, Point3(width / 2.0, 0, height / 2.0))
        ll = cameraNP.getRelativePoint(card, Point3(-width / 2.0, 0, -height / 2.0))
        lr = cameraNP.getRelativePoint(card, Point3(width / 2.0, 0, -height / 2.0))
        camvec = planeNP.getPos() - cameraNP.getPos()
        camdist = camvec.length()
        ul.setY(camdist)
        ur.setY(camdist)
        ll.setY(camdist)
        lr.setY(camdist)
        lens.setFrustumFromCorners(ul, ur, ll, lr, Lens.FCCameraPlane | Lens.FCOffAxis | Lens.FCAspectRatio)
        return Task.cont
    taskMgr.add(moveCamera, name, priority=40)
    card.setTexture(buffer.getTexture())
    return root

def showFrustum(np):
    if False:
        for i in range(10):
            print('nop')
    cameraNP = np.find('**/+Camera')
    camera = cameraNP.node()
    lens = camera.getLens()
    geomNode = GeomNode('frustum')
    geomNode.addGeom(lens.makeGeometry())
    cameraNP.attachNewNode(geomNode)
if __name__ == '__main__':
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    panda = base.loader.loadModel('panda')
    panda.setH(180)
    panda.setPos(0, 10, -2.5)
    panda.setScale(0.5)
    panda.reparentTo(base.render)
    myMirror = setupMirror('mirror', 10, 10, bufferSize=1024, clearColor=(0, 0, 1, 1))
    myMirror.setPos(0, 15, 2.5)
    myMirror.setH(180)
    base.run()