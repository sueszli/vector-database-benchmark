try:
    from raytracing import *
except ImportError:
    raise ImportError('Raytracing module not found: "pip install raytracing"')
from raytracing import *

class test(Objective):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(test, self).__init__(f=180 / 20, NA=1.0, focusToFocusLength=75, backAperture=18, workingDistance=2, label='XLUMPLFN20XW', url='https://www.olympus-lifescience.com/en/objectives/lumplfln-w/')

class Sparq:

    @staticmethod
    def illuminationFromObjective():
        if False:
            i = 10
            return i + 15
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        obj = test()
        obj.flipOrientation()
        illumination = ImagingPath()
        illumination.label = 'Sparq illumination'
        illumination.objectHeight = 0.7
        illumination.fanAngle = 0.5
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = True
        illumination.append(obj)
        illumination.append(Space(d=120))
        illumination.append(L1)
        illumination.append(Space(d=40))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=30))
        illumination.append(L2)
        illumination.append(Space(d=57))
        illumination.append(L3)
        illumination.append(Space(d=40))
        illumination.append(L4)
        illumination.append(Space(d=20))
        illumination.append(LExc)
        illumination.append(Space(d=45))
        return illumination

    def illuminationFromSource():
        if False:
            return 10
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        obj = test()
        illumination = ImagingPath()
        illumination.label = 'Sparq illumination with Excelitas'
        illumination.objectHeight = 5
        illumination.fanAngle = 0.25
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = True
        illumination.append(Space(d=60))
        illumination.append(LExc)
        illumination.append(Space(d=20))
        illumination.append(L4)
        illumination.append(Space(d=40))
        illumination.append(L3)
        illumination.append(Space(d=57))
        illumination.append(L2)
        illumination.append(Space(d=30))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=40))
        illumination.append(L1)
        illumination.append(Space(d=80))
        illumination.append(Aperture(diameter=20, label='Nosepiece'))
        illumination.append(Space(d=200))
        return illumination

    def illuminationFromObjectiveWithOptotune():
        if False:
            while True:
                i = 10
        optotuneFocal = 100
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        obj = test()
        obj.flipOrientation()
        illumination = ImagingPath()
        illumination.label = 'Sparq illumination with Optotune'
        illumination.objectHeight = 0.7
        illumination.fanAngle = 0.5
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = False
        illumination.append(obj)
        illumination.append(Space(d=10))
        illumination.append(Optotune)
        illumination.append(Space(d=45 + 47.5))
        illumination.append(L1)
        illumination.append(Space(d=40))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=30))
        illumination.append(L2)
        illumination.append(Space(d=57))
        illumination.append(L3)
        illumination.append(Space(d=40))
        illumination.append(L4)
        illumination.append(Space(d=20))
        illumination.append(LExc)
        illumination.append(Space(d=45))
        return illumination

    def illuminationFromSourceWithOptotune():
        if False:
            while True:
                i = 10
        optotuneFocal = 100
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        obj = test()
        illumination = ImagingPath()
        illumination.label = 'Sparq illumination with Optotune'
        illumination.objectHeight = 3.15
        illumination.fanAngle = 0.5
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = False
        illumination.append(Space(d=45))
        illumination.append(LExc)
        illumination.append(Space(d=20))
        illumination.append(L4)
        illumination.append(Space(d=40))
        illumination.append(L3)
        illumination.append(Space(d=57))
        illumination.append(L2)
        illumination.append(Space(d=30))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=40))
        illumination.append(L1)
        illumination.append(Space(d=120))
        illumination.append(Optotune)
        illumination.append(obj)
        return illumination

    def illuminationFromSourceWithOptotuneAndDivergentLens():
        if False:
            print('Hello World!')
        optotuneFocal = 58.5
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        L5 = Lens(f=-75, diameter=35, label='Divergent lens')
        obj = olympus.XLUMPlanFLN20X()
        illumination = ImagingPath()
        illumination.label = 'Sparq illumination with Optotune and divergent lens'
        illumination.objectHeight = 3.15
        illumination.fanAngle = 0.5
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = False
        illumination.append(Space(d=45))
        illumination.append(LExc)
        illumination.append(Space(d=20))
        illumination.append(L4)
        illumination.append(Space(d=40))
        illumination.append(L3)
        illumination.append(Space(d=57))
        illumination.append(L2)
        illumination.append(Space(d=30))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=40))
        illumination.append(L1)
        illumination.append(Space(d=65))
        illumination.append(L5)
        illumination.append(Space(d=10))
        illumination.append(Optotune)
        illumination.append(Space(d=45))
        illumination.append(obj)
        return illumination

    def illuminationFromObjectiveToCamera():
        if False:
            i = 10
            return i + 15
        optotuneFocal = 40
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        obj = test()
        tubeLens = Lens(f=180, diameter=60, label='Tube Lens')
        obj.flipOrientation()
        illumination = ImagingPath()
        illumination.label = 'Microscope system'
        illumination.objectHeight = 0.7
        illumination.fanAngle = 0.5
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = True
        "Raytracing doesn't let me change the space between object and objective... \n        I wanna change it to 3.5-2.7 = 0.8"
        illumination.append(obj)
        illumination.append(Space(d=10))
        illumination.append(Optotune)
        illumination.append(Space(d=55))
        illumination.append(tubeLens)
        illumination.append(Space(d=180))
        return illumination

    def illuminationFromCameraToObjective():
        if False:
            for i in range(10):
                print('nop')
        optotuneFocal = 40
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        obj = test()
        tubeLens = Lens(f=180, diameter=60, label='Tube Lens')
        illumination = ImagingPath()
        illumination.label = 'Microscope system'
        illumination.objectHeight = 5
        illumination.fanAngle = 0.07
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = True
        illumination.append(Space(d=180))
        illumination.append(tubeLens)
        illumination.append(Space(d=55))
        illumination.append(Optotune)
        illumination.append(Space(d=10))
        illumination.append(obj)
        return illumination

    def tracingForIlluminatorMagnification():
        if False:
            while True:
                i = 10
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        illumination = ImagingPath()
        illumination.label = 'Illumination only illuminator'
        illumination.objectHeight = 2
        illumination.fanAngle = 0.08726
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = True
        illumination.append(Space(d=45))
        illumination.append(LExc)
        illumination.append(Space(d=20))
        illumination.append(L4)
        illumination.append(Space(d=40))
        illumination.append(L3)
        illumination.append(Space(d=57))
        illumination.append(L2)
        illumination.append(Space(d=30))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=40))
        illumination.append(L1)
        illumination.append(Space(d=120))
        illumination.append(Space(d=30))
        illumination.append(Lens(f=30, diameter=35, label='Added lens'))
        illumination.append(Space(d=90))
        illumination.append(Lens(f=60, diameter=35, label='Added lens'))
        illumination.append(Space(d=100))
        illumination.append(Space(d=30))
        illumination.append(Lens(f=30, diameter=25, label='ajout'))
        illumination.append(Space(d=230))
        illumination.append(Lens(f=200, diameter=25, label='ajout 2'))
        illumination.append(Space(d=210))
        return illumination

    def investigationOptotuneAtBackAperture():
        if False:
            for i in range(10):
                print('nop')
        optotuneFocal = 40
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        obj = test()
        illumination = ImagingPath()
        illumination.label = 'Investigation Optotune at Back Aperture'
        illumination.objectHeight = 10
        illumination.fanAngle = 0
        illumination.fanNumber = 15
        illumination.rayNumber = 15
        illumination.showImages = False
        illumination.append(Space(d=20))
        illumination.append(Optotune)
        illumination.append(Space(d=10))
        illumination.append(obj)
        illumination.append(Space(d=1))
        return illumination

    def investigationOptotuneAndCamera():
        if False:
            i = 10
            return i + 15
        optotuneFocal = 40
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        obj1 = test()
        obj2 = test()
        obj2.flipOrientation()
        illumination = ImagingPath()
        illumination.label = 'Investigation Optotune at Back Aperture'
        illumination.objectHeight = 10
        illumination.fanAngle = 0
        illumination.fanNumber = 15
        illumination.rayNumber = 15
        illumination.showImages = False
        illumination.append(Space(d=20))
        illumination.append(Optotune)
        illumination.append(Space(d=10))
        illumination.append(obj1)
        illumination.append(Space(d=-5.4))
        illumination.append(obj2)
        illumination.append(Space(d=10))
        illumination.append(Optotune)
        illumination.append(Space(d=20))
        return illumination

    def illuminationFormSourceWithOptotuneAndCamera():
        if False:
            print('Hello World!')
        optotuneFocal = 40
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        Optotune = Lens(f=optotuneFocal, diameter=16, label='Optotune')
        tubeLens = Lens(f=180, diameter=60, label='$tubeLens$')
        obj1 = test()
        obj2 = test()
        obj2.flipOrientation()
        illumination = ImagingPath()
        illumination.label = 'Sparq illumination with Optotune'
        illumination.objectHeight = 3.15
        illumination.fanAngle = 0.5
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = False
        illumination.append(Space(d=45))
        illumination.append(LExc)
        illumination.append(Space(d=20))
        illumination.append(L4)
        illumination.append(Space(d=40))
        illumination.append(L3)
        illumination.append(Space(d=57))
        illumination.append(L2)
        illumination.append(Space(d=30))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=40))
        illumination.append(L1)
        illumination.append(Space(d=110))
        illumination.append(Optotune)
        illumination.append(Space(d=10))
        illumination.append(obj1)
        illumination.append(Space(d=-5.4))
        illumination.append(obj2)
        illumination.append(Space(d=10))
        illumination.append(Optotune)
        illumination.append(Space(d=55))
        illumination.append(tubeLens)
        illumination.append(Space(d=180))
        return illumination

    def investigationObjectiveBehaviour():
        if False:
            while True:
                i = 10
        L1 = Lens(f=40, diameter=30, label='$L_1$')
        L2 = Lens(f=30, diameter=20, label='$L_2$')
        L3 = Lens(f=-35, diameter=22, label='$L_3$')
        L4 = Lens(f=75, diameter=32, label='$L_4$')
        LExc = Lens(f=45, diameter=35, label='Exc')
        obj = test()
        illumination = ImagingPath()
        illumination.label = 'Sparq illumination with Excelitas'
        illumination.objectHeight = 3.15
        illumination.fanAngle = 0.25
        illumination.fanNumber = 11
        illumination.rayNumber = 3
        illumination.showImages = True
        illumination.append(Space(d=45))
        illumination.append(LExc)
        illumination.append(Space(d=20))
        illumination.append(L4)
        illumination.append(Space(d=40))
        illumination.append(L3)
        illumination.append(Space(d=57))
        illumination.append(L2)
        illumination.append(Space(d=30))
        illumination.append(Aperture(diameter=30, label='CF'))
        illumination.append(Space(d=20))
        illumination.append(Aperture(diameter=30, label='AF'))
        illumination.append(Space(d=40))
        illumination.append(L1)
        illumination.append(Space(d=80))
        illumination.append(Aperture(diameter=20, label='Nosepiece'))
        illumination.append(Space(d=40))
        illumination.append(obj)
        return illumination

    def Optique():
        if False:
            i = 10
            return i + 15
        L1 = Lens(f=200)
        Lobj = Lens(f=5)
        illumination = ImagingPath()
        illumination.showImage = True
        illumination.append(Space(d=10))
        illumination.append(Lobj)
        illumination.append(Space(d=205))
        illumination.append(L1)
        illumination.append(Space(2000))
if __name__ == '__main__':
    Sparq.investigationObjectiveBehaviour().display()