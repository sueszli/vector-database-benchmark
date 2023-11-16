from raytracing import *
import raytracing.thorlabs as thorlabs
import raytracing.eo as eo
import raytracing.olympus as olympus
path = ImagingPath()

class test(Objective):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(test, self).__init__(f=90 / 2, NA=0.5, focusToFocusLength=137, backAperture=54, workingDistance=20, label='MVPlapo2XC', url='')
path.append(Space(d=100))
path.append(eo.PN_33_921())
path.append(Space(d=300))
path.append(eo.PN_88_593())
path.append(Space(d=183))
path.append(olympus.MVPlapo2XC())
path.append(Space(d=100))
path.objectHeight = 22
path.objectPosition = 0.0
path.fanAngle = 0.27
path.fanNumber = 10
path.rayNumber = 3
path.display()