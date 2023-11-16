from raytracing import *
import raytracing.thorlabs as thorlabs
import raytracing.eo as eo
import raytracing.olympus as olympus
path = ImagingPath()


class test(Objective):
    def __init__(self):
        super(test, self).__init__(f=90/2,
                                         NA=0.5,
                                         focusToFocusLength=137,
                                         backAperture=54,
                                         workingDistance=20,
                                         label='MVPlapo2XC',
                                         url="")

path.append(Space(d=100))
path.append(eo.PN_33_921())
path.append(Space(d=300))
path.append(eo.PN_88_593())
path.append(Space(d=183))
path.append(olympus.MVPlapo2XC())
path.append(Space(d=100))
path.objectHeight = 22 # object height (full).
# FIXME: Python 2.7 r e q u i r e s 1 . 0 , not 1 ( f l o a t )
path.objectPosition = 0.0 # always at z=0 f o r now.
# FIXME: Python 2.7 r e q u i r e s 1 . 0 , not 1 ( f l o a t )
path.fanAngle = 0.27 # full fan angle for rays
path.fanNumber = 10 # number of rays in fan
path.rayNumber = 3
path.display()
#
# path.fanAngle = 0.27  # full fan angle for rays
# path.fanNumber = 10  # number of rays in fan
# path.rayNumber = 3
# path.objectHeight = 7.3
# obj = test()
# #path.append(Space(d=20))
# path.append(obj.flipOrientation())
# path.append(Space(d=183))
# path.append(eo.PN_88_593())
# path.append(Space(d=300))
# path.append(eo.PN_33_921())
# path.append(Space(d=100))
# path.display()
