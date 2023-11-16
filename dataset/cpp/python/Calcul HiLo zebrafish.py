import numpy as np

""" Calculs des différents paramètres pour le microscope HiLo, distance en mm"""
""" Les valeurs choisies sont actuellement celles du macro HiLo, mais elles seront modifiées au fil de la 
progression du microscope """

# Objective

ObjectiveNA = 0.5
ObjectiveWorkingDistance = 3.5

ObjectiveNA = 1.0
ObjectiveWorkingDistance = 2
FocalOfLensHabituallyUsedWithObjective = 180
ObjectiveMagnification = 20
FocalObjective = FocalOfLensHabituallyUsedWithObjective/ObjectiveMagnification
ObjectiveDiameterEntrancePupil = 2*FocalObjective*ObjectiveNA
FocalOfTubeLens = 180
FieldNumber = 26.5
FieldNumber = 22
Magnification = FocalOfTubeLens/FocalObjective
ObjectiveMaximumFOV = FieldNumber/Magnification

# Camera
CameraDiagonal = 18.826
CameraMaximumFOV = CameraDiagonal/Magnification
CameraPixelSize = 6.5  # µm

# Suite objective
ObjectiveINV = ObjectiveNA*(ObjectiveMaximumFOV*0.5)
ObjectiveNAWithINV = ObjectiveINV/(ObjectiveDiameterEntrancePupil*0.5)
SourceMinAngleAtObjective = np.degrees(ObjectiveNAWithINV)

# Illuminator system, see "Code illuminateur.py, The last lens before objective is at this moment L1 but it will
# probably be the optotune lens"
L1Focal = 40
L1Diameter = 30
IlluminatorMagnification = 2.85714  # Determine with "Code illuminateur.py"
DistanceBetweenL1andObj = 120

# Source (Diffuser)
SourceDiameter = 3.15
SourceDiameterToFillEntrancePupil = ObjectiveDiameterEntrancePupil/IlluminatorMagnification
SourceDiameterAtEntrancePupil = SourceDiameter*IlluminatorMagnification
SourceMaxAngleAtObjective = np.degrees(np.sin(((L1Diameter-SourceDiameterAtEntrancePupil)/2)/DistanceBetweenL1andObj))
DiffuserNA = 1

# Fiber and Speckles
Wavelength = 488*10**-6
FiberRadius = 0.75
FiberNA = 0.5
AverageGrainSize = Wavelength/(2*FiberNA)
MaxGrainNumber = np.pi*(FiberRadius/AverageGrainSize)**2
# DistanceBetweenFiberDifuser = ?

# Resolution
IndexBetweenObjectiveAndSample = 1.333  # water
ResolutionLateralTheoretical = 1.22*Wavelength/ObjectiveNA
ResolutionAxialTheoretical = IndexBetweenObjectiveAndSample*(Wavelength/ObjectiveNA**2+ResolutionLateralTheoretical/(ObjectiveMagnification*ObjectiveNA))

print("FocalObjective = {} mm".format(FocalObjective),
      "ObjectiveDiameterEntrancePupil = {} mm".format(ObjectiveDiameterEntrancePupil),
      "Magnification = {}".format(Magnification),
      "ObjectiveMaximumFOV = {} mm".format(ObjectiveMaximumFOV),
      "CameraMaximumFOV = {} mm".format(CameraMaximumFOV),
      "ObjectiveINV = {}".format(ObjectiveINV),
      "ObjectiveNAWithINV = {}".format(ObjectiveNAWithINV),
      "SourceMinAngleAtObjective = {}°".format(SourceMinAngleAtObjective),
      "IlluminatorMagnification = {}".format(IlluminatorMagnification),
      "SourceOriginalDiameterToFillEntrancePupil = {} mm".format(SourceDiameterToFillEntrancePupil),
      "SourceDiameterAtEntrancePupil = {} mm".format(SourceDiameterAtEntrancePupil),
      "SourceMaxAngleAtObjective = {}°".format(SourceMaxAngleAtObjective),
      "AverageGrainSize = {} nm".format(AverageGrainSize*10**6),
      "MaxGrainNumber = {}".format(MaxGrainNumber),
      "ResolutionLateralTheoretical = {} µm".format(ResolutionLateralTheoretical*10**3),
      "ResolutionAxialTheoretical = {} µm".format(ResolutionAxialTheoretical*10**3),
      sep="\n")


