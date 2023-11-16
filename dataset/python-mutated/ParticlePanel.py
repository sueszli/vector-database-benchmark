"""PANDA3D Particle Panel"""
__all__ = ['ParticlePanel']
from panda3d.core import ColorBlendAttrib, Filename, Point2, Point3, Vec3, Vec4, getModelPath
from panda3d.physics import BaseParticleEmitter, BaseParticleRenderer, BoxEmitter, ColorInterpolationFunctionConstant, ColorInterpolationFunctionLinear, ColorInterpolationFunctionSinusoid, ColorInterpolationFunctionStepwave, DiscEmitter, GeomParticleRenderer, LinearCylinderVortexForce, LinearDistanceForce, LinearFrictionForce, LinearJitterForce, LinearNoiseForce, LinearSinkForce, LinearSourceForce, LinearUserDefinedForce, LinearVectorForce, LineEmitter, LineParticleRenderer, PointEmitter, PointParticleRenderer, RectangleEmitter, RingEmitter, SparkleParticleRenderer, SphereSurfaceEmitter, SphereVolumeEmitter, SpriteAnim, SpriteParticleRenderer, TangentRingEmitter
from panda3d.direct import getParticlePath
from direct.tkwidgets.AppShell import AppShell
from direct.tkwidgets import Dial
from direct.tkwidgets import Floater
from direct.tkwidgets import Slider
from direct.tkwidgets import VectorWidgets
from direct.tkpanels import Placer
from direct.particles import ForceGroup
from direct.particles import Particles
from direct.particles import ParticleEffect
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.simpledialog import askstring
import Pmw
import os
import tkinter as tk

class ParticlePanel(AppShell):
    appname = 'Particle Panel'
    frameWidth = 375
    frameHeight = 675
    usecommandarea = 0
    usestatusarea = 0
    balloonState = 'both'

    def __init__(self, particleEffect=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        INITOPT = Pmw.INITOPT
        optiondefs = (('title', self.appname, None),)
        self.defineoptions(kw, optiondefs)
        if particleEffect is not None:
            self.particleEffect = particleEffect
        else:
            base.enableParticles()
            particles = Particles.Particles()
            particles.setBirthRate(0.02)
            particles.setLitterSize(10)
            particles.setLitterSpread(0)
            particles.setFactory('PointParticleFactory')
            particles.setRenderer('PointParticleRenderer')
            particles.setEmitter('SphereVolumeEmitter')
            particles.enable()
            pe = ParticleEffect.ParticleEffect('effect-1', particles)
            self.particleEffect = pe
            pe.reparentTo(render)
            pe.enable()
        AppShell.__init__(self)
        self.initialiseoptions(ParticlePanel)
        self.selectEffectNamed(next(iter(self.effectsDict)))
        self.updateMenusAndLabels()
        for forceGroup in self.particleEffect.getForceGroupList():
            self.addForceGroupNotebookPage(self.particleEffect, forceGroup)

    def appInit(self):
        if False:
            print('Hello World!')
        self.widgetDict = {}
        self.variableDict = {}
        self.effectsDict = {}
        self.effectsDict[self.particleEffect.getName()] = self.particleEffect
        self.forcePagesDict = {}

    def createInterface(self):
        if False:
            while True:
                i = 10
        interior = self.interior()
        self.menuBar.deletemenuitems('File', 0, 0)
        self.menuBar.addmenuitem('File', 'command', label='Load Params', command=self.loadParticleEffectFromFile)
        self.menuBar.addmenuitem('File', 'command', label='Save Params', command=self.saveParticleEffectToFile)
        self.menuBar.addmenuitem('File', 'command', label='Print Params', command=lambda s=self: s.particles.printParams())
        self.menuBar.addmenuitem('File', 'command', 'Quit this application', label='Quit', command=self.quit)
        self.menuBar.addmenu('ParticleMgr', 'ParticleMgr Operations')
        self.particleMgrActive = tk.IntVar()
        self.particleMgrActive.set(base.isParticleMgrEnabled())
        self.menuBar.addmenuitem('ParticleMgr', 'checkbutton', 'Enable/Disable ParticleMgr', label='Active', variable=self.particleMgrActive, command=self.toggleParticleMgr)
        labelFrame = tk.Frame(interior)
        self.effectsLabel = tk.Menubutton(labelFrame, width=10, relief=tk.RAISED, borderwidth=2, font=('MSSansSerif', 12, 'bold'), activebackground='#909090')
        self.effectsLabelMenu = tk.Menu(self.effectsLabel, tearoff=0)
        self.effectsLabel['menu'] = self.effectsLabelMenu
        self.effectsLabel.pack(side=tk.LEFT, fill='x', expand=1)
        self.bind(self.effectsLabel, 'Select effect to configure or create new effect')
        self.effectsLabelMenu.add_command(label='Create New Effect', command=self.createNewEffect)
        self.effectsLabelMenu.add_command(label='Select Particle Effect', command=lambda s=self: base.direct.select(s.particleEffect))
        self.effectsLabelMenu.add_command(label='Place Particle Effect', command=lambda s=self: Placer.place(s.particleEffect))

        def togglePEVis(s=self):
            if False:
                i = 10
                return i + 15
            if s.particleEffect.isHidden():
                s.particleEffect.show()
            else:
                s.particleEffect.hide()
        self.effectsLabelMenu.add_command(label='Toggle Effect Vis', command=togglePEVis)
        self.effectsEnableMenu = tk.Menu(self.effectsLabelMenu, tearoff=0)
        self.effectsLabelMenu.add_cascade(label='Enable/Disable', menu=self.effectsEnableMenu)
        self.effectsLabelMenu.add_separator()
        self.particlesLabel = tk.Menubutton(labelFrame, width=10, relief=tk.RAISED, borderwidth=2, font=('MSSansSerif', 12, 'bold'), activebackground='#909090')
        self.particlesLabelMenu = tk.Menu(self.particlesLabel, tearoff=0)
        self.particlesLabel['menu'] = self.particlesLabelMenu
        self.particlesLabel.pack(side=tk.LEFT, fill='x', expand=1)
        self.bind(self.particlesLabel, 'Select particles object to configure ' + 'or add new particles object to current effect')
        self.particlesLabelMenu.add_command(label='Create New Particles', command=self.createNewParticles)
        self.particlesEnableMenu = tk.Menu(self.particlesLabelMenu, tearoff=0)
        self.particlesLabelMenu.add_cascade(label='Enable/Disable', menu=self.particlesEnableMenu)
        self.particlesLabelMenu.add_separator()
        self.forceGroupLabel = tk.Menubutton(labelFrame, width=10, relief=tk.RAISED, borderwidth=2, font=('MSSansSerif', 12, 'bold'), activebackground='#909090')
        self.forceGroupLabelMenu = tk.Menu(self.forceGroupLabel, tearoff=0)
        self.forceGroupLabel['menu'] = self.forceGroupLabelMenu
        self.forceGroupLabel.pack(side=tk.LEFT, fill='x', expand=1)
        self.bind(self.forceGroupLabel, 'Select force group to configure ' + 'or add a new force group to current effect')
        self.forceGroupLabelMenu.add_command(label='Create New ForceGroup', command=self.createNewForceGroup)
        self.forceGroupEnableMenu = tk.Menu(self.forceGroupLabelMenu, tearoff=0)
        self.forceGroupLabelMenu.add_cascade(label='Enable/Disable', menu=self.forceGroupEnableMenu)
        self.forceGroupLabelMenu.add_separator()
        labelFrame.pack(fill='x', expand=0)
        self.mainNotebook = Pmw.NoteBook(interior)
        self.mainNotebook.pack(fill=tk.BOTH, expand=1)
        systemPage = self.mainNotebook.add('System')
        factoryPage = self.mainNotebook.add('Factory')
        emitterPage = self.mainNotebook.add('Emitter')
        rendererPage = self.mainNotebook.add('Renderer')
        forcePage = self.mainNotebook.add('Force')
        self.mainNotebook['raisecommand'] = self.updateInfo
        systemFloaterDefs = (('System', 'Pool Size', 'Max number of simultaneous particles', self.setSystemPoolSize, 1.0, 2000000, 1.0), ('System', 'Birth Rate', 'Seconds between particle births', self.setSystemBirthRate, 0.0, None, None), ('System', 'Litter Size', 'Number of particle created at each birth', self.setSystemLitterSize, 1.0, 2147483647, 1.0), ('System', 'Litter Spread', 'Variation in litter size', self.setSystemLitterSpread, 0.0, 2147483647, 1.0), ('System', 'Lifespan', 'Age in seconds at which the system (vs. particles) should die', self.setSystemLifespan, 0.0, None, None))
        self.createFloaters(systemPage, systemFloaterDefs)
        self.createCheckbutton(systemPage, 'System', 'Render Space Velocities', 'On: velocities are in render space; ' + 'Off: velocities are in particle local space', self.toggleSystemLocalVelocity, 0)
        self.createCheckbutton(systemPage, 'System', 'System Grows Older', 'On: system has a lifespan', self.toggleSystemGrowsOlder, 0)
        pos = self.createVector3Entry(systemPage, 'System', 'Pos', 'Particle system position', command=self.setSystemPos)
        pos.addMenuItem('Popup Placer Panel', Placer.Placer)
        hpr = self.createVector3Entry(systemPage, 'System', 'Hpr', 'Particle system orientation', fGroup_labels=('H', 'P', 'R'), command=self.setSystemHpr)
        hpr.addMenuItem('Popup Placer Panel', Placer.Placer)
        self.createOptionMenu(factoryPage, 'Factory', 'Factory Type', 'Select type of particle factory', ('PointParticleFactory', 'ZSpinParticleFactory'), self.selectFactoryType)
        factoryWidgets = (('Factory', 'Life Span', 'Average particle lifespan in seconds', self.setFactoryLifeSpan, 0.0, None, None), ('Factory', 'Life Span Spread', 'Variation in lifespan', self.setFactoryLifeSpanSpread, 0.0, None, None), ('Factory', 'Mass', 'Average particle mass', self.setFactoryParticleMass, 0.001, None, None), ('Factory', 'Mass Spread', 'Variation in particle mass', self.setFactoryParticleMassSpread, 0.0, None, None), ('Factory', 'Terminal Velocity', 'Cap on average particle velocity', self.setFactoryTerminalVelocity, 0.0, None, None), ('Factory', 'Terminal Vel. Spread', 'Variation in terminal velocity', self.setFactoryTerminalVelocitySpread, 0.0, None, None))
        self.createFloaters(factoryPage, factoryWidgets)
        self.factoryNotebook = Pmw.NoteBook(factoryPage, tabpos=None)
        factoryPointPage = self.factoryNotebook.add('PointParticleFactory')
        zSpinPage = self.factoryNotebook.add('ZSpinParticleFactory')
        self.createCheckbutton(zSpinPage, 'Z Spin Factory', 'Enable Angular Velocity', 'On: angular velocity is used; Off: final angle is used', self.toggleAngularVelocity, 0, side=tk.TOP)
        self.createFloater(zSpinPage, 'Z Spin Factory', 'Angular Velocity', 'How fast sprites rotate', command=self.setFactoryZSpinAngularVelocity)
        self.createFloater(zSpinPage, 'Z Spin Factory', 'Angular Velocity Spread', 'Variation in how fast sprites rotate', command=self.setFactoryZSpinAngularVelocitySpread)
        self.createAngleDial(zSpinPage, 'Z Spin Factory', 'Initial Angle', 'Starting angle in degrees', fRollover=1, command=self.setFactoryZSpinInitialAngle)
        self.createAngleDial(zSpinPage, 'Z Spin Factory', 'Initial Angle Spread', 'Spread of the initial angle', fRollover=1, command=self.setFactoryZSpinInitialAngleSpread)
        self.createAngleDial(zSpinPage, 'Z Spin Factory', 'Final Angle', 'Final angle in degrees', fRollover=1, command=self.setFactoryZSpinFinalAngle)
        self.createAngleDial(zSpinPage, 'Z Spin Factory', 'Final Angle Spread', 'Spread of the final angle', fRollover=1, command=self.setFactoryZSpinFinalAngleSpread)
        orientedPage = self.factoryNotebook.add('OrientedParticleFactory')
        tk.Label(orientedPage, text='Not implemented').pack(expand=1, fill=tk.BOTH)
        self.factoryNotebook.pack(expand=1, fill=tk.BOTH)
        self.createOptionMenu(emitterPage, 'Emitter', 'Emitter Type', 'Select type of particle emitter', ('BoxEmitter', 'DiscEmitter', 'LineEmitter', 'PointEmitter', 'RectangleEmitter', 'RingEmitter', 'SphereVolumeEmitter', 'SphereSurfaceEmitter', 'TangentRingEmitter'), self.selectEmitterType)
        self.emissionType = tk.IntVar()
        self.emissionType.set(BaseParticleEmitter.ETRADIATE)
        emissionFrame = tk.Frame(emitterPage)
        self.createRadiobutton(emissionFrame, 'left', 'Emitter', 'Explicit Emission', 'particles are all emitted in parallel, direction is based ' + 'on explicit velocity vector', self.emissionType, BaseParticleEmitter.ETEXPLICIT, self.setEmissionType)
        self.createRadiobutton(emissionFrame, 'left', 'Emitter', 'Radiate Emission', 'particles are emitted away from a specific point', self.emissionType, BaseParticleEmitter.ETRADIATE, self.setEmissionType)
        self.createRadiobutton(emissionFrame, 'left', 'Emitter', 'Custom Emission', 'particles are emitted with a velocity that ' + 'is determined by the particular emitter', self.emissionType, BaseParticleEmitter.ETCUSTOM, self.setEmissionType)
        emissionFrame.pack(fill='x', expand=0)
        self.createFloater(emitterPage, 'Emitter', 'Velocity Multiplier', 'launch velocity multiplier (all emission modes)', command=self.setEmitterAmplitude, min=None)
        self.createFloater(emitterPage, 'Emitter', 'Velocity Multiplier Spread', 'spread for launch velocity multiplier (all emission modes)', command=self.setEmitterAmplitudeSpread)
        self.createVector3Entry(emitterPage, 'Emitter', 'Offset Velocity', 'Velocity vector applied to all particles', command=self.setEmitterOffsetForce)
        self.createVector3Entry(emitterPage, 'Emitter', 'Explicit Velocity', 'all particles launch with this velocity in Explicit mode', command=self.setEmitterExplicitLaunchVector)
        self.createVector3Entry(emitterPage, 'Emitter', 'Radiate Origin', 'particles launch away from this point in Radiate mode', command=self.setEmitterRadiateOrigin)
        self.emitterNotebook = Pmw.NoteBook(emitterPage, tabpos=None)
        boxPage = self.emitterNotebook.add('BoxEmitter')
        self.createVector3Entry(boxPage, 'Box Emitter', 'Min', 'Min point defining emitter box', command=self.setEmitterBoxPoint1)
        self.createVector3Entry(boxPage, 'Box Emitter', 'Max', 'Max point defining emitter box', command=self.setEmitterBoxPoint2, value=(1.0, 1.0, 1.0))
        discPage = self.emitterNotebook.add('DiscEmitter')
        self.createFloater(discPage, 'Disc Emitter', 'Radius', 'Radius of disc', command=self.setEmitterDiscRadius, min=0.01)
        customPage = self.discCustomFrame = tk.Frame(discPage)
        self.createAngleDial(customPage, 'Disc Emitter', 'Inner Angle', 'Particle launch angle at center of disc', command=self.setEmitterDiscInnerAngle)
        self.createFloater(customPage, 'Disc Emitter', 'Inner Velocity', 'Launch velocity multiplier at center of disc', command=self.setEmitterDiscInnerVelocity)
        self.createAngleDial(customPage, 'Disc Emitter', 'Outer Angle', 'Particle launch angle at outer edge of disc', command=self.setEmitterDiscOuterAngle)
        self.createFloater(customPage, 'Disc Emitter', 'Outer Velocity', 'Launch velocity multiplier at edge of disc', command=self.setEmitterDiscOuterVelocity)
        self.createCheckbutton(customPage, 'Disc Emitter', 'Cubic Lerping', 'On: magnitude/angle interpolation from center', self.toggleEmitterDiscCubicLerping, 0)
        customPage.pack(fill=tk.BOTH, expand=1)
        linePage = self.emitterNotebook.add('LineEmitter')
        self.createVector3Entry(linePage, 'Line Emitter', 'Min', 'Min point defining emitter line', command=self.setEmitterLinePoint1)
        self.createVector3Entry(linePage, 'Line Emitter', 'Max', 'Max point defining emitter line', command=self.setEmitterLinePoint2, value=(1.0, 0.0, 0.0))
        emitterPointPage = self.emitterNotebook.add('PointEmitter')
        self.createVector3Entry(emitterPointPage, 'Point Emitter', 'Position', 'Position of emitter point', command=self.setEmitterPointPosition)
        rectanglePage = self.emitterNotebook.add('RectangleEmitter')
        self.createVector2Entry(rectanglePage, 'Rectangle Emitter', 'Min', 'Point defining rectangle', command=self.setEmitterRectanglePoint1)
        self.createVector2Entry(rectanglePage, 'Rectangle Emitter', 'Max', 'Point defining rectangle', command=self.setEmitterRectanglePoint2)
        ringPage = self.emitterNotebook.add('RingEmitter')
        self.createFloater(ringPage, 'Ring Emitter', 'Radius', 'Radius of ring', command=self.setEmitterRingRadius, min=0.01)
        self.createFloater(ringPage, 'Ring Emitter', 'Radius Spread', 'Variation in radius of ring', command=self.setEmitterRingRadiusSpread, min=0.0)
        self.ringCustomFrame = tk.Frame(ringPage)
        self.createAngleDial(self.ringCustomFrame, 'Ring Emitter', 'Angle', 'Particle launch angle', command=self.setEmitterRingLaunchAngle)
        self.ringCustomFrame.pack(fill=tk.BOTH, expand=1)
        sphereVolumePage = self.emitterNotebook.add('SphereVolumeEmitter')
        self.createFloater(sphereVolumePage, 'Sphere Volume Emitter', 'Radius', 'Radius of sphere', command=self.setEmitterSphereVolumeRadius, min=0.01)
        sphereSurfacePage = self.emitterNotebook.add('SphereSurfaceEmitter')
        self.createFloater(sphereSurfacePage, 'Sphere Surface Emitter', 'Radius', 'Radius of sphere', command=self.setEmitterSphereSurfaceRadius, min=0.01)
        tangentRingPage = self.emitterNotebook.add('TangentRingEmitter')
        self.createFloater(tangentRingPage, 'Tangent Ring Emitter', 'Radius', 'Radius of ring', command=self.setEmitterTangentRingRadius, min=0.01)
        self.createFloater(tangentRingPage, 'Tangent Ring Emitter', 'Radius Spread', 'Variation in radius of ring', command=self.setEmitterTangentRingRadiusSpread)
        self.emitterNotebook.pack(fill=tk.X)
        self.createOptionMenu(rendererPage, 'Renderer', 'Renderer Type', 'Select type of particle renderer', ('LineParticleRenderer', 'GeomParticleRenderer', 'PointParticleRenderer', 'SparkleParticleRenderer', 'SpriteParticleRenderer'), self.selectRendererType)
        self.createOptionMenu(rendererPage, 'Renderer', 'Alpha Mode', "alpha setting over particles' lifetime", ('NO_ALPHA', 'ALPHA_IN', 'ALPHA_OUT', 'ALPHA_IN_OUT', 'ALPHA_USER'), self.setRendererAlphaMode)
        self.createSlider(rendererPage, 'Renderer', 'User Alpha', 'alpha value for ALPHA_USER alpha mode', command=self.setRendererUserAlpha)
        self.rendererNotebook = Pmw.NoteBook(rendererPage, tabpos=None)
        self.rendererNotebook.pack(fill=tk.BOTH, expand=1)
        linePage = self.rendererNotebook.add('LineParticleRenderer')
        self.createColorEntry(linePage, 'Line Renderer', 'Head Color', 'Head color of line', command=self.setRendererLineHeadColor)
        self.createColorEntry(linePage, 'Line Renderer', 'Tail Color', 'Tail color of line', command=self.setRendererLineTailColor)
        self.createFloater(linePage, 'Line Renderer', 'Line Scale Factor', 'Scale Factor applied to length of line', command=self.setRendererLineScaleFactor)
        geomPage = self.rendererNotebook.add('GeomParticleRenderer')
        f = tk.Frame(geomPage)
        f.pack(fill=tk.X)
        tk.Label(f, width=12, text='Geom Node', pady=3).pack(side=tk.LEFT)
        self.rendererGeomNode = tk.StringVar()
        self.rendererGeomNodeEntry = tk.Entry(f, width=12, textvariable=self.rendererGeomNode)
        self.rendererGeomNodeEntry.bind('<Return>', self.setRendererGeomNode)
        self.rendererGeomNodeEntry.pack(side=tk.LEFT, expand=1, fill=tk.X)
        f = tk.Frame(geomPage)
        f.pack(fill=tk.BOTH, expand=1)
        rendererGeomNotebook = Pmw.NoteBook(f)
        rendererGeomNotebook.pack(fill=tk.BOTH, expand=1)
        rendererGeomBlendPage = rendererGeomNotebook.add('Blend')
        rendererGeomScalePage = rendererGeomNotebook.add('Scale')
        rendererGeomInterpolationPage = rendererGeomNotebook.add('Interpolate')
        p = tk.Frame(rendererGeomBlendPage)
        p.pack(fill=tk.X)
        self.createOptionMenu(p, 'Geom Renderer', 'Color Blend', 'How to render semi-transparent colors', ('MNone', 'MAdd', 'MSubtract', 'MInvSubtract', 'MMin', 'MMax'), self.setRendererGeomColorBlendMethod)
        self.createOptionMenu(p, 'Geom Renderer', 'Incoming Op.', 'See ColorBlendAttrib.h for explanation', ('OOne', 'OIncomingColor', 'OOneMinusIncomingColor', 'OFbufferColor', 'OOneMinusFbufferColor', 'OIncomingAlpha', 'OOneMinusIncomingAlpha', 'OFbufferAlpha', 'OOneMinusFbufferAlpha', 'OConstantColor', 'OOneMinusConstantColor', 'OConstantAlpha', 'OOneMinusConstantAlpha', 'OIncomingColorSaturate', 'OZero'), self.setRendererGeomColorBlendIncomingOperand)
        self.getVariable('Geom Renderer', 'Incoming Op.').set('OIncomingAlpha')
        self.createOptionMenu(p, 'Geom Renderer', 'Fbuffer Op.', 'See ColorBlendAttrib.h for explanation', ('OOne', 'OIncomingColor', 'OOneMinusIncomingColor', 'OFbufferColor', 'OOneMinusFbufferColor', 'OIncomingAlpha', 'OOneMinusIncomingAlpha', 'OFbufferAlpha', 'OOneMinusFbufferAlpha', 'OConstantColor', 'OOneMinusConstantColor', 'OConstantAlpha', 'OOneMinusConstantAlpha', 'OZero'), self.setRendererGeomColorBlendFbufferOperand)
        self.getVariable('Geom Renderer', 'Fbuffer Op.').set('OOneMinusIncomingAlpha')
        p = tk.Frame(rendererGeomScalePage)
        p.pack(fill=tk.X)
        self.createCheckbutton(p, 'Geom Renderer', 'X Scale', "On: x scale is interpolated over particle's life; " + 'Off: stays as start_X_Scale', self.toggleRendererGeomXScale, 0, side=tk.LEFT)
        self.createCheckbutton(p, 'Geom Renderer', 'Y Scale', "On: y scale is interpolated over particle's life; " + 'Off: stays as start_Y_Scale', self.toggleRendererGeomYScale, 0, side=tk.LEFT)
        self.createCheckbutton(p, 'Geom Renderer', 'Z Scale', "On: z scale is interpolated over particle's life; " + 'Off: stays as start_Z_Scale', self.toggleRendererGeomZScale, 0, side=tk.LEFT)
        p = tk.Frame(rendererGeomScalePage)
        p.pack(fill=tk.X)
        self.createFloater(p, 'Geom Renderer', 'Initial X Scale', 'Initial X scaling factor', command=self.setRendererGeomInitialXScale)
        self.createFloater(p, 'Geom Renderer', 'Final X Scale', 'Final X scaling factor, if xScale enabled', command=self.setRendererGeomFinalXScale)
        self.createFloater(p, 'Geom Renderer', 'Initial Y Scale', 'Initial Y scaling factor', command=self.setRendererGeomInitialYScale)
        self.createFloater(p, 'Geom Renderer', 'Final Y Scale', 'Final Y scaling factor, if yScale enabled', command=self.setRendererGeomFinalYScale)
        self.createFloater(p, 'Geom Renderer', 'Initial Z Scale', 'Initial Z scaling factor', command=self.setRendererGeomInitialZScale)
        self.createFloater(p, 'Geom Renderer', 'Final Z Scale', 'Final Z scaling factor, if zScale enabled', command=self.setRendererGeomFinalZScale)
        p = tk.Frame(rendererGeomInterpolationPage)
        p.pack(fill=tk.X)
        addSegmentButton = tk.Menubutton(p, text='Add Segment', relief=tk.RAISED, borderwidth=2, font=('MSSansSerif', 14, 'bold'), activebackground='#909090')
        segmentMenu = tk.Menu(addSegmentButton)
        addSegmentButton['menu'] = segmentMenu
        segmentMenu.add_command(label='Add Constant segment', command=self.addConstantInterpolationSegment)
        segmentMenu.add_command(label='Add Linear segment', command=self.addLinearInterpolationSegment)
        segmentMenu.add_command(label='Add Stepwave segment', command=self.addStepwaveInterpolationSegment)
        segmentMenu.add_command(label='Add Sinusoid segment', command=self.addSinusoidInterpolationSegment)
        addSegmentButton.pack(expand=0)
        sf = Pmw.ScrolledFrame(p, horizflex='elastic')
        sf.pack(fill=tk.BOTH, expand=1)
        self.rendererGeomSegmentFrame = sf.interior()
        self.rendererGeomSegmentFrame.pack(fill=tk.BOTH, expand=1)
        self.rendererSegmentWidgetList = []
        rendererGeomNotebook.setnaturalsize()
        rendererPointPage = self.rendererNotebook.add('PointParticleRenderer')
        self.createFloater(rendererPointPage, 'Point Renderer', 'Point Size', 'Width and height of points in pixels', command=self.setRendererPointSize)
        self.createColorEntry(rendererPointPage, 'Point Renderer', 'Start Color', 'Starting color of point', command=self.setRendererPointStartColor)
        self.createColorEntry(rendererPointPage, 'Point Renderer', 'End Color', 'Ending color of point', command=self.setRendererPointEndColor)
        self.createOptionMenu(rendererPointPage, 'Point Renderer', 'Blend Type', 'Type of color blending used for particle', ('PP_ONE_COLOR', 'PP_BLEND_LIFE', 'PP_BLEND_VEL'), self.rendererPointSelectBlendType)
        self.createOptionMenu(rendererPointPage, 'Point Renderer', 'Blend Method', 'Interpolation method between colors', ('PP_NO_BLEND', 'PP_BLEND_LINEAR', 'PP_BLEND_CUBIC'), self.rendererPointSelectBlendMethod)
        sparklePage = self.rendererNotebook.add('SparkleParticleRenderer')
        self.createColorEntry(sparklePage, 'Sparkle Renderer', 'Center Color', 'Color of sparkle center', command=self.setRendererSparkleCenterColor)
        self.createColorEntry(sparklePage, 'Sparkle Renderer', 'Edge Color', 'Color of sparkle line endpoints', command=self.setRendererSparkleEdgeColor)
        self.createFloater(sparklePage, 'Sparkle Renderer', 'Birth Radius', 'Initial sparkle radius', command=self.setRendererSparkleBirthRadius)
        self.createFloater(sparklePage, 'Sparkle Renderer', 'Death Radius', 'Final sparkle radius', command=self.setRendererSparkleDeathRadius)
        self.createOptionMenu(sparklePage, 'Sparkle Renderer', 'Life Scale', 'Does particle scale over its lifetime?', ('SP_NO_SCALE', 'SP_SCALE'), self.setRendererSparkleLifeScale)
        spritePage = self.rendererNotebook.add('SpriteParticleRenderer')
        f = tk.Frame(spritePage)
        f.pack(fill=tk.BOTH, expand=1)
        rendererSpriteNotebook = Pmw.NoteBook(f)
        rendererSpriteNotebook.pack(fill=tk.BOTH, expand=1)
        rendererSpriteTexturePage = rendererSpriteNotebook.add('Texture')
        rendererSpriteScalePage = rendererSpriteNotebook.add('Scale')
        rendererSpriteBlendPage = rendererSpriteNotebook.add('Blend')
        rendererSpriteInterpolationPage = rendererSpriteNotebook.add('Interpolate')
        p = tk.Frame(rendererSpriteTexturePage)
        p.pack(fill=tk.BOTH, expand=1)
        bp = tk.Frame(p)
        bp.pack(expand=0, side=tk.TOP)
        bbp = tk.Frame(bp)
        bbp.pack()
        self.createCheckbutton(bbp, 'Sprite Renderer', 'Enable Animation', 'On: Multitexture node will be animated; ' + 'Off: Only the first frame of a node is rendered', self.setRendererSpriteAnimationEnable, 0, side=tk.LEFT)
        self.createFloater(bbp, 'Sprite Renderer', 'Frame Rate', 'Animation frame rate', command=self.setRendererSpriteAnimationFrameRate).pack(side=tk.LEFT)
        bbp = tk.Frame(bp)
        bbp.pack(pady=3)
        tk.Button(bbp, text='Add Texture', command=self.addRendererSpriteAnimationTexture).pack(pady=3, padx=15, side=tk.LEFT)
        tk.Button(bbp, text='Add Animation', command=self.addRendererSpriteAnimationFromNode).pack(pady=3, padx=15, side=tk.LEFT)
        pp = tk.Frame(p)
        pp.pack(fill=tk.BOTH, expand=1, pady=3)
        sf = Pmw.ScrolledFrame(pp, horizflex='elastic')
        sf.pack(fill=tk.BOTH, expand=1)
        self.rendererSpriteAnimationFrame = sf.interior()
        self.rendererSpriteAnimationFrame.pack(fill=tk.BOTH, expand=1)
        self.rendererSpriteAnimationWidgetList = []
        self.rendererSpriteTexture = tk.StringVar()
        self.rendererSpriteFile = tk.StringVar()
        self.rendererSpriteNode = tk.StringVar()
        p = tk.Frame(rendererSpriteScalePage)
        p.pack(fill=tk.X)
        self.createCheckbutton(p, 'Sprite Renderer', 'X Scale', "On: x scale is interpolated over particle's life; " + 'Off: stays as start_X_Scale', self.toggleRendererSpriteXScale, 0, side=tk.LEFT)
        self.createCheckbutton(p, 'Sprite Renderer', 'Y Scale', "On: y scale is interpolated over particle's life; " + 'Off: stays as start_Y_Scale', self.toggleRendererSpriteYScale, 0, side=tk.LEFT)
        self.createCheckbutton(p, 'Sprite Renderer', 'Anim Angle', 'On: particles that are set to spin on the Z axis will ' + 'spin appropriately', self.toggleRendererSpriteAnimAngle, 0, side=tk.LEFT)
        p = tk.Frame(rendererSpriteScalePage)
        p.pack(fill=tk.X)
        self.createFloater(p, 'Sprite Renderer', 'Initial X Scale', 'Initial X scaling factor', command=self.setRendererSpriteInitialXScale)
        self.createFloater(p, 'Sprite Renderer', 'Final X Scale', 'Final X scaling factor, if xScale enabled', command=self.setRendererSpriteFinalXScale)
        self.createFloater(p, 'Sprite Renderer', 'Initial Y Scale', 'Initial Y scaling factor', command=self.setRendererSpriteInitialYScale)
        self.createFloater(p, 'Sprite Renderer', 'Final Y Scale', 'Final Y scaling factor, if yScale enabled', command=self.setRendererSpriteFinalYScale)
        self.createAngleDial(p, 'Sprite Renderer', 'Non Animated Theta', 'If animAngle is false: counter clockwise ' + 'Z rotation of all sprites', command=self.setRendererSpriteNonAnimatedTheta)
        p = tk.Frame(rendererSpriteBlendPage)
        p.pack(fill=tk.X)
        self.createOptionMenu(p, 'Sprite Renderer', 'Blend Type', 'Interpolation blend type for X and Y scaling', ('PP_NO_BLEND', 'PP_LINEAR', 'PP_CUBIC'), self.setRendererSpriteBlendMethod)
        self.createCheckbutton(p, 'Sprite Renderer', 'Alpha Disable', 'On: alpha blending is disabled', self.toggleRendererSpriteAlphaDisable, 0)
        self.createOptionMenu(p, 'Sprite Renderer', 'Color Blend', 'How to render semi-transparent colors', ('MNone', 'MAdd', 'MSubtract', 'MInvSubtract', 'MMin', 'MMax'), self.setRendererSpriteColorBlendMethod)
        self.createOptionMenu(p, 'Sprite Renderer', 'Incoming Op.', 'See ColorBlendAttrib.h for explanation', ('OOne', 'OIncomingColor', 'OOneMinusIncomingColor', 'OFbufferColor', 'OOneMinusFbufferColor', 'OIncomingAlpha', 'OOneMinusIncomingAlpha', 'OFbufferAlpha', 'OOneMinusFbufferAlpha', 'OConstantColor', 'OOneMinusConstantColor', 'OConstantAlpha', 'OOneMinusConstantAlpha', 'OIncomingColorSaturate', 'OZero'), self.setRendererSpriteColorBlendIncomingOperand)
        self.getVariable('Sprite Renderer', 'Incoming Op.').set('OIncomingAlpha')
        self.createOptionMenu(p, 'Sprite Renderer', 'Fbuffer Op.', 'See ColorBlendAttrib.h for explanation', ('OOne', 'OIncomingColor', 'OOneMinusIncomingColor', 'OFbufferColor', 'OOneMinusFbufferColor', 'OIncomingAlpha', 'OOneMinusIncomingAlpha', 'OFbufferAlpha', 'OOneMinusFbufferAlpha', 'OConstantColor', 'OOneMinusConstantColor', 'OConstantAlpha', 'OOneMinusConstantAlpha', 'OZero'), self.setRendererSpriteColorBlendFbufferOperand)
        self.getVariable('Sprite Renderer', 'Fbuffer Op.').set('OOneMinusIncomingAlpha')
        p = tk.Frame(rendererSpriteInterpolationPage)
        p.pack(fill=tk.BOTH, expand=1)
        addSegmentButton = tk.Menubutton(p, text='Add Segment', relief=tk.RAISED, borderwidth=2, font=('MSSansSerif', 14, 'bold'), activebackground='#909090')
        segmentMenu = tk.Menu(addSegmentButton)
        addSegmentButton['menu'] = segmentMenu
        segmentMenu.add_command(label='Add Constant segment', command=self.addConstantInterpolationSegment)
        segmentMenu.add_command(label='Add Linear segment', command=self.addLinearInterpolationSegment)
        segmentMenu.add_command(label='Add Stepwave segment', command=self.addStepwaveInterpolationSegment)
        segmentMenu.add_command(label='Add Sinusoid segment', command=self.addSinusoidInterpolationSegment)
        addSegmentButton.pack(expand=0)
        pp = tk.Frame(p)
        pp.pack(fill=tk.BOTH, expand=1, pady=3)
        sf = Pmw.ScrolledFrame(pp, horizflex='elastic')
        sf.pack(fill=tk.BOTH, expand=1)
        self.rendererSpriteSegmentFrame = sf.interior()
        self.rendererSpriteSegmentFrame.pack(fill=tk.BOTH, expand=1)
        self.rendererSegmentWidgetList = []
        rendererSpriteNotebook.setnaturalsize()
        self.addForceButton = tk.Menubutton(forcePage, text='Add Force', relief=tk.RAISED, borderwidth=2, font=('MSSansSerif', 14, 'bold'), activebackground='#909090')
        forceMenu = tk.Menu(self.addForceButton)
        self.addForceButton['menu'] = forceMenu
        forceMenu.add_command(label='Add Linear Vector Force', command=self.addLinearVectorForce)
        forceMenu.add_command(label='Add Linear Noise Force', command=self.addLinearNoiseForce)
        forceMenu.add_command(label='Add Linear Jitter Force', command=self.addLinearJitterForce)
        forceMenu.add_command(label='Add Linear Friction Force', command=self.addLinearFrictionForce)
        forceMenu.add_command(label='Add Linear Cylinder Vortex Force', command=self.addLinearCylinderVortexForce)
        forceMenu.add_command(label='Add Linear Sink Force', command=self.addLinearSinkForce)
        forceMenu.add_command(label='Add Linear Source Force', command=self.addLinearSourceForce)
        "\n        # Avoid for now\n        forceMenu.add_command(label = 'Add Linear User Defined Force',\n                            command = self.addLinearUserDefinedForce)\n        "
        self.addForceButton.pack(expand=0)
        self.sf = Pmw.ScrolledFrame(forcePage, horizflex='elastic')
        self.sf.pack(fill='both', expand=1)
        self.forceFrame = self.sf.interior()
        self.forceGroupNotebook = Pmw.NoteBook(self.forceFrame, tabpos=None)
        self.forceGroupNotebook.pack(fill=tk.X)
        self.factoryNotebook.setnaturalsize()
        self.emitterNotebook.setnaturalsize()
        self.rendererNotebook.setnaturalsize()
        self.forceGroupNotebook.setnaturalsize()
        self.mainNotebook.setnaturalsize()
        self.initialiseoptions(ParticlePanel)

    def createCheckbutton(self, parent, category, text, balloonHelp, command, initialState, side='top'):
        if False:
            while True:
                i = 10
        bool = tk.BooleanVar()
        bool.set(initialState)
        widget = tk.Checkbutton(parent, text=text, anchor=tk.W, variable=bool)
        widget['command'] = command
        widget.pack(fill=tk.X, side=side)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        self.variableDict[category + '-' + text] = bool
        return widget

    def createRadiobutton(self, parent, side, category, text, balloonHelp, variable, value, command):
        if False:
            print('Hello World!')
        widget = tk.Radiobutton(parent, text=text, anchor=tk.W, variable=variable, value=value)
        widget['command'] = command
        widget.pack(side=side, fill=tk.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createFloaters(self, parent, widgetDefinitions):
        if False:
            i = 10
            return i + 15
        widgets = []
        for (category, label, balloonHelp, command, min, max, resolution) in widgetDefinitions:
            widgets.append(self.createFloater(parent, category, label, balloonHelp, command, min, max, resolution))
        return widgets

    def createFloater(self, parent, category, text, balloonHelp, command=None, min=0.0, max=None, resolution=None, numDigits=None, **kw):
        if False:
            print('Hello World!')
        kw['text'] = text
        kw['min'] = min
        if max is not None:
            kw['max'] = max
        kw['resolution'] = resolution
        if numDigits is None:
            if resolution is not None and int(resolution) == resolution and (min is None or int(min) == min) and (max is None or int(max) == max):
                numDigits = 0
            else:
                numDigits = 3
        kw['numDigits'] = numDigits
        widget = Floater.Floater(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tk.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createAngleDial(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        kw['text'] = text
        kw['style'] = 'mini'
        widget = Dial.AngleDial(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tk.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createSlider(self, parent, category, text, balloonHelp, command=None, min=0.0, max=1.0, resolution=0.001, **kw):
        if False:
            return 10
        kw['text'] = text
        kw['min'] = min
        kw['max'] = max
        kw['resolution'] = resolution
        widget = Slider.Slider(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tk.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createVector2Entry(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            return 10
        kw['text'] = text
        widget = VectorWidgets.Vector2Entry(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tk.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createVector3Entry(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            return 10
        kw['text'] = text
        widget = VectorWidgets.Vector3Entry(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tk.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createColorEntry(self, parent, category, text, balloonHelp, command=None, **kw):
        if False:
            while True:
                i = 10
        kw['text'] = text
        widget = VectorWidgets.ColorEntry(parent, **kw)
        widget['command'] = command
        widget.pack(fill=tk.X)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def createOptionMenu(self, parent, category, text, balloonHelp, items, command):
        if False:
            print('Hello World!')
        optionVar = tk.StringVar()
        if len(items) > 0:
            optionVar.set(items[0])
        widget = Pmw.OptionMenu(parent, labelpos=tk.W, label_text=text, label_width=12, menu_tearoff=1, menubutton_textvariable=optionVar, items=items)
        widget['command'] = command
        widget.pack(fill=tk.X)
        self.bind(widget.component('menubutton'), balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        self.variableDict[category + '-' + text] = optionVar
        return optionVar

    def createComboBox(self, parent, category, text, balloonHelp, items, command, history=0):
        if False:
            return 10
        widget = Pmw.ComboBox(parent, labelpos=tk.W, label_text=text, label_anchor='w', label_width=12, entry_width=16, history=history, scrolledlist_items=items)
        widget.configure(entryfield_entry_state='disabled')
        if len(items) > 0:
            widget.selectitem(items[0])
        widget['selectioncommand'] = command
        widget.pack(side='left', expand=0)
        self.bind(widget, balloonHelp)
        self.widgetDict[category + '-' + text] = widget
        return widget

    def updateMenusAndLabels(self):
        if False:
            return 10
        self.updateMenus()
        self.updateLabels()

    def updateLabels(self):
        if False:
            print('Hello World!')
        self.effectsLabel['text'] = self.particleEffect.getName()
        self.particlesLabel['text'] = self.particles.getName()
        if self.forceGroup is not None:
            self.forceGroupLabel['text'] = self.forceGroup.getName()
        else:
            self.forceGroupLabel['text'] = 'Force Group'

    def updateMenus(self):
        if False:
            print('Hello World!')
        self.updateEffectsMenus()
        self.updateParticlesMenus()
        self.updateForceGroupMenus()

    def updateEffectsMenus(self):
        if False:
            i = 10
            return i + 15
        self.effectsEnableMenu.delete(0, 'end')
        self.effectsLabelMenu.delete(5, 'end')
        self.effectsLabelMenu.add_separator()
        keys = sorted(self.effectsDict.keys())
        for name in keys:
            effect = self.effectsDict[name]
            self.effectsLabelMenu.add_command(label=effect.getName(), command=lambda s=self, e=effect: s.selectEffectNamed(e.getName()))
            effectActive = tk.IntVar()
            effectActive.set(effect.isEnabled())
            self.effectsEnableMenu.add_checkbutton(label=effect.getName(), variable=effectActive, command=lambda s=self, e=effect, v=effectActive: s.toggleEffect(e, v))

    def updateParticlesMenus(self):
        if False:
            for i in range(10):
                print('nop')
        self.particlesEnableMenu.delete(0, 'end')
        self.particlesLabelMenu.delete(2, 'end')
        self.particlesLabelMenu.add_separator()
        particles = self.particleEffect.getParticlesList()
        for name in sorted((x.getName() for x in particles)):
            particle = self.particleEffect.getParticlesNamed(name)
            self.particlesLabelMenu.add_command(label=name, command=lambda s=self, n=name: s.selectParticlesNamed(n))
            particleActive = tk.IntVar()
            particleActive.set(particle.isEnabled())
            self.particlesEnableMenu.add_checkbutton(label=name, variable=particleActive, command=lambda s=self, p=particle, v=particleActive: s.toggleParticles(p, v))

    def updateForceGroupMenus(self):
        if False:
            return 10
        self.forceGroupEnableMenu.delete(0, 'end')
        self.forceGroupLabelMenu.delete(2, 'end')
        self.forceGroupLabelMenu.add_separator()
        forceGroupList = self.particleEffect.getForceGroupList()
        for name in sorted((x.getName() for x in forceGroupList)):
            force = self.particleEffect.getForceGroupNamed(name)
            self.forceGroupLabelMenu.add_command(label=name, command=lambda s=self, n=name: s.selectForceGroupNamed(n))
            forceActive = tk.IntVar()
            forceActive.set(force.isEnabled())
            self.forceGroupEnableMenu.add_checkbutton(label=name, variable=forceActive, command=lambda s=self, f=force, v=forceActive: s.toggleForceGroup(f, v))

    def selectEffectNamed(self, name):
        if False:
            for i in range(10):
                print('nop')
        effect = self.effectsDict.get(name, None)
        if effect is not None:
            self.particleEffect = effect
            self.particles = self.particleEffect.getParticlesList()[0]
            forceGroupList = self.particleEffect.getForceGroupList()
            if len(forceGroupList) > 0:
                self.forceGroup = forceGroupList[0]
            else:
                self.forceGroup = None
            self.mainNotebook.selectpage('System')
            self.updateInfo('System')
        else:
            print('ParticlePanel: No effect named ' + name)

    def toggleEffect(self, effect, var):
        if False:
            for i in range(10):
                print('nop')
        if var.get():
            effect.enable()
        else:
            effect.disable()

    def selectParticlesNamed(self, name):
        if False:
            return 10
        particles = self.particleEffect.getParticlesNamed(name)
        if particles is not None:
            self.particles = particles
            self.updateInfo()

    def toggleParticles(self, particles, var):
        if False:
            print('Hello World!')
        if var.get():
            particles.enable()
        else:
            particles.disable()

    def selectForceGroupNamed(self, name):
        if False:
            return 10
        forceGroup = self.particleEffect.getForceGroupNamed(name)
        if forceGroup is not None:
            self.forceGroup = forceGroup
            self.updateInfo('Force')

    def toggleForceGroup(self, forceGroup, var):
        if False:
            print('Hello World!')
        if var.get():
            forceGroup.enable()
        else:
            forceGroup.disable()

    def toggleForce(self, force, pageName, variableName):
        if False:
            i = 10
            return i + 15
        v = self.getVariable(pageName, variableName)
        if v.get():
            force.setActive(1)
        else:
            force.setActive(0)

    def getWidget(self, category, text):
        if False:
            return 10
        return self.widgetDict[category + '-' + text]

    def getVariable(self, category, text):
        if False:
            print('Hello World!')
        return self.variableDict[category + '-' + text]

    def loadParticleEffectFromFile(self):
        if False:
            for i in range(10):
                print('nop')
        pPath = getParticlePath()
        if pPath.getNumDirectories() > 0:
            if repr(pPath.getDirectory(0)) == '.':
                path = '.'
            else:
                path = pPath.getDirectory(0).toOsSpecific()
        else:
            path = '.'
        if not os.path.isdir(path):
            print('ParticlePanel Warning: Invalid default DNA directory!')
            print('Using current directory')
            path = '.'
        particleFilename = askopenfilename(defaultextension='.ptf', filetypes=(('Particle Files', '*.ptf'), ('All files', '*')), initialdir=path, title='Load Particle Effect', parent=self.parent)
        if particleFilename and particleFilename != 'None':
            self.particleEffect.loadConfig(Filename.fromOsSpecific(particleFilename))
            self.selectEffectNamed(self.particleEffect.getName())
            self.particleEffect.enable()

    def saveParticleEffectToFile(self):
        if False:
            return 10
        pPath = getParticlePath()
        if pPath.getNumDirectories() > 0:
            if repr(pPath.getDirectory(0)) == '.':
                path = '.'
            else:
                path = pPath.getDirectory(0).toOsSpecific()
        else:
            path = '.'
        if not os.path.isdir(path):
            print('ParticlePanel Warning: Invalid default DNA directory!')
            print('Using current directory')
            path = '.'
        particleFilename = asksaveasfilename(defaultextension='.ptf', filetypes=(('Particle Files', '*.ptf'), ('All files', '*')), initialdir=path, title='Save Particle Effect as', parent=self.parent)
        if particleFilename:
            self.particleEffect.saveConfig(Filename(particleFilename))

    def toggleParticleMgr(self):
        if False:
            while True:
                i = 10
        if self.particleMgrActive.get():
            base.enableParticles()
        else:
            base.disableParticles()

    def updateInfo(self, page='System'):
        if False:
            while True:
                i = 10
        self.updateMenusAndLabels()
        if page == 'System':
            self.updateSystemWidgets()
        elif page == 'Factory':
            self.selectFactoryPage()
            self.updateFactoryWidgets()
        elif page == 'Emitter':
            self.selectEmitterPage()
            self.updateEmitterWidgets()
        elif page == 'Renderer':
            self.selectRendererPage()
            self.updateRendererWidgets()
        elif page == 'Force':
            self.updateForceWidgets()

    def toggleParticleEffect(self):
        if False:
            for i in range(10):
                print('nop')
        if self.getVariable('Effect', 'Active').get():
            self.particleEffect.enable()
        else:
            self.particleEffect.disable()

    def updateSystemWidgets(self):
        if False:
            print('Hello World!')
        poolSize = self.particles.getPoolSize()
        self.getWidget('System', 'Pool Size').set(int(poolSize), 0)
        birthRate = self.particles.getBirthRate()
        self.getWidget('System', 'Birth Rate').set(birthRate, 0)
        litterSize = self.particles.getLitterSize()
        self.getWidget('System', 'Litter Size').set(int(litterSize), 0)
        litterSpread = self.particles.getLitterSpread()
        self.getWidget('System', 'Litter Spread').set(litterSpread, 0)
        systemLifespan = self.particles.getSystemLifespan()
        self.getWidget('System', 'Lifespan').set(systemLifespan, 0)
        pos = self.particles.nodePath.getPos()
        self.getWidget('System', 'Pos').set([pos[0], pos[1], pos[2]], 0)
        hpr = self.particles.nodePath.getHpr()
        self.getWidget('System', 'Hpr').set([hpr[0], hpr[1], hpr[2]], 0)
        self.getVariable('System', 'Render Space Velocities').set(self.particles.getLocalVelocityFlag())
        self.getVariable('System', 'System Grows Older').set(self.particles.getSystemGrowsOlderFlag())

    def setSystemPoolSize(self, value):
        if False:
            return 10
        self.particles.setPoolSize(int(value))

    def setSystemBirthRate(self, value):
        if False:
            return 10
        self.particles.setBirthRate(value)

    def setSystemLitterSize(self, value):
        if False:
            i = 10
            return i + 15
        self.particles.setLitterSize(int(value))

    def setSystemLitterSpread(self, value):
        if False:
            i = 10
            return i + 15
        self.particles.setLitterSpread(int(value))

    def setSystemLifespan(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.particles.setSystemLifespan(value)

    def toggleSystemLocalVelocity(self):
        if False:
            i = 10
            return i + 15
        self.particles.setLocalVelocityFlag(self.getVariable('System', 'Render Space Velocities').get())

    def toggleSystemGrowsOlder(self):
        if False:
            print('Hello World!')
        self.particles.setSystemGrowsOlderFlag(self.getVariable('System', 'System Grows Older').get())

    def setSystemPos(self, pos):
        if False:
            for i in range(10):
                print('nop')
        self.particles.nodePath.setPos(Vec3(pos[0], pos[1], pos[2]))

    def setSystemHpr(self, pos):
        if False:
            print('Hello World!')
        self.particles.nodePath.setHpr(Vec3(pos[0], pos[1], pos[2]))

    def selectFactoryType(self, type):
        if False:
            i = 10
            return i + 15
        self.factoryNotebook.selectpage(type)
        self.particles.setFactory(type)
        self.updateFactoryWidgets()

    def selectFactoryPage(self):
        if False:
            while True:
                i = 10
        pass

    def updateFactoryWidgets(self):
        if False:
            for i in range(10):
                print('nop')
        factory = self.particles.factory
        lifespan = factory.getLifespanBase()
        self.getWidget('Factory', 'Life Span').set(lifespan, 0)
        lifespanSpread = factory.getLifespanSpread()
        self.getWidget('Factory', 'Life Span Spread').set(lifespanSpread, 0)
        mass = factory.getMassBase()
        self.getWidget('Factory', 'Mass').set(mass, 0)
        massSpread = factory.getMassSpread()
        self.getWidget('Factory', 'Mass Spread').set(massSpread, 0)
        terminalVelocity = factory.getTerminalVelocityBase()
        self.getWidget('Factory', 'Terminal Velocity').set(terminalVelocity, 0)
        terminalVelocitySpread = factory.getTerminalVelocitySpread()
        self.getWidget('Factory', 'Terminal Vel. Spread').set(terminalVelocitySpread, 0)

    def setFactoryLifeSpan(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.particles.factory.setLifespanBase(value)

    def setFactoryLifeSpanSpread(self, value):
        if False:
            while True:
                i = 10
        self.particles.factory.setLifespanSpread(value)

    def setFactoryParticleMass(self, value):
        if False:
            i = 10
            return i + 15
        self.particles.factory.setMassBase(value)

    def setFactoryParticleMassSpread(self, value):
        if False:
            while True:
                i = 10
        self.particles.factory.setMassSpread(value)

    def setFactoryTerminalVelocity(self, value):
        if False:
            while True:
                i = 10
        self.particles.factory.setTerminalVelocityBase(value)

    def setFactoryTerminalVelocitySpread(self, value):
        if False:
            print('Hello World!')
        self.particles.factory.setTerminalVelocitySpread(value)

    def setFactoryZSpinInitialAngle(self, angle):
        if False:
            return 10
        self.particles.factory.setInitialAngle(angle)

    def setFactoryZSpinInitialAngleSpread(self, spread):
        if False:
            i = 10
            return i + 15
        self.particles.factory.setInitialAngleSpread(spread)

    def setFactoryZSpinFinalAngle(self, angle):
        if False:
            while True:
                i = 10
        self.particles.factory.setFinalAngle(angle)

    def setFactoryZSpinFinalAngleSpread(self, spread):
        if False:
            while True:
                i = 10
        self.particles.factory.setFinalAngleSpread(spread)

    def setFactoryZSpinAngularVelocity(self, vel):
        if False:
            while True:
                i = 10
        self.particles.factory.setAngularVelocity(vel)

    def setFactoryZSpinAngularVelocitySpread(self, spread):
        if False:
            while True:
                i = 10
        self.particles.factory.setAngularVelocitySpread(spread)

    def selectEmitterType(self, type):
        if False:
            while True:
                i = 10
        self.emitterNotebook.selectpage(type)
        self.particles.setEmitter(type)
        self.updateEmitterWidgets()

    def selectEmitterPage(self):
        if False:
            print('Hello World!')
        type = self.particles.emitter.__class__.__name__
        self.emitterNotebook.selectpage(type)
        self.getVariable('Emitter', 'Emitter Type').set(type)

    def updateEmitterWidgets(self):
        if False:
            i = 10
            return i + 15
        emitter = self.particles.emitter
        self.setEmissionType(self.particles.emitter.getEmissionType())
        amp = emitter.getAmplitude()
        self.getWidget('Emitter', 'Velocity Multiplier').set(amp)
        spread = emitter.getAmplitudeSpread()
        self.getWidget('Emitter', 'Velocity Multiplier Spread').set(spread)
        vec = emitter.getOffsetForce()
        self.getWidget('Emitter', 'Offset Velocity').set([vec[0], vec[1], vec[2]], 0)
        vec = emitter.getRadiateOrigin()
        self.getWidget('Emitter', 'Radiate Origin').set([vec[0], vec[1], vec[2]], 0)
        vec = emitter.getExplicitLaunchVector()
        self.getWidget('Emitter', 'Explicit Velocity').set([vec[0], vec[1], vec[2]], 0)
        if isinstance(emitter, BoxEmitter):
            min = emitter.getMinBound()
            self.getWidget('Box Emitter', 'Min').set([min[0], min[1], min[2]], 0)
            max = emitter.getMaxBound()
            self.getWidget('Box Emitter', 'Max').set([max[0], max[1], max[2]], 0)
        elif isinstance(emitter, DiscEmitter):
            radius = emitter.getRadius()
            self.getWidget('Disc Emitter', 'Radius').set(radius, 0)
            innerAngle = emitter.getInnerAngle()
            self.getWidget('Disc Emitter', 'Inner Angle').set(innerAngle, 0)
            innerMagnitude = emitter.getInnerMagnitude()
            self.getWidget('Disc Emitter', 'Inner Velocity').set(innerMagnitude, 0)
            outerAngle = emitter.getOuterAngle()
            self.getWidget('Disc Emitter', 'Outer Angle').set(outerAngle, 0)
            outerMagnitude = emitter.getOuterMagnitude()
            self.getWidget('Disc Emitter', 'Inner Velocity').set(outerMagnitude, 0)
            cubicLerping = emitter.getCubicLerping()
            self.getVariable('Disc Emitter', 'Cubic Lerping').set(cubicLerping)
        elif isinstance(emitter, LineEmitter):
            min = emitter.getEndpoint1()
            self.getWidget('Line Emitter', 'Min').set([min[0], min[1], min[2]], 0)
            max = emitter.getEndpoint2()
            self.getWidget('Line Emitter', 'Max').set([max[0], max[1], max[2]], 0)
        elif isinstance(emitter, PointEmitter):
            location = emitter.getLocation()
            self.getWidget('Point Emitter', 'Position').set([location[0], location[1], location[2]], 0)
        elif isinstance(emitter, RectangleEmitter):
            min = emitter.getMinBound()
            self.getWidget('Rectangle Emitter', 'Min').set([min[0], min[1]], 0)
            max = emitter.getMaxBound()
            self.getWidget('Rectangle Emitter', 'Max').set([max[0], max[1]], 0)
        elif isinstance(emitter, RingEmitter):
            radius = emitter.getRadius()
            self.getWidget('Ring Emitter', 'Radius').set(radius, 0)
            radiusSpread = emitter.getRadiusSpread()
            self.getWidget('Ring Emitter', 'Radius Spread').set(radiusSpread, 0)
            angle = emitter.getAngle()
            self.getWidget('Ring Emitter', 'Angle').set(angle, 0)
        elif isinstance(emitter, SphereVolumeEmitter):
            radius = emitter.getRadius()
            self.getWidget('Sphere Volume Emitter', 'Radius').set(radius, 0)
        elif isinstance(emitter, SphereSurfaceEmitter):
            radius = emitter.getRadius()
            self.getWidget('Sphere Surface Emitter', 'Radius').set(radius, 0)
        elif isinstance(emitter, TangentRingEmitter):
            radius = emitter.getRadius()
            self.getWidget('Tangent Ring Emitter', 'Radius').set(radius, 0)
            radiusSpread = emitter.getRadiusSpread()
            self.getWidget('Tangent Ring Emitter', 'Radius Spread').set(radiusSpread, 0)

    def setEmissionType(self, newType=None):
        if False:
            for i in range(10):
                print('nop')
        if newType:
            type = newType
            self.emissionType.set(type)
        else:
            type = self.emissionType.get()
        self.particles.emitter.setEmissionType(type)
        if type == BaseParticleEmitter.ETEXPLICIT:
            self.getWidget('Emitter', 'Radiate Origin')['state'] = 'disabled'
            self.getWidget('Emitter', 'Explicit Velocity')['state'] = 'normal'
            if isinstance(self.particles.emitter, DiscEmitter):
                self.discCustomFrame.pack_forget()
            elif isinstance(self.particles.emitter, RingEmitter):
                self.ringCustomFrame.pack_forget()
        elif type == BaseParticleEmitter.ETRADIATE:
            self.getWidget('Emitter', 'Radiate Origin')['state'] = 'normal'
            self.getWidget('Emitter', 'Explicit Velocity')['state'] = 'disabled'
            if isinstance(self.particles.emitter, DiscEmitter):
                self.discCustomFrame.pack_forget()
            elif isinstance(self.particles.emitter, RingEmitter):
                self.ringCustomFrame.pack_forget()
        elif type == BaseParticleEmitter.ETCUSTOM:
            self.getWidget('Emitter', 'Radiate Origin')['state'] = 'disabled'
            self.getWidget('Emitter', 'Explicit Velocity')['state'] = 'disabled'
            if isinstance(self.particles.emitter, DiscEmitter):
                self.discCustomFrame.pack(fill=tk.BOTH, expand=1)
            elif isinstance(self.particles.emitter, RingEmitter):
                self.ringCustomFrame.pack(fill=tk.BOTH, expand=1)

    def setEmitterAmplitude(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.particles.emitter.setAmplitude(value)

    def setEmitterAmplitudeSpread(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.particles.emitter.setAmplitudeSpread(value)

    def setEmitterOffsetForce(self, vec):
        if False:
            for i in range(10):
                print('nop')
        self.particles.emitter.setOffsetForce(Vec3(vec[0], vec[1], vec[2]))

    def setEmitterRadiateOrigin(self, origin):
        if False:
            return 10
        self.particles.emitter.setRadiateOrigin(Point3(origin[0], origin[1], origin[2]))

    def setEmitterExplicitLaunchVector(self, vec):
        if False:
            i = 10
            return i + 15
        self.particles.emitter.setExplicitLaunchVector(Vec3(vec[0], vec[1], vec[2]))

    def setEmitterBoxPoint1(self, point):
        if False:
            while True:
                i = 10
        self.particles.emitter.setMinBound(Point3(point[0], point[1], point[2]))

    def setEmitterBoxPoint2(self, point):
        if False:
            return 10
        self.particles.emitter.setMaxBound(Point3(point[0], point[1], point[2]))

    def setEmitterDiscRadius(self, radius):
        if False:
            for i in range(10):
                print('nop')
        self.particles.emitter.setRadius(radius)

    def setEmitterDiscInnerAngle(self, angle):
        if False:
            while True:
                i = 10
        self.particles.emitter.setInnerAngle(angle)

    def setEmitterDiscInnerVelocity(self, velocity):
        if False:
            for i in range(10):
                print('nop')
        self.particles.emitter.setInnerMagnitude(velocity)

    def setEmitterDiscOuterAngle(self, angle):
        if False:
            print('Hello World!')
        self.particles.emitter.setOuterAngle(angle)

    def setEmitterDiscOuterVelocity(self, velocity):
        if False:
            return 10
        self.particles.emitter.setOuterMagnitude(velocity)

    def toggleEmitterDiscCubicLerping(self):
        if False:
            print('Hello World!')
        self.particles.emitter.setCubicLerping(self.getVariable('Disc Emitter', 'Cubic Lerping').get())

    def setEmitterLinePoint1(self, point):
        if False:
            print('Hello World!')
        self.particles.emitter.setEndpoint1(Point3(point[0], point[1], point[2]))

    def setEmitterLinePoint2(self, point):
        if False:
            while True:
                i = 10
        self.particles.emitter.setEndpoint2(Point3(point[0], point[1], point[2]))

    def setEmitterPointPosition(self, pos):
        if False:
            while True:
                i = 10
        self.particles.emitter.setLocation(Point3(pos[0], pos[1], pos[2]))

    def setEmitterRectanglePoint1(self, point):
        if False:
            for i in range(10):
                print('nop')
        self.particles.emitter.setMinBound(Point2(point[0], point[1]))

    def setEmitterRectanglePoint2(self, point):
        if False:
            return 10
        self.particles.emitter.setMaxBound(Point2(point[0], point[1]))

    def setEmitterRingRadius(self, radius):
        if False:
            for i in range(10):
                print('nop')
        self.particles.emitter.setRadius(radius)

    def setEmitterRingRadiusSpread(self, radiusSpread):
        if False:
            print('Hello World!')
        self.particles.emitter.setRadiusSpread(radiusSpread)

    def setEmitterRingLaunchAngle(self, angle):
        if False:
            print('Hello World!')
        self.particles.emitter.setAngle(angle)

    def setEmitterSphereSurfaceRadius(self, radius):
        if False:
            i = 10
            return i + 15
        self.particles.emitter.setRadius(radius)

    def setEmitterSphereVolumeRadius(self, radius):
        if False:
            return 10
        self.particles.emitter.setRadius(radius)

    def setEmitterTangentRingRadius(self, radius):
        if False:
            while True:
                i = 10
        self.particles.emitter.setRadius(radius)

    def setEmitterTangentRingRadiusSpread(self, radiusSpread):
        if False:
            print('Hello World!')
        self.particles.emitter.setRadiusSpread(radiusSpread)

    def selectRendererType(self, type):
        if False:
            print('Hello World!')
        self.rendererNotebook.selectpage(type)
        self.particles.setRenderer(type)
        self.updateRendererWidgets()

    def updateRendererWidgets(self):
        if False:
            return 10
        renderer = self.particles.renderer
        alphaMode = renderer.getAlphaMode()
        if alphaMode == BaseParticleRenderer.PRALPHANONE:
            aMode = 'NO_ALPHA'
        elif alphaMode == BaseParticleRenderer.PRALPHAOUT:
            aMode = 'ALPHA_OUT'
        elif alphaMode == BaseParticleRenderer.PRALPHAIN:
            aMode = 'ALPHA_IN'
        elif alphaMode == BaseParticleRenderer.PRALPHAINOUT:
            aMode = 'ALPHA_IN_OUT'
        elif alphaMode == BaseParticleRenderer.PRALPHAUSER:
            aMode = 'ALPHA_USER'
        self.getVariable('Renderer', 'Alpha Mode').set(aMode)
        userAlpha = renderer.getUserAlpha()
        self.getWidget('Renderer', 'User Alpha').set(userAlpha)
        if isinstance(renderer, LineParticleRenderer):
            headColor = renderer.getHeadColor() * 255.0
            self.getWidget('Line Renderer', 'Head Color').set([headColor[0], headColor[1], headColor[2], headColor[3]])
            tailColor = renderer.getTailColor() * 255.0
            self.getWidget('Line Renderer', 'Tail Color').set([tailColor[0], tailColor[1], tailColor[2], tailColor[3]])
            self.getWidget('Line Renderer', 'Line Scale Factor').set(renderer.getLineScaleFactor())
        elif isinstance(renderer, GeomParticleRenderer):
            self.getVariable('Geom Renderer', 'X Scale').set(renderer.getXScaleFlag())
            self.getVariable('Geom Renderer', 'Y Scale').set(renderer.getYScaleFlag())
            self.getVariable('Geom Renderer', 'Z Scale').set(renderer.getZScaleFlag())
            initialXScale = renderer.getInitialXScale()
            self.getWidget('Geom Renderer', 'Initial X Scale').set(initialXScale)
            initialYScale = renderer.getInitialYScale()
            self.getWidget('Geom Renderer', 'Initial Y Scale').set(initialYScale)
            initialZScale = renderer.getInitialZScale()
            self.getWidget('Geom Renderer', 'Initial Z Scale').set(initialZScale)
            finalXScale = renderer.getFinalXScale()
            self.getWidget('Geom Renderer', 'Final X Scale').set(finalXScale)
            finalYScale = renderer.getFinalYScale()
            self.getWidget('Geom Renderer', 'Final Y Scale').set(finalYScale)
            finalZScale = renderer.getFinalZScale()
            self.getWidget('Geom Renderer', 'Final Z Scale').set(finalZScale)
            if self.getVariable('Geom Renderer', 'Color Blend').get() in ['MAdd', 'MSubtract', 'MInvSubtract']:
                self.getWidget('Geom Renderer', 'Incoming Op.').pack(fill=tk.X)
                self.getWidget('Geom Renderer', 'Fbuffer Op.').pack(fill=tk.X)
            else:
                self.getWidget('Geom Renderer', 'Incoming Op.').pack_forget()
                self.getWidget('Geom Renderer', 'Fbuffer Op.').pack_forget()
            for x in self.rendererSegmentWidgetList:
                x.pack_forget()
                x.destroy()
            self.rendererSegmentWidgetList = []
            for id in self.particles.renderer.getColorInterpolationManager().getSegmentIdList().split():
                self.createWidgetForExistingInterpolationSegment(eval(id))
        elif isinstance(renderer, PointParticleRenderer):
            pointSize = renderer.getPointSize()
            self.getWidget('Point Renderer', 'Point Size').set(pointSize)
            startColor = renderer.getStartColor() * 255.0
            self.getWidget('Point Renderer', 'Start Color').set([startColor[0], startColor[1], startColor[2], startColor[3]])
            endColor = renderer.getEndColor() * 255.0
            self.getWidget('Point Renderer', 'End Color').set([endColor[0], endColor[1], endColor[2], endColor[3]])
            blendType = renderer.getBlendType()
            if blendType == PointParticleRenderer.PPONECOLOR:
                bType = 'PP_ONE_COLOR'
            elif blendType == PointParticleRenderer.PPBLENDLIFE:
                bType = 'PP_BLEND_LIFE'
            elif blendType == PointParticleRenderer.PPBLENDVEL:
                bType = 'PP_BLEND_VEL'
            self.getVariable('Point Renderer', 'Blend Type').set(bType)
            blendMethod = renderer.getBlendMethod()
            bMethod = 'PP_NO_BLEND'
            if blendMethod == BaseParticleRenderer.PPNOBLEND:
                bMethod = 'PP_NO_BLEND'
            elif blendMethod == BaseParticleRenderer.PPBLENDLINEAR:
                bMethod = 'PP_BLEND_LINEAR'
            elif blendMethod == BaseParticleRenderer.PPBLENDCUBIC:
                bMethod = 'PP_BLEND_CUBIC'
            self.getVariable('Point Renderer', 'Blend Method').set(bMethod)
        elif isinstance(renderer, SparkleParticleRenderer):
            centerColor = renderer.getCenterColor() * 255.0
            self.getWidget('Sparkle Renderer', 'Center Color').set([centerColor[0], centerColor[1], centerColor[2], centerColor[3]])
            edgeColor = renderer.getEdgeColor() * 255.0
            self.getWidget('Sparkle Renderer', 'Edge Color').set([edgeColor[0], edgeColor[1], edgeColor[2], edgeColor[3]])
            birthRadius = renderer.getBirthRadius()
            self.getWidget('Sparkle Renderer', 'Birth Radius').set(birthRadius)
            deathRadius = renderer.getDeathRadius()
            self.getWidget('Sparkle Renderer', 'Death Radius').set(deathRadius)
            lifeScale = renderer.getLifeScale()
            lScale = 'SP_NO_SCALE'
            if lifeScale == SparkleParticleRenderer.SPSCALE:
                lScale = 'SP_SCALE'
            self.getVariable('Sparkle Renderer', 'Life Scale').set(lScale)
        elif isinstance(renderer, SpriteParticleRenderer):
            self.getWidget('Sprite Renderer', 'Frame Rate').set(renderer.getAnimateFramesRate(), 0)
            self.getVariable('Sprite Renderer', 'Enable Animation').set(renderer.getAnimateFramesEnable())
            self.readSpriteRendererAnimations()
            self.getVariable('Sprite Renderer', 'X Scale').set(renderer.getXScaleFlag())
            self.getVariable('Sprite Renderer', 'Y Scale').set(renderer.getYScaleFlag())
            self.getVariable('Sprite Renderer', 'Anim Angle').set(renderer.getAnimAngleFlag())
            initialXScale = renderer.getInitialXScale()
            self.getWidget('Sprite Renderer', 'Initial X Scale').set(initialXScale)
            initialYScale = renderer.getInitialYScale()
            self.getWidget('Sprite Renderer', 'Initial Y Scale').set(initialYScale)
            finalXScale = renderer.getFinalXScale()
            self.getWidget('Sprite Renderer', 'Final X Scale').set(finalXScale)
            finalYScale = renderer.getFinalYScale()
            self.getWidget('Sprite Renderer', 'Final Y Scale').set(finalYScale)
            nonanimatedTheta = renderer.getNonanimatedTheta()
            self.getWidget('Sprite Renderer', 'Non Animated Theta').set(nonanimatedTheta)
            blendMethod = renderer.getAlphaBlendMethod()
            bMethod = 'PP_NO_BLEND'
            if blendMethod == BaseParticleRenderer.PPNOBLEND:
                bMethod = 'PP_NO_BLEND'
            elif blendMethod == BaseParticleRenderer.PPBLENDLINEAR:
                bMethod = 'PP_BLEND_LINEAR'
            elif blendMethod == BaseParticleRenderer.PPBLENDCUBIC:
                bMethod = 'PP_BLEND_CUBIC'
            self.getVariable('Sprite Renderer', 'Alpha Disable').set(renderer.getAlphaDisable())
            if self.getVariable('Sprite Renderer', 'Color Blend').get() in ['MAdd', 'MSubtract', 'MInvSubtract']:
                self.getWidget('Sprite Renderer', 'Incoming Op.').pack(fill=tk.X)
                self.getWidget('Sprite Renderer', 'Fbuffer Op.').pack(fill=tk.X)
            else:
                self.getWidget('Sprite Renderer', 'Incoming Op.').pack_forget()
                self.getWidget('Sprite Renderer', 'Fbuffer Op.').pack_forget()
            for x in self.rendererSegmentWidgetList:
                x.pack_forget()
                x.destroy()
            self.rendererSegmentWidgetList = []
            for id in self.particles.renderer.getColorInterpolationManager().getSegmentIdList().split():
                self.createWidgetForExistingInterpolationSegment(eval(id))

    def selectRendererPage(self):
        if False:
            while True:
                i = 10
        type = self.particles.renderer.__class__.__name__
        if type == 'SpriteParticleRendererExt':
            type = 'SpriteParticleRenderer'
        self.rendererNotebook.selectpage(type)
        self.getVariable('Renderer', 'Renderer Type').set(type)

    def setRendererAlphaMode(self, alphaMode):
        if False:
            i = 10
            return i + 15
        if alphaMode == 'NO_ALPHA':
            aMode = BaseParticleRenderer.PRALPHANONE
        elif alphaMode == 'ALPHA_OUT':
            aMode = BaseParticleRenderer.PRALPHAOUT
        elif alphaMode == 'ALPHA_IN':
            aMode = BaseParticleRenderer.PRALPHAIN
        elif alphaMode == 'ALPHA_IN_OUT':
            aMode = BaseParticleRenderer.PRALPHAINOUT
        elif alphaMode == 'ALPHA_USER':
            aMode = BaseParticleRenderer.PRALPHAUSER
        self.particles.renderer.setAlphaMode(aMode)

    def setRendererUserAlpha(self, alpha):
        if False:
            for i in range(10):
                print('nop')
        self.particles.renderer.setUserAlpha(alpha)

    def setRendererLineHeadColor(self, color):
        if False:
            for i in range(10):
                print('nop')
        self.particles.renderer.setHeadColor(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

    def setRendererLineTailColor(self, color):
        if False:
            for i in range(10):
                print('nop')
        self.particles.renderer.setTailColor(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

    def setRendererLineScaleFactor(self, sf):
        if False:
            print('Hello World!')
        self.particles.renderer.setLineScaleFactor(sf)

    def setRendererGeomNode(self, event):
        if False:
            for i in range(10):
                print('nop')
        node = None
        nodePath = base.loader.loadModel(self.rendererGeomNode.get())
        if nodePath is not None:
            node = nodePath.node()
        if node is not None:
            self.particles.geomReference = self.rendererGeomNode.get()
            self.particles.renderer.setGeomNode(node)

    def setRendererPointSize(self, size):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setPointSize(size)

    def setRendererPointStartColor(self, color):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setStartColor(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

    def setRendererPointEndColor(self, color):
        if False:
            return 10
        self.particles.renderer.setEndColor(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

    def rendererPointSelectBlendType(self, blendType):
        if False:
            print('Hello World!')
        if blendType == 'PP_ONE_COLOR':
            bType = PointParticleRenderer.PPONECOLOR
        elif blendType == 'PP_BLEND_LIFE':
            bType = PointParticleRenderer.PPBLENDLIFE
        elif blendType == 'PP_BLEND_VEL':
            bType = PointParticleRenderer.PPBLENDVEL
        self.particles.renderer.setBlendType(bType)

    def rendererPointSelectBlendMethod(self, blendMethod):
        if False:
            print('Hello World!')
        if blendMethod == 'PP_NO_BLEND':
            bMethod = BaseParticleRenderer.PPNOBLEND
        elif blendMethod == 'PP_BLEND_LINEAR':
            bMethod = BaseParticleRenderer.PPBLENDLINEAR
        elif blendMethod == 'PP_BLEND_CUBIC':
            bMethod = BaseParticleRenderer.PPBLENDCUBIC
        self.particles.renderer.setBlendMethod(bMethod)

    def setRendererSparkleCenterColor(self, color):
        if False:
            print('Hello World!')
        self.particles.renderer.setCenterColor(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

    def setRendererSparkleEdgeColor(self, color):
        if False:
            return 10
        self.particles.renderer.setEdgeColor(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

    def setRendererSparkleBirthRadius(self, radius):
        if False:
            for i in range(10):
                print('nop')
        self.particles.renderer.setBirthRadius(radius)

    def setRendererSparkleDeathRadius(self, radius):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setDeathRadius(radius)

    def setRendererSparkleLifeScale(self, lifeScaleMethod):
        if False:
            print('Hello World!')
        if lifeScaleMethod == 'SP_NO_SCALE':
            lScale = SparkleParticleRenderer.SPNOSCALE
        else:
            lScale = SparkleParticleRenderer.SPSCALE
        self.particles.renderer.setLifeScale(lScale)

    def setSpriteSourceType(self):
        if False:
            while True:
                i = 10
        if self.rendererSpriteSourceType.get() == 0:
            self.rendererSpriteTextureEntry['state'] = 'normal'
            self.rendererSpriteFileEntry['state'] = 'disabled'
            self.rendererSpriteNodeEntry['state'] = 'disabled'
            self.rendererSpriteTextureEntry['background'] = '#FFFFFF'
            self.rendererSpriteFileEntry['background'] = '#C0C0C0'
            self.rendererSpriteNodeEntry['background'] = '#C0C0C0'
        else:
            self.rendererSpriteTextureEntry['state'] = 'disabled'
            self.rendererSpriteFileEntry['state'] = 'normal'
            self.rendererSpriteNodeEntry['state'] = 'normal'
            self.rendererSpriteTextureEntry['background'] = '#C0C0C0'
            self.rendererSpriteFileEntry['background'] = '#FFFFFF'
            self.rendererSpriteNodeEntry['background'] = '#FFFFFF'

    def setRendererSpriteAnimationFrameRate(self, rate):
        if False:
            while True:
                i = 10
        self.particles.renderer.setAnimateFramesRate(rate)

    def setRendererSpriteAnimationEnable(self):
        if False:
            print('Hello World!')
        self.particles.renderer.setAnimateFramesEnable(self.getVariable('Sprite Renderer', 'Enable Animation').get())

    def addRendererSpriteAnimationTexture(self):
        if False:
            print('Hello World!')
        ren = self.particles.getRenderer()
        parent = self.rendererSpriteAnimationFrame
        if ren.addTextureFromFile():
            animId = len([x for x in self.rendererSpriteAnimationWidgetList if x and x.valid])
            anim = ren.getAnim(animId)
            frameNum = len([x for x in self.rendererSpriteAnimationWidgetList if x])
            self.rendererSpriteAnimationWidgetList.append(self.createSpriteAnimationTextureWidget(parent, anim, repr(frameNum)))
        else:
            animId = len([x for x in self.rendererSpriteAnimationWidgetList if x and x.valid])
            anim = SpriteAnim.STTexture
            frameNum = len([x for x in self.rendererSpriteAnimationWidgetList if x])
            self.rendererSpriteAnimationWidgetList.append(self.createSpriteAnimationTextureWidget(parent, anim, repr(frameNum)))
        parent.pack(fill=tk.BOTH, expand=1)

    def addRendererSpriteAnimationFromNode(self):
        if False:
            for i in range(10):
                print('nop')
        ren = self.particles.getRenderer()
        parent = self.rendererSpriteAnimationFrame
        if ren.addTextureFromNode():
            animId = len([x for x in self.rendererSpriteAnimationWidgetList if x and x.valid])
            anim = ren.getAnim(animId)
            frameNum = len([x for x in self.rendererSpriteAnimationWidgetList if x])
            self.rendererSpriteAnimationWidgetList.append(self.createSpriteAnimationNodeWidget(parent, anim, repr(frameNum)))
        else:
            animId = len([x for x in self.rendererSpriteAnimationWidgetList if x and x.valid])
            anim = SpriteAnim.STFromNode
            frameNum = len([x for x in self.rendererSpriteAnimationWidgetList if x])
            self.rendererSpriteAnimationWidgetList.append(self.createSpriteAnimationNodeWidget(parent, anim, repr(frameNum)))
        parent.pack(fill=tk.BOTH, expand=1)

    def toggleRendererSpriteXScale(self):
        if False:
            print('Hello World!')
        self.particles.renderer.setXScaleFlag(self.getVariable('Sprite Renderer', 'X Scale').get())

    def toggleRendererSpriteYScale(self):
        if False:
            return 10
        self.particles.renderer.setYScaleFlag(self.getVariable('Sprite Renderer', 'Y Scale').get())

    def toggleRendererSpriteAnimAngle(self):
        if False:
            for i in range(10):
                print('nop')
        self.particles.renderer.setAnimAngleFlag(self.getVariable('Sprite Renderer', 'Anim Angle').get())

    def toggleAngularVelocity(self):
        if False:
            while True:
                i = 10
        self.particles.factory.enableAngularVelocity(self.getVariable('Z Spin Factory', 'Enable Angular Velocity').get())

    def setRendererSpriteInitialXScale(self, xScale):
        if False:
            return 10
        self.particles.renderer.setInitialXScale(xScale)

    def setRendererSpriteFinalXScale(self, xScale):
        if False:
            return 10
        self.particles.renderer.setFinalXScale(xScale)

    def setRendererSpriteInitialYScale(self, yScale):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setInitialYScale(yScale)

    def setRendererSpriteFinalYScale(self, yScale):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setFinalYScale(yScale)

    def setRendererSpriteNonAnimatedTheta(self, theta):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setNonanimatedTheta(theta)

    def setRendererSpriteBlendMethod(self, blendMethod):
        if False:
            i = 10
            return i + 15
        if blendMethod == 'PP_NO_BLEND':
            bMethod = BaseParticleRenderer.PPNOBLEND
        elif blendMethod == 'PP_BLEND_LINEAR':
            bMethod = BaseParticleRenderer.PPBLENDLINEAR
        elif blendMethod == 'PP_BLEND_CUBIC':
            bMethod = BaseParticleRenderer.PPBLENDCUBIC
        else:
            bMethod = BaseParticleRenderer.PPNOBLEND

    def toggleRendererSpriteAlphaDisable(self):
        if False:
            while True:
                i = 10
        self.particles.renderer.setAlphaDisable(self.getVariable('Sprite Renderer', 'Alpha Disable').get())

    def setRendererColorBlendAttrib(self, rendererName, blendMethodStr, incomingOperandStr, fbufferOperandStr):
        if False:
            while True:
                i = 10
        self.particles.getRenderer().setColorBlendMode(getattr(ColorBlendAttrib, blendMethodStr), getattr(ColorBlendAttrib, incomingOperandStr), getattr(ColorBlendAttrib, fbufferOperandStr))
        if blendMethodStr in ['MAdd', 'MSubtract', 'MInvSubtract']:
            self.getWidget(rendererName, 'Incoming Op.').pack(fill=tk.X)
            self.getWidget(rendererName, 'Fbuffer Op.').pack(fill=tk.X)
        else:
            self.getWidget(rendererName, 'Incoming Op.').pack_forget()
            self.getWidget(rendererName, 'Fbuffer Op.').pack_forget()
        self.updateRendererWidgets()

    def setRendererSpriteColorBlendMethod(self, blendMethod):
        if False:
            for i in range(10):
                print('nop')
        blendMethodStr = blendMethod
        incomingOperandStr = self.getVariable('Sprite Renderer', 'Incoming Op.').get()
        fbufferOperandStr = self.getVariable('Sprite Renderer', 'Fbuffer Op.').get()
        self.setRendererColorBlendAttrib('Sprite Renderer', blendMethodStr, incomingOperandStr, fbufferOperandStr)

    def setRendererSpriteColorBlendIncomingOperand(self, operand):
        if False:
            for i in range(10):
                print('nop')
        blendMethodStr = self.getVariable('Sprite Renderer', 'Color Blend').get()
        incomingOperandStr = operand
        fbufferOperandStr = self.getVariable('Sprite Renderer', 'Fbuffer Op.').get()
        self.setRendererColorBlendAttrib('Sprite Renderer', blendMethodStr, incomingOperandStr, fbufferOperandStr)

    def setRendererSpriteColorBlendFbufferOperand(self, operand):
        if False:
            i = 10
            return i + 15
        blendMethodStr = self.getVariable('Sprite Renderer', 'Color Blend').get()
        incomingOperandStr = self.getVariable('Sprite Renderer', 'Incoming Op.').get()
        fbufferOperandStr = operand
        self.setRendererColorBlendAttrib('Sprite Renderer', blendMethodStr, incomingOperandStr, fbufferOperandStr)

    def toggleRendererGeomXScale(self):
        if False:
            return 10
        self.particles.renderer.setXScaleFlag(self.getVariable('Geom Renderer', 'X Scale').get())

    def toggleRendererGeomYScale(self):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setYScaleFlag(self.getVariable('Geom Renderer', 'Y Scale').get())

    def toggleRendererGeomZScale(self):
        if False:
            for i in range(10):
                print('nop')
        self.particles.renderer.setZScaleFlag(self.getVariable('Geom Renderer', 'Z Scale').get())

    def setRendererGeomInitialXScale(self, xScale):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setInitialXScale(xScale)

    def setRendererGeomFinalXScale(self, xScale):
        if False:
            return 10
        self.particles.renderer.setFinalXScale(xScale)

    def setRendererGeomInitialYScale(self, yScale):
        if False:
            while True:
                i = 10
        self.particles.renderer.setInitialYScale(yScale)

    def setRendererGeomFinalYScale(self, yScale):
        if False:
            while True:
                i = 10
        self.particles.renderer.setFinalYScale(yScale)

    def setRendererGeomInitialZScale(self, zScale):
        if False:
            print('Hello World!')
        self.particles.renderer.setInitialZScale(zScale)

    def setRendererGeomFinalZScale(self, zScale):
        if False:
            i = 10
            return i + 15
        self.particles.renderer.setFinalZScale(zScale)

    def setRendererGeomColorBlendMethod(self, blendMethod):
        if False:
            for i in range(10):
                print('nop')
        blendMethodStr = blendMethod
        incomingOperandStr = self.getVariable('Geom Renderer', 'Incoming Op.').get()
        fbufferOperandStr = self.getVariable('Geom Renderer', 'Fbuffer Op.').get()
        self.setRendererColorBlendAttrib('Geom Renderer', blendMethodStr, incomingOperandStr, fbufferOperandStr)

    def setRendererGeomColorBlendIncomingOperand(self, operand):
        if False:
            return 10
        blendMethodStr = self.getVariable('Geom Renderer', 'Color Blend').get()
        incomingOperandStr = operand
        fbufferOperandStr = self.getVariable('Geom Renderer', 'Fbuffer Op.').get()
        self.setRendererColorBlendAttrib('Geom Renderer', blendMethodStr, incomingOperandStr, fbufferOperandStr)

    def setRendererGeomColorBlendFbufferOperand(self, operand):
        if False:
            i = 10
            return i + 15
        blendMethodStr = self.getVariable('Geom Renderer', 'Color Blend').get()
        incomingOperandStr = self.getVariable('Geom Renderer', 'Incoming Op.').get()
        fbufferOperandStr = operand
        self.setRendererColorBlendAttrib('Geom Renderer', blendMethodStr, incomingOperandStr, fbufferOperandStr)

    def addConstantInterpolationSegment(self, id=None):
        if False:
            while True:
                i = 10
        ren = self.particles.getRenderer()
        cim = ren.getColorInterpolationManager()
        if id is None:
            seg = cim.getSegment(cim.addConstant())
        else:
            seg = cim.getSegment(id)
        if ren.__class__.__name__ == 'SpriteParticleRendererExt':
            parent = self.rendererSpriteSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Constant'
            self.rendererSegmentWidgetList.append(self.createConstantInterpolationSegmentWidget(parent, segName, seg))
        elif ren.__class__.__name__ == 'GeomParticleRenderer':
            parent = self.rendererGeomSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Constant'
            self.rendererSegmentWidgetList.append(self.createConstantInterpolationSegmentWidget(parent, segName, seg))
        parent.pack(fill=tk.BOTH, expand=1)

    def addLinearInterpolationSegment(self, id=None):
        if False:
            while True:
                i = 10
        ren = self.particles.getRenderer()
        cim = ren.getColorInterpolationManager()
        if id is None:
            seg = cim.getSegment(cim.addLinear())
        else:
            seg = cim.getSegment(id)
        if ren.__class__.__name__ == 'SpriteParticleRendererExt':
            parent = self.rendererSpriteSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Linear'
            self.rendererSegmentWidgetList.append(self.createLinearInterpolationSegmentWidget(parent, segName, seg))
        elif ren.__class__.__name__ == 'GeomParticleRenderer':
            parent = self.rendererGeomSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Linear'
            self.rendererSegmentWidgetList.append(self.createLinearInterpolationSegmentWidget(parent, segName, seg))
        parent.pack(fill=tk.BOTH, expand=1)

    def addStepwaveInterpolationSegment(self, id=None):
        if False:
            return 10
        ren = self.particles.getRenderer()
        cim = ren.getColorInterpolationManager()
        if id is None:
            seg = cim.getSegment(cim.addStepwave())
        else:
            seg = cim.getSegment(id)
        if ren.__class__.__name__ == 'SpriteParticleRendererExt':
            parent = self.rendererSpriteSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Stepwave'
            self.rendererSegmentWidgetList.append(self.createStepwaveInterpolationSegmentWidget(parent, segName, seg))
        elif ren.__class__.__name__ == 'GeomParticleRenderer':
            parent = self.rendererGeomSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Stepwave'
            self.rendererSegmentWidgetList.append(self.createStepwaveInterpolationSegmentWidget(parent, segName, seg))
        parent.pack(fill=tk.BOTH, expand=1)

    def addSinusoidInterpolationSegment(self, id=None):
        if False:
            for i in range(10):
                print('nop')
        ren = self.particles.getRenderer()
        cim = ren.getColorInterpolationManager()
        if id is None:
            seg = cim.getSegment(cim.addSinusoid())
        else:
            seg = cim.getSegment(id)
        if ren.__class__.__name__ == 'SpriteParticleRendererExt':
            parent = self.rendererSpriteSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Sinusoid'
            self.rendererSegmentWidgetList.append(self.createSinusoidInterpolationSegmentWidget(parent, segName, seg))
        elif ren.__class__.__name__ == 'GeomParticleRenderer':
            parent = self.rendererGeomSegmentFrame
            segName = repr(len(self.rendererSegmentWidgetList)) + ':Sinusoid'
            self.rendererSegmentWidgetList.append(self.createSinusoidInterpolationSegmentWidget(parent, segName, seg))
        parent.pack(fill=tk.BOTH, expand=1)

    def createWidgetForExistingInterpolationSegment(self, id):
        if False:
            for i in range(10):
                print('nop')
        ren = self.particles.getRenderer()
        cim = ren.getColorInterpolationManager()
        seg = cim.getSegment(id)
        assert seg
        fun = seg.getFunction()
        if isinstance(fun, ColorInterpolationFunctionSinusoid):
            self.addSinusoidInterpolationSegment(id)
        elif isinstance(fun, ColorInterpolationFunctionStepwave):
            self.addStepwaveInterpolationSegment(id)
        elif isinstance(fun, ColorInterpolationFunctionLinear):
            self.addLinearInterpolationSegment(id)
        elif isinstance(fun, ColorInterpolationFunctionConstant):
            self.addConstantInterpolationSegment(id)

    def createInterpolationSegmentFrame(self, parent, segName, seg):
        if False:
            while True:
                i = 10
        frame = tk.Frame(parent, relief=tk.RAISED, borderwidth=2)
        lFrame = tk.Frame(frame, relief=tk.FLAT)

        def removeInterpolationSegmentFrame(s=self, seg=seg, fr=frame):
            if False:
                i = 10
                return i + 15
            s.particles.getRenderer().getColorInterpolationManager().clearSegment(seg.getId())
            fr.pack_forget()

        def setSegEnabled(s=self, n=segName):
            if False:
                for i in range(10):
                    print('nop')
            enabled = s.getVariable('Sprite Renderer', n + ' Enabled')
            seg.setEnabled(enabled.get())

        def setIsModulated(s=self, n=segName):
            if False:
                for i in range(10):
                    print('nop')
            modulated = s.getVariable('Sprite Renderer', n + ' isModulated')
            seg.setIsModulated(modulated.get())

        def setSegBegin(time):
            if False:
                while True:
                    i = 10
            seg.setTimeBegin(time)

        def setSegEnd(time):
            if False:
                print('Hello World!')
            seg.setTimeEnd(time)
        tk.Button(lFrame, text='X', command=removeInterpolationSegmentFrame).pack(side=tk.RIGHT, expand=0)
        tk.Label(lFrame, text=segName, foreground='Blue', font=('MSSansSerif', 12, 'bold')).pack(fill=tk.X, expand=1)
        lFrame.pack(fill=tk.X, expand=1)
        lFrame = tk.Frame(frame, relief=tk.FLAT)
        self.createCheckbutton(lFrame, 'Sprite Renderer', segName + ' Enabled', 'On: Enabled\n' + 'Off: Disabled', command=setSegEnabled, initialState=seg.isEnabled())
        self.createCheckbutton(lFrame, 'Sprite Renderer', segName + ' isModulated', 'On: Modulate\n' + 'Off: Add', command=setIsModulated, initialState=seg.isModulated())
        lFrame.pack(fill=tk.X, expand=1)
        f = tk.Frame(frame)
        self.createSlider(f, 'Sprite Renderer', segName + ' Begin', '', command=setSegBegin, value=seg.getTimeBegin())
        self.createSlider(f, 'Sprite Renderer', segName + ' End', '', command=setSegEnd, value=seg.getTimeEnd())
        f.pack(fill=tk.X, expand=0)
        frame.pack(pady=3, fill=tk.X, expand=0)
        return frame

    def createConstantInterpolationSegmentWidget(self, parent, segName, segment):
        if False:
            for i in range(10):
                print('nop')
        fun = segment.getFunction()

        def setSegColorA(color):
            if False:
                return 10
            fun.setColorA(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))
        frame = self.createInterpolationSegmentFrame(parent, segName, segment)
        f = tk.Frame(frame)
        c = fun.getColorA()
        c = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0]
        self.createColorEntry(f, 'Sprite Renderer', segName + ' Color A', '', command=setSegColorA, value=c)
        f.pack(fill=tk.X)
        return frame

    def createLinearInterpolationSegmentWidget(self, parent, segName, segment):
        if False:
            return 10
        fun = segment.getFunction()

        def setSegColorA(color):
            if False:
                return 10
            fun.setColorA(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

        def setSegColorB(color):
            if False:
                i = 10
                return i + 15
            fun.setColorB(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))
        frame = self.createInterpolationSegmentFrame(parent, segName, segment)
        f = tk.Frame(frame)
        c = fun.getColorA()
        c = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0]
        self.createColorEntry(f, 'Sprite Renderer', segName + ' Color A', '', command=setSegColorA, value=c)
        c = fun.getColorB()
        c = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0]
        self.createColorEntry(f, 'Sprite Renderer', segName + ' Color B', '', command=setSegColorB, value=c)
        f.pack(fill=tk.X)
        return frame

    def createStepwaveInterpolationSegmentWidget(self, parent, segName, segment):
        if False:
            return 10
        fun = segment.getFunction()

        def setColorA(color):
            if False:
                while True:
                    i = 10
            fun.setColorA(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

        def setColorB(color):
            if False:
                i = 10
                return i + 15
            fun.setColorB(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

        def setWidthA(width):
            if False:
                return 10
            fun.setWidthA(width)

        def setWidthB(width):
            if False:
                i = 10
                return i + 15
            fun.setWidthB(width)
        frame = self.createInterpolationSegmentFrame(parent, segName, segment)
        f = tk.Frame(frame)
        c = fun.getColorA()
        c = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0]
        self.createColorEntry(f, 'Sprite Renderer', segName + ' Color A', '', command=setColorA, value=c)
        c = fun.getColorB()
        c = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0]
        self.createColorEntry(f, 'Sprite Renderer', segName + ' Color B', '', command=setColorB, value=c)
        w = fun.getWidthA()
        self.createSlider(f, 'Sprite Renderer', segName + ' Width A', '', command=setWidthA, value=w)
        w = fun.getWidthB()
        self.createSlider(f, 'Sprite Renderer', segName + ' Width B', '', command=setWidthB, value=w)
        f.pack(fill=tk.X)
        return frame

    def createSinusoidInterpolationSegmentWidget(self, parent, segName, segment):
        if False:
            return 10
        fun = segment.getFunction()

        def setColorA(color):
            if False:
                return 10
            fun.setColorA(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

        def setColorB(color):
            if False:
                i = 10
                return i + 15
            fun.setColorB(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))

        def setPeriod(period):
            if False:
                print('Hello World!')
            fun.setPeriod(period)
        frame = self.createInterpolationSegmentFrame(parent, segName, segment)
        f = tk.Frame(frame)
        c = fun.getColorA()
        c = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0]
        self.createColorEntry(f, 'Sprite Renderer', segName + ' Color A', '', command=setColorA, value=c)
        c = fun.getColorB()
        c = [c[0] * 255.0, c[1] * 255.0, c[2] * 255.0, c[3] * 255.0]
        self.createColorEntry(f, 'Sprite Renderer', segName + ' Color B', '', command=setColorB, value=c)
        p = fun.getPeriod()
        self.createFloater(f, 'Sprite Renderer', segName + ' Period', '', command=setPeriod, value=p)
        f.pack(fill=tk.X)
        return frame

    def createSpriteAnimationFrame(self, parent, anim, animName):
        if False:
            return 10
        ren = self.particles.getRenderer()
        frame = tk.Frame(parent, relief=tk.RAISED, borderwidth=2)
        frame.pack(pady=1, fill=tk.X, expand=0)
        lFrame = tk.Frame(frame, relief=tk.FLAT)
        lFrame.pack(fill=tk.X, expand=1)

        def delete(s=self, fr=frame):
            if False:
                for i in range(10):
                    print('nop')
            i = s.rendererSpriteAnimationWidgetList.index(fr)
            s.rendererSpriteAnimationWidgetList[i] = None
            fr.pack_forget()
            fr.destroy()
            s.writeSpriteRendererAnimations()
            s.readSpriteRendererAnimations()
        tk.Button(lFrame, text='X', foreground='Red', font=('MSSansSerif', 8, 'bold'), command=delete).pack(side=tk.RIGHT, expand=0)
        if anim == SpriteAnim.STTexture or anim == SpriteAnim.STFromNode:
            frame.valid = False
            frame.animSourceType = anim
            if anim == SpriteAnim.STTexture:
                type = 'Texture'
            else:
                type = 'From Node'
        else:
            frame.valid = True
            if anim.getSourceType() == SpriteAnim.STTexture:
                frame.animSourceType = SpriteAnim.STTexture
                type = 'Texture'
            else:
                frame.animSourceType = SpriteAnim.STFromNode
                type = 'From Node'
        tk.Label(lFrame, text=animName + ': ' + type, foreground='Blue', font=('MSSansSerif', 12, 'bold')).pack(fill=tk.X, expand=1)
        return frame

    def createSpriteAnimationTextureWidget(self, parent, anim, animName):
        if False:
            print('Hello World!')
        ren = self.particles.getRenderer()
        frame = self.createSpriteAnimationFrame(parent, anim, animName)
        f = tk.Frame(frame)
        f.pack(fill=tk.X)
        tk.Label(f, text='Texture: ', font=('MSSansSerif', 12), width=7).pack(side=tk.LEFT)
        strVar = tk.StringVar()
        entry = tk.Entry(f, textvariable=strVar).pack(padx=3, pady=3, side=tk.LEFT, fill=tk.X, expand=1)
        if frame.valid:
            strVar.set(anim.getTexSource())
        else:
            strVar.set('Base model path: ' + repr(getModelPath().getValue()))

        def checkForTexture(strVar=strVar):
            if False:
                while True:
                    i = 10
            tex = base.loader.loadTexture(strVar.get())
            if tex:
                frame.valid = True
            else:
                frame.valid = False
            self.writeSpriteRendererAnimations()
        tk.Button(f, text='Update', command=checkForTexture).pack(side=tk.LEFT)
        self.variableDict['Sprite Renderer-' + animName + ' Anim Texture'] = strVar
        self.widgetDict['Sprite Renderer-' + animName + ' Anim Texture'] = entry
        return frame

    def createSpriteAnimationNodeWidget(self, parent, anim, animName):
        if False:
            while True:
                i = 10
        ren = self.particles.getRenderer()
        frame = self.createSpriteAnimationFrame(parent, anim, animName)
        f = tk.Frame(frame)
        f.pack(fill=tk.X)
        lf = tk.Frame(f)
        lf.pack(fill=tk.X, expand=1)
        tk.Label(lf, text='Model: ', font=('MSSansSerif', 12), width=7).pack(side=tk.LEFT)
        mStrVar = tk.StringVar()
        entry = tk.Entry(lf, textvariable=mStrVar).pack(padx=3, pady=3, side=tk.LEFT, fill=tk.X, expand=1)
        if frame.valid:
            mStrVar.set(anim.getModelSource())
        else:
            mStrVar.set('Base model path: ' + repr(getModelPath().getValue()))
        mlf = lf
        self.variableDict['Sprite Renderer-' + animName + ' Anim Model'] = mStrVar
        self.widgetDict['Sprite Renderer-' + animName + ' Anim Model'] = entry
        lf = tk.Frame(f)
        lf.pack(fill=tk.X, expand=1)
        tk.Label(lf, text='Node: ', font=('MSSansSerif', 12), width=7).pack(side=tk.LEFT)
        nStrVar = tk.StringVar()
        entry = tk.Entry(lf, textvariable=nStrVar).pack(padx=3, pady=3, side=tk.LEFT, fill=tk.X, expand=1)
        if frame.valid:
            nStrVar.set(anim.getNodeSource())
        else:
            nStrVar.set('**/*')
        nlf = lf
        self.variableDict['Sprite Renderer-' + animName + ' Anim Node'] = nStrVar
        self.widgetDict['Sprite Renderer-' + animName + ' Anim Node'] = entry

        def checkForNode(modelStrVar=mStrVar, nodeStrVar=nStrVar):
            if False:
                print('Hello World!')
            mod = base.loader.loadModel(modelStrVar.get())
            if mod:
                node = mod.find(nodeStrVar.get())
                if node:
                    frame.valid = True
                else:
                    frame.valid = False
            else:
                frame.valid = False
            self.writeSpriteRendererAnimations()
        tk.Button(mlf, text='Update', command=checkForNode).pack(side=tk.LEFT)
        tk.Button(nlf, text='Update', command=checkForNode).pack(side=tk.LEFT)
        return frame

    def readSpriteRendererAnimations(self):
        if False:
            while True:
                i = 10
        ren = self.particles.getRenderer()
        for widget in self.rendererSpriteAnimationWidgetList:
            if widget:
                widget.pack_forget()
                widget.destroy()
        self.rendererSpriteAnimationWidgetList = []
        for anim in [ren.getAnim(x) for x in range(ren.getNumAnims())]:
            if anim.getSourceType() == SpriteAnim.STTexture:
                w = self.createSpriteAnimationTextureWidget(self.rendererSpriteAnimationFrame, anim, repr(len(self.rendererSpriteAnimationWidgetList)))
            else:
                w = self.createSpriteAnimationNodeWidget(self.rendererSpriteAnimationFrame, anim, repr(len(self.rendererSpriteAnimationWidgetList)))
            self.rendererSpriteAnimationWidgetList.append(w)

    def writeSpriteRendererAnimations(self):
        if False:
            for i in range(10):
                print('nop')
        ren = self.particles.getRenderer()
        for x in range(ren.getNumAnims()):
            ren.removeAnimation(0)
        for x in range(len(self.rendererSpriteAnimationWidgetList)):
            widget = self.rendererSpriteAnimationWidgetList[x]
            if widget and widget.valid:
                if widget.animSourceType == SpriteAnim.STTexture:
                    texSource = self.getVariable('Sprite Renderer', repr(x) + ' Anim Texture').get()
                    ren.addTextureFromFile(texSource)
                else:
                    modelSource = self.getVariable('Sprite Renderer', repr(x) + ' Anim Model').get()
                    nodeSource = self.getVariable('Sprite Renderer', repr(x) + ' Anim Node').get()
                    ren.addTextureFromNode(modelSource, nodeSource)

    def updateForceWidgets(self):
        if False:
            print('Hello World!')
        if self.forceGroup is not None:
            self.forceGroupNotebook.pack(fill=tk.X)
            self.forcePageName = self.particleEffect.getName() + '-' + self.forceGroup.getName()
            self.forcePage = self.forcePagesDict.get(self.forcePageName, None)
            if self.forcePage is None:
                self.addForceGroupNotebookPage(self.particleEffect, self.forceGroup)
            self.forceGroupNotebook.selectpage(self.forcePageName)
        else:
            self.forceGroupNotebook.pack_forget()

    def addLinearVectorForce(self):
        if False:
            for i in range(10):
                print('nop')
        self.addForce(LinearVectorForce())

    def addLinearFrictionForce(self):
        if False:
            print('Hello World!')
        self.addForce(LinearFrictionForce())

    def addLinearJitterForce(self):
        if False:
            while True:
                i = 10
        self.addForce(LinearJitterForce())

    def addLinearNoiseForce(self):
        if False:
            return 10
        self.addForce(LinearNoiseForce())

    def addLinearSinkForce(self):
        if False:
            i = 10
            return i + 15
        self.addForce(LinearSinkForce())

    def addLinearSourceForce(self):
        if False:
            return 10
        self.addForce(LinearSourceForce())

    def addLinearCylinderVortexForce(self):
        if False:
            while True:
                i = 10
        self.addForce(LinearCylinderVortexForce())

    def addLinearUserDefinedForce(self):
        if False:
            while True:
                i = 10
        self.addForce(LinearUserDefinedForce())

    def addForce(self, f):
        if False:
            while True:
                i = 10
        if self.forceGroup is None:
            self.createNewForceGroup()
        self.forceGroup.addForce(f)
        self.addForceWidget(self.forceGroup, f)

    def createNewEffect(self):
        if False:
            i = 10
            return i + 15
        name = askstring('Particle Panel', 'Effect Name:', parent=self.parent)
        if name:
            particles = Particles.Particles()
            particles.setBirthRate(0.02)
            particles.setLitterSize(10)
            particles.setLitterSpread(0)
            particles.setFactory('PointParticleFactory')
            particles.setRenderer('PointParticleRenderer')
            particles.setEmitter('SphereVolumeEmitter')
            particles.enable()
            effect = ParticleEffect.ParticleEffect(name, particles)
            self.effectsDict[name] = effect
            self.updateMenusAndLabels()
            self.selectEffectNamed(name)
            effect.reparentTo(render)
            effect.enable()

    def createNewParticles(self):
        if False:
            while True:
                i = 10
        name = askstring('Particle Panel', 'Particles Name:', parent=self.parent)
        if name:
            p = Particles.Particles(name)
            self.particleEffect.addParticles(p)
            self.updateParticlesMenus()
            self.selectParticlesNamed(name)
            p.enable()

    def createNewForceGroup(self):
        if False:
            i = 10
            return i + 15
        name = askstring('Particle Panel', 'ForceGroup Name:', parent=self.parent)
        if name:
            forceGroup = ForceGroup.ForceGroup(name)
            self.particleEffect.addForceGroup(forceGroup)
            self.updateForceGroupMenus()
            self.addForceGroupNotebookPage(self.particleEffect, forceGroup)
            self.selectForceGroupNamed(name)
            forceGroup.enable()

    def addForceGroupNotebookPage(self, particleEffect, forceGroup):
        if False:
            i = 10
            return i + 15
        self.forcePageName = particleEffect.getName() + '-' + forceGroup.getName()
        self.forcePage = self.forceGroupNotebook.add(self.forcePageName)
        self.forcePagesDict[self.forcePageName] = self.forcePage
        for force in forceGroup:
            self.addForceWidget(forceGroup, force)

    def addForceWidget(self, forceGroup, force):
        if False:
            return 10
        forcePage = self.forcePage
        pageName = self.forcePageName
        count = 0
        for f in forceGroup:
            if f.getClassType() == force.getClassType():
                count += 1
        if isinstance(force, LinearVectorForce):
            self.createLinearVectorForceWidget(forcePage, pageName, count, force)
        elif isinstance(force, LinearNoiseForce):
            self.createLinearRandomForceWidget(forcePage, pageName, count, force, 'Noise')
        elif isinstance(force, LinearJitterForce):
            self.createLinearRandomForceWidget(forcePage, pageName, count, force, 'Jitter')
        elif isinstance(force, LinearFrictionForce):
            self.createLinearFrictionForceWidget(forcePage, pageName, count, force)
        elif isinstance(force, LinearCylinderVortexForce):
            self.createLinearCylinderVortexForceWidget(forcePage, pageName, count, force)
        elif isinstance(force, LinearSinkForce):
            self.createLinearDistanceForceWidget(forcePage, pageName, count, force, 'Sink')
        elif isinstance(force, LinearSourceForce):
            self.createLinearDistanceForceWidget(forcePage, pageName, count, force, 'Source')
        elif isinstance(force, LinearUserDefinedForce):
            pass
        self.forceGroupNotebook.setnaturalsize()

    def createForceFrame(self, forcePage, forceName, force):
        if False:
            i = 10
            return i + 15
        frame = tk.Frame(forcePage, relief=tk.RAISED, borderwidth=2)
        lFrame = tk.Frame(frame, relief=tk.FLAT)

        def removeForce(s=self, f=force, fr=frame):
            if False:
                print('Hello World!')
            s.forceGroup.removeForce(f)
            fr.pack_forget()
        b = tk.Button(lFrame, text='X', command=removeForce)
        b.pack(side='right', expand=0)
        tk.Label(lFrame, text=forceName, foreground='Blue', font=('MSSansSerif', 12, 'bold')).pack(expand=1, fill='x')
        lFrame.pack(fill='x', expand=1)
        frame.pack(pady=3, fill='x', expand=0)
        return frame

    def createLinearForceWidgets(self, frame, pageName, forceName, force):
        if False:
            for i in range(10):
                print('nop')

        def setAmplitude(amp, f=force):
            if False:
                print('Hello World!')
            f.setAmplitude(amp)

        def toggleMassDependent(s=self, f=force, p=pageName, n=forceName):
            if False:
                print('Hello World!')
            v = s.getVariable(p, n + ' Mass Dependent')
            f.setMassDependent(v.get())

        def setVectorMasks(s=self, f=force, p=pageName, n=forceName):
            if False:
                while True:
                    i = 10
            xMask = s.getVariable(p, n + ' Mask X').get()
            yMask = s.getVariable(p, n + ' Mask Y').get()
            zMask = s.getVariable(p, n + ' Mask Z').get()
            f.setVectorMasks(xMask, yMask, zMask)
        self.createFloater(frame, pageName, forceName + ' Amplitude', 'Force amplitude multiplier', command=setAmplitude, value=force.getAmplitude())
        cbf = tk.Frame(frame, relief=tk.FLAT)
        self.createCheckbutton(cbf, pageName, forceName + ' Mass Dependent', 'On: force depends on mass; ' + 'Off: force does not depend on mass', toggleMassDependent, force.getMassDependent())
        self.createCheckbutton(cbf, pageName, forceName + ' Mask X', 'On: enable force along X axis', setVectorMasks, 1)
        self.createCheckbutton(cbf, pageName, forceName + ' Mask Y', 'On: enable force along X axis', setVectorMasks, 1)
        self.createCheckbutton(cbf, pageName, forceName + ' Mask Z', 'On: enable force along X axis', setVectorMasks, 1)
        cbf.pack(fill='x', expand=0)

    def createForceActiveWidget(self, frame, pageName, forceName, force):
        if False:
            for i in range(10):
                print('nop')
        cbName = forceName + ' Active'

        def toggle(s=self, f=force, p=pageName, n=cbName):
            if False:
                while True:
                    i = 10
            s.toggleForce(f, p, n)
        self.createCheckbutton(frame, pageName, cbName, 'On: force is enabled; Off: force is disabled', toggle, 1)

    def createLinearVectorForceWidget(self, forcePage, pageName, count, force):
        if False:
            for i in range(10):
                print('nop')

        def setVec(vec, f=force):
            if False:
                i = 10
                return i + 15
            f.setVector(vec[0], vec[1], vec[2])
        forceName = 'Vector Force-' + repr(count)
        frame = self.createForceFrame(forcePage, forceName, force)
        self.createLinearForceWidgets(frame, pageName, forceName, force)
        vec = force.getLocalVector()
        self.createVector3Entry(frame, pageName, forceName, 'Set force direction and magnitude', command=setVec, value=[vec[0], vec[1], vec[2]])
        self.createForceActiveWidget(frame, pageName, forceName, force)

    def createLinearRandomForceWidget(self, forcePage, pageName, count, force, type):
        if False:
            while True:
                i = 10
        forceName = type + ' Force-' + repr(count)
        frame = self.createForceFrame(forcePage, forceName, force)
        self.createLinearForceWidgets(frame, pageName, forceName, force)
        self.createForceActiveWidget(frame, pageName, forceName, force)

    def createLinearFrictionForceWidget(self, forcePage, pageName, count, force):
        if False:
            for i in range(10):
                print('nop')

        def setCoef(coef, f=force):
            if False:
                i = 10
                return i + 15
            f.setCoef(coef)
        forceName = 'Friction Force-' + repr(count)
        frame = self.createForceFrame(forcePage, forceName, force)
        self.createLinearForceWidgets(frame, pageName, forceName, force)
        self.createFloater(frame, pageName, forceName + ' Coef', 'Set linear friction force', command=setCoef, min=None, value=force.getCoef())
        self.createForceActiveWidget(frame, pageName, forceName, force)

    def createLinearCylinderVortexForceWidget(self, forcePage, pageName, count, force):
        if False:
            for i in range(10):
                print('nop')
        forceName = 'Vortex Force-' + repr(count)

        def setCoef(coef, f=force):
            if False:
                return 10
            f.setCoef(coef)

        def setLength(length, f=force):
            if False:
                print('Hello World!')
            f.setLength(length)

        def setRadius(radius, f=force):
            if False:
                return 10
            f.setRadius(radius)
        frame = self.createForceFrame(forcePage, forceName, force)
        self.createLinearForceWidgets(frame, pageName, forceName, force)
        self.createFloater(frame, pageName, forceName + ' Coef', 'Set linear cylinder vortex coefficient', command=setCoef, value=force.getCoef())
        self.createFloater(frame, pageName, forceName + ' Length', 'Set linear cylinder vortex length', command=setLength, value=force.getLength())
        self.createFloater(frame, pageName, forceName + ' Radius', 'Set linear cylinder vortex radius', command=setRadius, value=force.getRadius())
        self.createForceActiveWidget(frame, pageName, forceName, force)

    def createLinearDistanceForceWidget(self, forcePage, pageName, count, force, type):
        if False:
            while True:
                i = 10

        def setFalloffType(type, f=force):
            if False:
                i = 10
                return i + 15
            if type == 'FT_ONE_OVER_R':
                f.setFalloffType(0)
            if type == 'FT_ONE_OVER_R_SQUARED':
                f.setFalloffType(1)
            if type == 'FT_ONE_OVER_R_CUBED':
                f.setFalloffType(2)

        def setForceCenter(vec, f=force):
            if False:
                for i in range(10):
                    print('nop')
            f.setForceCenter(Point3(vec[0], vec[1], vec[2]))

        def setRadius(radius, f=force):
            if False:
                i = 10
                return i + 15
            f.setRadius(radius)
        forceName = type + ' Force-' + repr(count)
        frame = self.createForceFrame(forcePage, forceName, force)
        self.createLinearForceWidgets(frame, pageName, forceName, force)
        var = self.createOptionMenu(frame, pageName, forceName + ' Falloff', 'Set force falloff type', ('FT_ONE_OVER_R', 'FT_ONE_OVER_R_SQUARED', 'FT_ONE_OVER_R_CUBED'), command=setFalloffType)
        self.getWidget(pageName, forceName + ' Falloff').configure(label_width=16)
        falloff = force.getFalloffType()
        if falloff == LinearDistanceForce.FTONEOVERR:
            var.set('FT_ONE_OVER_R')
        elif falloff == LinearDistanceForce.FTONEOVERRSQUARED:
            var.set('FT_ONE_OVER_R_SQUARED')
        elif falloff == LinearDistanceForce.FTONEOVERRCUBED:
            var.set('FT_ONE_OVER_R_CUBED')
        vec = force.getForceCenter()
        self.createVector3Entry(frame, pageName, forceName + ' Center', 'Set center of force', command=setForceCenter, label_width=16, value=[vec[0], vec[1], vec[2]])
        self.createFloater(frame, pageName, forceName + ' Radius', 'Set falloff radius', command=setRadius, min=0.01, value=force.getRadius())
        self.createForceActiveWidget(frame, pageName, forceName, force)