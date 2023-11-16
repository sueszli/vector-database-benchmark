"""This module contains a high-level interface for loading models, textures,
sound, music, shaders and fonts from disk.
"""
__all__ = ['Loader']
from panda3d.core import ConfigVariableBool, Filename, FontPool, LoaderFileTypeRegistry, LoaderOptions, ModelFlattenRequest, ModelNode, ModelPool, NodePath, PandaNode, SamplerState, ShaderPool, StaticTextFont, TexturePool, VBase4
from panda3d.core import Loader as PandaLoader
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject
import warnings
import sys
phaseChecker = None

class Loader(DirectObject):
    """
    Load models, textures, sounds, and code.
    """
    notify = directNotify.newCategory('Loader')
    loaderIndex = 0
    _loadedPythonFileTypes = False

    class _Callback:
        """Returned by loadModel when used asynchronously.  This class is
        modelled after Future, and can be awaited."""
        _asyncio_future_blocking = False

        def __init__(self, loader, numObjects, gotList, callback, extraArgs):
            if False:
                print('Hello World!')
            self._loader = loader
            self.objects = [None] * numObjects
            self.gotList = gotList
            self.callback = callback
            self.extraArgs = extraArgs
            self.requests = set()
            self.requestList = []

        def gotObject(self, index, object):
            if False:
                return 10
            self.objects[index] = object
            if not self.requests:
                self._loader = None
                if self.callback:
                    if self.gotList:
                        self.callback(self.objects, *self.extraArgs)
                    else:
                        self.callback(*self.objects + self.extraArgs)

        def cancel(self):
            if False:
                for i in range(10):
                    print('nop')
            "Cancels the request.  Callback won't be called."
            if self._loader:
                for request in self.requests:
                    self._loader.loader.remove(request)
                    del self._loader._requests[request]
                self._loader = None
                self.requests = None
                self.requestList = None

        def cancelled(self):
            if False:
                print('Hello World!')
            'Returns true if the request was cancelled.'
            return self.requestList is None

        def done(self):
            if False:
                print('Hello World!')
            'Returns true if all the requests were finished or cancelled.'
            return not self.requests

        def result(self):
            if False:
                i = 10
                return i + 15
            'Returns the results, suspending the thread to wait if necessary.'
            for r in list(self.requests):
                r.wait()
            if self.gotList:
                return self.objects
            else:
                return self.objects[0]

        def exception(self):
            if False:
                for i in range(10):
                    print('nop')
            assert self.done() and (not self.cancelled())
            return None

        def __await__(self):
            if False:
                i = 10
                return i + 15
            " Returns a generator that raises StopIteration when the loading\n            is complete.  This allows this class to be used with 'await'."
            if self.requests:
                self._asyncio_future_blocking = True
                while self.requests:
                    yield self
            if self.gotList:
                return self.objects
            else:
                return self.objects[0]

        async def __aiter__(self):
            """ This allows using `async for` to iterate asynchronously over
            the results of this class.  It does guarantee to return the
            results in order, though, even though they may not be loaded in
            that order. """
            requestList = self.requestList
            assert requestList is not None, 'Request was cancelled.'
            for req in requestList:
                yield (await req)

    def __init__(self, base):
        if False:
            i = 10
            return i + 15
        self.base = base
        self.loader = PandaLoader.getGlobalPtr()
        self._requests = {}
        self.hook = 'async_loader_%s' % Loader.loaderIndex
        Loader.loaderIndex += 1
        self.accept(self.hook, self.__gotAsyncObject)
        self._loadPythonFileTypes()

    def destroy(self):
        if False:
            return 10
        self.ignore(self.hook)
        self.loader.stopThreads()
        del self.base
        del self.loader

    @classmethod
    def _loadPythonFileTypes(cls):
        if False:
            print('Hello World!')
        if cls._loadedPythonFileTypes:
            return
        if not ConfigVariableBool('loader-support-entry-points', True):
            return
        from importlib.metadata import entry_points
        eps = entry_points()
        if sys.version_info < (3, 10):
            loaders = eps.get('panda3d.loaders', ())
        else:
            loaders = eps.select(group='panda3d.loaders')
        if loaders:
            registry = LoaderFileTypeRegistry.getGlobalPtr()
            for entry_point in loaders:
                registry.register_deferred_type(entry_point)
            cls._loadedPythonFileTypes = True

    def loadModel(self, modelPath, loaderOptions=None, noCache=None, allowInstance=False, okMissing=None, callback=None, extraArgs=[], priority=None, blocking=None):
        if False:
            while True:
                i = 10
        '\n        Attempts to load a model or models from one or more relative\n        pathnames.  If the input modelPath is a string (a single model\n        pathname), the return value will be a NodePath to the model\n        loaded if the load was successful, or None otherwise.  If the\n        input modelPath is a list of pathnames, the return value will\n        be a list of `.NodePath` objects and/or Nones.\n\n        loaderOptions may optionally be passed in to control details\n        about the way the model is searched and loaded.  See the\n        `.LoaderOptions` class for more.\n\n        The default is to look in the `.ModelPool` (RAM) cache first,\n        and return a copy from that if the model can be found there.\n        If the bam cache is enabled (via the `model-cache-dir` config\n        variable), then that will be consulted next, and if both\n        caches fail, the file will be loaded from disk.  If noCache is\n        True, then neither cache will be consulted or updated.\n\n        If allowInstance is True, a shared instance may be returned\n        from the `.ModelPool`.  This is dangerous, since it is easy to\n        accidentally modify the shared instance, and invalidate future\n        load attempts of the same model.  Normally, you should leave\n        allowInstance set to False, which will always return a unique\n        copy.\n\n        If okMissing is True, None is returned if the model is not\n        found or cannot be read, and no error message is printed.\n        Otherwise, an `IOError` is raised if the model is not found or\n        cannot be read (similar to attempting to open a nonexistent\n        file).  (If modelPath is a list of filenames, then `IOError`\n        is raised if *any* of the models could not be loaded.)\n\n        If callback is not None, then the model load will be performed\n        asynchronously.  In this case, loadModel() will initiate a\n        background load and return immediately.  The return value will\n        be an object that can be used to check the status, cancel the\n        request, or use it in an `await` expression.  Unless callback\n        is the special value True, when the requested model(s) have\n        finished loading, it will be invoked with the n\n        loaded models passed as its parameter list.  It is possible\n        that the callback will be invoked immediately, even before\n        loadModel() returns.  If you use callback, you may also\n        specify a priority, which specifies the relative importance\n        over this model over all of the other asynchronous load\n        requests (higher numbers are loaded first).\n\n        True asynchronous model loading requires Panda to have been\n        compiled with threading support enabled (you can test\n        `.Thread.isThreadingSupported()`).  In the absence of threading\n        support, the asynchronous interface still exists and still\n        behaves exactly as described, except that loadModel() might\n        not return immediately.\n\n        '
        assert Loader.notify.debug('Loading model: %s' % (modelPath,))
        if loaderOptions is None:
            loaderOptions = LoaderOptions()
        else:
            loaderOptions = LoaderOptions(loaderOptions)
        if okMissing is not None:
            if okMissing:
                loaderOptions.setFlags(loaderOptions.getFlags() & ~LoaderOptions.LFReportErrors)
            else:
                loaderOptions.setFlags(loaderOptions.getFlags() | LoaderOptions.LFReportErrors)
        else:
            okMissing = loaderOptions.getFlags() & LoaderOptions.LFReportErrors == 0
        if noCache is not None:
            if noCache:
                loaderOptions.setFlags(loaderOptions.getFlags() | LoaderOptions.LFNoCache)
            else:
                loaderOptions.setFlags(loaderOptions.getFlags() & ~LoaderOptions.LFNoCache)
        if allowInstance:
            loaderOptions.setFlags(loaderOptions.getFlags() | LoaderOptions.LFAllowInstance)
        if not isinstance(modelPath, (tuple, list, set)):
            modelList = [modelPath]
            if phaseChecker:
                phaseChecker(modelPath, loaderOptions)
            gotList = False
        else:
            modelList = modelPath
            gotList = True
        if blocking is None:
            blocking = callback is None
        if blocking:
            result = []
            for modelPath in modelList:
                node = self.loader.loadSync(Filename(modelPath), loaderOptions)
                if node is not None:
                    nodePath = NodePath(node)
                else:
                    nodePath = None
                result.append(nodePath)
            if not okMissing and None in result:
                message = 'Could not load model file(s): %s' % (modelList,)
                raise IOError(message)
            if gotList:
                return result
            else:
                return result[0]
        else:
            cb = Loader._Callback(self, len(modelList), gotList, callback, extraArgs)
            i = 0
            for modelPath in modelList:
                request = self.loader.makeAsyncRequest(Filename(modelPath), loaderOptions)
                if priority is not None:
                    request.setPriority(priority)
                request.setDoneEvent(self.hook)
                self.loader.loadAsync(request)
                cb.requests.add(request)
                cb.requestList.append(request)
                self._requests[request] = (cb, i)
                i += 1
            return cb

    def cancelRequest(self, cb):
        if False:
            i = 10
            return i + 15
        'Cancels an aysynchronous loading or flatten request issued\n        earlier.  The callback associated with the request will not be\n        called after cancelRequest() has been performed.\n\n        This is now deprecated: call cb.cancel() instead. '
        if __debug__:
            warnings.warn('This is now deprecated: call cb.cancel() instead.', DeprecationWarning, stacklevel=2)
        cb.cancel()

    def isRequestPending(self, cb):
        if False:
            return 10
        ' Returns true if an asynchronous loading or flatten request\n        issued earlier is still pending, or false if it has completed or\n        been cancelled.\n\n        This is now deprecated: call cb.done() instead. '
        if __debug__:
            warnings.warn('This is now deprecated: call cb.done() instead.', DeprecationWarning, stacklevel=2)
        return bool(cb.requests)

    def loadModelOnce(self, modelPath):
        if False:
            i = 10
            return i + 15
        '\n        modelPath is a string.\n\n        Attempt to load a model from modelPool, if not present\n        then attempt to load it from disk. Return a nodepath to\n        the model if successful or None otherwise\n        '
        if __debug__:
            warnings.warn('loader.loadModelOnce() is deprecated; use loader.loadModel() instead.', DeprecationWarning, stacklevel=2)
        return self.loadModel(modelPath, noCache=False)

    def loadModelCopy(self, modelPath, loaderOptions=None):
        if False:
            return 10
        'loadModelCopy(self, string)\n        NOTE: This method is deprecated and should not be used.\n        Attempt to load a model from modelPool, if not present\n        then attempt to load it from disk. Return a nodepath to\n        a copy of the model if successful or None otherwise\n        '
        if __debug__:
            warnings.warn('loader.loadModelCopy() is deprecated; use loader.loadModel() instead.', DeprecationWarning, stacklevel=2)
        return self.loadModel(modelPath, loaderOptions=loaderOptions, noCache=False)

    def loadModelNode(self, modelPath):
        if False:
            for i in range(10):
                print('nop')
        "\n        modelPath is a string.\n\n        This is like loadModelOnce in that it loads a model from the\n        modelPool, but it does not then instance it to hidden and it\n        returns a Node instead of a NodePath.  This is particularly\n        useful for special models like fonts that you don't care about\n        where they're parented to, and you don't want a NodePath\n        anyway--it prevents accumulation of instances of the font\n        model under hidden.\n\n        However, if you're loading a font, see loadFont(), below.\n        "
        if __debug__:
            warnings.warn('loader.loadModelNode() is deprecated; use loader.loadModel() instead.', DeprecationWarning, stacklevel=2)
        model = self.loadModel(modelPath, noCache=False)
        if model is not None:
            model = model.node()
        return model

    def unloadModel(self, model):
        if False:
            i = 10
            return i + 15
        '\n        model is the return value of loadModel().  For backward\n        compatibility, it may also be the filename that was passed to\n        loadModel(), though this requires a disk search.\n        '
        if isinstance(model, NodePath):
            modelNode = model.node()
        elif isinstance(model, ModelNode):
            modelNode = model
        elif isinstance(model, (str, Filename)):
            options = LoaderOptions(LoaderOptions.LFSearch | LoaderOptions.LFNoDiskCache | LoaderOptions.LFCacheOnly)
            modelNode = self.loader.loadSync(Filename(model), options)
            if modelNode is None:
                assert Loader.notify.debug('Unloading model not loaded: %s' % model)
                return
            assert Loader.notify.debug('%s resolves to %s' % (model, modelNode.getFullpath()))
        else:
            raise TypeError('Invalid parameter to unloadModel: %s' % model)
        assert Loader.notify.debug('Unloading model: %s' % modelNode.getFullpath())
        ModelPool.releaseModel(modelNode)

    def saveModel(self, modelPath, node, loaderOptions=None, callback=None, extraArgs=[], priority=None, blocking=None):
        if False:
            i = 10
            return i + 15
        ' Saves the model (a `NodePath` or `PandaNode`) to the indicated\n        filename path.  Returns true on success, false on failure.  If\n        a callback is used, the model is saved asynchronously, and the\n        true/false status is passed to the callback function. '
        if loaderOptions is None:
            loaderOptions = LoaderOptions()
        else:
            loaderOptions = LoaderOptions(loaderOptions)
        if not isinstance(modelPath, (tuple, list, set)):
            modelList = [modelPath]
            nodeList = [node]
            if phaseChecker:
                phaseChecker(modelPath, loaderOptions)
            gotList = False
        else:
            modelList = modelPath
            nodeList = node
            gotList = True
        assert len(modelList) == len(nodeList)
        for (i, node) in enumerate(nodeList):
            if isinstance(node, NodePath):
                nodeList[i] = node.node()
        modelList = list(zip(modelList, nodeList))
        if blocking is None:
            blocking = callback is None
        if blocking:
            result = []
            for (modelPath, node) in modelList:
                thisResult = self.loader.saveSync(Filename(modelPath), loaderOptions, node)
                result.append(thisResult)
            if gotList:
                return result
            else:
                return result[0]
        else:
            cb = Loader._Callback(self, len(modelList), gotList, callback, extraArgs)
            i = 0
            for (modelPath, node) in modelList:
                request = self.loader.makeAsyncSaveRequest(Filename(modelPath), loaderOptions, node)
                if priority is not None:
                    request.setPriority(priority)
                request.setDoneEvent(self.hook)
                self.loader.saveAsync(request)
                cb.requests.add(request)
                cb.requestList.append(request)
                self._requests[request] = (cb, i)
                i += 1
            return cb

    def loadFont(self, modelPath, spaceAdvance=None, lineHeight=None, pointSize=None, pixelsPerUnit=None, scaleFactor=None, textureMargin=None, polyMargin=None, minFilter=None, magFilter=None, anisotropicDegree=None, color=None, outlineWidth=None, outlineFeather=0.1, outlineColor=VBase4(0, 0, 0, 1), renderMode=None, okMissing=False):
        if False:
            while True:
                i = 10
        '\n        modelPath is a string.\n\n        This loads a special model as a `TextFont` object, for rendering\n        text with a `TextNode`.  A font file must be either a special\n        egg file (or bam file) generated with egg-mkfont, which is\n        considered a static font, or a standard font file (like a TTF\n        file) that is supported by FreeType, which is considered a\n        dynamic font.\n\n        okMissing should be True to indicate the method should return\n        None if the font file is not found.  If it is False, the\n        method will raise an exception if the font file is not found\n        or cannot be loaded.\n\n        Most font-customization parameters accepted by this method\n        (except lineHeight and spaceAdvance) may only be specified for\n        dynamic font files like TTF files, not for static egg files.\n\n        lineHeight specifies the vertical distance between consecutive\n        lines, in Panda units.  If unspecified, it is taken from the\n        font information.  This parameter may be specified for static\n        as well as dynamic fonts.\n\n        spaceAdvance specifies the width of a space character (ascii\n        32), in Panda units.  If unspecified, it is taken from the\n        font information.  This may be specified for static as well as\n        dynamic fonts.\n\n        The remaining parameters may only be specified for dynamic\n        fonts.\n\n        pixelsPerUnit controls the visual quality of the rendered text\n        characters.  It specifies the number of texture pixels per\n        each Panda unit of character height.  Increasing this number\n        increases the amount of detail that can be represented in the\n        characters, at the expense of texture memory.\n\n        scaleFactor also controls the visual quality of the rendered\n        text characters.  It is the amount by which the characters are\n        rendered bigger out of Freetype, and then downscaled to fit\n        within the texture.  Increasing this number may reduce some\n        artifacts of very small font characters, at a small cost of\n        processing time to generate the characters initially.\n\n        textureMargin specifies the number of pixels of the texture to\n        leave between adjacent characters.  It may be a floating-point\n        number.  This helps reduce bleed-through from nearby\n        characters within the texture space.  Increasing this number\n        reduces artifacts at the edges of the character cells\n        (especially for very small text scales), at the expense of\n        texture memory.\n\n        polyMargin specifies the amount of additional buffer to create\n        in the polygon that represents each character, in Panda units.\n        It is similar to textureMargin, but it controls the polygon\n        buffer, not the texture buffer.  Increasing this number\n        reduces artifacts from letters getting chopped off at the\n        edges (especially for very small text scales), with some\n        increasing risk of adjacent letters overlapping and obscuring\n        each other.\n\n        minFilter, magFilter, and anisotropicDegree specify the\n        texture filter modes that should be applied to the textures\n        that are created to hold the font characters.\n\n        If color is not None, it should be a VBase4 specifying the\n        foreground color of the font.  Specifying this option breaks\n        `TextNode.setColor()`, so you almost never want to use this\n        option; the default (white) is the most appropriate for a\n        font, as it allows text to have any arbitrary color assigned\n        at generation time.  However, if you want to use a colored\n        outline (below) with a different color for the interior, for\n        instance a yellow letter with a blue outline, then you need\n        this option, and then *all* text generated with this font will\n        have to be yellow and blue.\n\n        If outlineWidth is nonzero, an outline will be created at\n        runtime for the letters, and outlineWidth will be the desired\n        width of the outline, in points (most fonts are 10 points\n        high, so 0.5 is often a good choice).  If you specify\n        outlineWidth, you can also specify outlineFeather (0.0 .. 1.0)\n        and outlineColor.  You may need to increase pixelsPerUnit to\n        get the best results.\n\n        if renderMode is not None, it may be one of the following\n        symbols to specify a geometry-based font:\n\n            TextFont.RMTexture - this is the default.  Font characters\n              are rendered into a texture and applied to a polygon.\n              This gives the best general-purpose results.\n\n            TextFont.RMWireframe - Font characters are rendered as a\n              sequence of one-pixel lines.  Consider enabling line or\n              multisample antialiasing for best results.\n\n            TextFont.RMPolygon - Font characters are rendered as a\n              flat polygon.  This works best for very large\n              characters, and generally requires polygon or\n              multisample antialiasing to be enabled for best results.\n\n            TextFont.RMExtruded - Font characters are rendered with a\n              3-D outline made of polygons, like a cookie cutter.\n              This is appropriate for a 3-D scene, but may be\n              completely invisible when assigned to a 2-D scene and\n              viewed normally from the front, since polygons are\n              infinitely thin.\n\n            TextFont.RMSolid - A combination of RMPolygon and\n              RMExtruded: a flat polygon in front with a solid\n              three-dimensional edge.  This is best for letters that\n              will be tumbling in 3-D space.\n\n        If the texture mode is other than RMTexture, most of the above\n        parameters do not apply, though pixelsPerUnit still does apply\n        and roughly controls the tightness of the curve approximation\n        (and the number of vertices generated).\n\n        '
        assert Loader.notify.debug('Loading font: %s' % modelPath)
        if phaseChecker:
            loaderOptions = LoaderOptions()
            if okMissing:
                loaderOptions.setFlags(loaderOptions.getFlags() & ~LoaderOptions.LFReportErrors)
            phaseChecker(modelPath, loaderOptions)
        font = FontPool.loadFont(modelPath)
        if font is None:
            if not okMissing:
                message = 'Could not load font file: %s' % modelPath
                raise IOError(message)
            font = StaticTextFont(PandaNode('empty'))
        if hasattr(font, 'setPointSize'):
            if pointSize is not None:
                font.setPointSize(pointSize)
            if pixelsPerUnit is not None:
                font.setPixelsPerUnit(pixelsPerUnit)
            if scaleFactor is not None:
                font.setScaleFactor(scaleFactor)
            if textureMargin is not None:
                font.setTextureMargin(textureMargin)
            if polyMargin is not None:
                font.setPolyMargin(polyMargin)
            if minFilter is not None:
                font.setMinfilter(minFilter)
            if magFilter is not None:
                font.setMagfilter(magFilter)
            if anisotropicDegree is not None:
                font.setAnisotropicDegree(anisotropicDegree)
            if color:
                font.setFg(color)
                font.setBg(VBase4(color[0], color[1], color[2], 0.0))
            if outlineWidth:
                font.setOutline(outlineColor, outlineWidth, outlineFeather)
                font.setBg(VBase4(outlineColor[0], outlineColor[1], outlineColor[2], 0.0))
            if renderMode:
                font.setRenderMode(renderMode)
        if lineHeight is not None:
            font.setLineHeight(lineHeight)
        if spaceAdvance is not None:
            font.setSpaceAdvance(spaceAdvance)
        return font

    def loadTexture(self, texturePath, alphaPath=None, readMipmaps=False, okMissing=False, minfilter=None, magfilter=None, anisotropicDegree=None, loaderOptions=None, multiview=None):
        if False:
            i = 10
            return i + 15
        "\n        texturePath is a string.\n\n        Attempt to load a texture from the given file path using\n        `TexturePool` class.  Returns a `Texture` object, or raises\n        `IOError` if the file could not be loaded.\n\n        okMissing should be True to indicate the method should return\n        None if the texture file is not found.  If it is False, the\n        method will raise an exception if the texture file is not\n        found or cannot be loaded.\n\n        If alphaPath is not None, it is the name of a grayscale image\n        that is applied as the texture's alpha channel.\n\n        If readMipmaps is True, then the filename string must contain\n        a sequence of hash characters ('#') that are filled in with\n        the mipmap index number, and n images will be loaded\n        individually which define the n mipmap levels of the texture.\n        The base level is mipmap level 0, and this defines the size of\n        the texture and the number of expected mipmap images.\n\n        If minfilter or magfilter is not None, they should be a symbol\n        like `SamplerState.FTLinear` or `SamplerState.FTNearest`.\n        (minfilter may be further one of the Mipmap filter type symbols.)\n        These specify the filter mode that will automatically be applied\n        to the texture when it is loaded.  Note that this setting may\n        override the texture's existing settings, even if it has\n        already been loaded.  See `egg-texture-cards` for a more robust\n        way to apply per-texture filter types and settings.\n\n        If anisotropicDegree is not None, it specifies the anisotropic degree\n        to apply to the texture when it is loaded.  Like minfilter and\n        magfilter, `egg-texture-cards` may be a more robust way to apply\n        this setting.\n\n        If multiview is true, it indicates to load a multiview or\n        stereo texture.  In this case, the filename should contain a\n        hash character ('#') that will be replaced with '0' for the\n        left image and '1' for the right image.  Larger numbers are\n        also allowed if you need more than two views.\n        "
        if loaderOptions is None:
            loaderOptions = LoaderOptions()
        else:
            loaderOptions = LoaderOptions(loaderOptions)
        if multiview is not None:
            flags = loaderOptions.getTextureFlags()
            if multiview:
                flags |= LoaderOptions.TFMultiview
            else:
                flags &= ~LoaderOptions.TFMultiview
            loaderOptions.setTextureFlags(flags)
        sampler = SamplerState()
        if minfilter is not None:
            sampler.setMinfilter(minfilter)
        if magfilter is not None:
            sampler.setMagfilter(magfilter)
        if anisotropicDegree is not None:
            sampler.setAnisotropicDegree(anisotropicDegree)
        if alphaPath is None:
            assert Loader.notify.debug('Loading texture: %s' % texturePath)
            texture = TexturePool.loadTexture(texturePath, 0, readMipmaps, loaderOptions, sampler)
        else:
            assert Loader.notify.debug('Loading texture: %s %s' % (texturePath, alphaPath))
            texture = TexturePool.loadTexture(texturePath, alphaPath, 0, 0, readMipmaps, loaderOptions, sampler)
        if not texture and (not okMissing):
            message = 'Could not load texture: %s' % texturePath
            raise IOError(message)
        return texture

    def load3DTexture(self, texturePattern, readMipmaps=False, okMissing=False, minfilter=None, magfilter=None, anisotropicDegree=None, loaderOptions=None, multiview=None, numViews=2):
        if False:
            for i in range(10):
                print('nop')
        "\n        texturePattern is a string that contains a sequence of one or\n        more hash characters ('#'), which will be filled in with the\n        z-height number.  Returns a 3-D `Texture` object, suitable for\n        rendering volumetric textures.\n\n        okMissing should be True to indicate the method should return\n        None if the texture file is not found.  If it is False, the\n        method will raise an exception if the texture file is not\n        found or cannot be loaded.\n\n        If readMipmaps is True, then the filename string must contain\n        two sequences of hash characters; the first group is filled in\n        with the z-height number, and the second group with the mipmap\n        index number.\n\n        If multiview is true, it indicates to load a multiview or\n        stereo texture.  In this case, numViews should also be\n        specified (the default is 2), and the sequence of texture\n        images will be divided into numViews views.  The total\n        z-height will be (numImages / numViews).  For instance, if you\n        read 16 images with numViews = 2, then you have created a\n        stereo multiview image, with z = 8.  In this example, images\n        numbered 0 - 7 will be part of the left eye view, and images\n        numbered 8 - 15 will be part of the right eye view.\n        "
        assert Loader.notify.debug('Loading 3-D texture: %s' % texturePattern)
        if loaderOptions is None:
            loaderOptions = LoaderOptions()
        else:
            loaderOptions = LoaderOptions(loaderOptions)
        if multiview is not None:
            flags = loaderOptions.getTextureFlags()
            if multiview:
                flags |= LoaderOptions.TFMultiview
            else:
                flags &= ~LoaderOptions.TFMultiview
            loaderOptions.setTextureFlags(flags)
            loaderOptions.setTextureNumViews(numViews)
        sampler = SamplerState()
        if minfilter is not None:
            sampler.setMinfilter(minfilter)
        if magfilter is not None:
            sampler.setMagfilter(magfilter)
        if anisotropicDegree is not None:
            sampler.setAnisotropicDegree(anisotropicDegree)
        texture = TexturePool.load3dTexture(texturePattern, readMipmaps, loaderOptions, sampler)
        if not texture and (not okMissing):
            message = 'Could not load 3-D texture: %s' % texturePattern
            raise IOError(message)
        return texture

    def load2DTextureArray(self, texturePattern, readMipmaps=False, okMissing=False, minfilter=None, magfilter=None, anisotropicDegree=None, loaderOptions=None, multiview=None, numViews=2):
        if False:
            for i in range(10):
                print('nop')
        "\n        texturePattern is a string that contains a sequence of one or\n        more hash characters ('#'), which will be filled in with the\n        z-height number.  Returns a 2-D `Texture` array object, suitable\n        for rendering array of textures.\n\n        okMissing should be True to indicate the method should return\n        None if the texture file is not found.  If it is False, the\n        method will raise an exception if the texture file is not\n        found or cannot be loaded.\n\n        If readMipmaps is True, then the filename string must contain\n        two sequences of hash characters; the first group is filled in\n        with the z-height number, and the second group with the mipmap\n        index number.\n\n        If multiview is true, it indicates to load a multiview or\n        stereo texture.  In this case, numViews should also be\n        specified (the default is 2), and the sequence of texture\n        images will be divided into numViews views.  The total\n        z-height will be (numImages / numViews).  For instance, if you\n        read 16 images with numViews = 2, then you have created a\n        stereo multiview image, with z = 8.  In this example, images\n        numbered 0 - 7 will be part of the left eye view, and images\n        numbered 8 - 15 will be part of the right eye view.\n        "
        assert Loader.notify.debug('Loading 2-D texture array: %s' % texturePattern)
        if loaderOptions is None:
            loaderOptions = LoaderOptions()
        else:
            loaderOptions = LoaderOptions(loaderOptions)
        if multiview is not None:
            flags = loaderOptions.getTextureFlags()
            if multiview:
                flags |= LoaderOptions.TFMultiview
            else:
                flags &= ~LoaderOptions.TFMultiview
            loaderOptions.setTextureFlags(flags)
            loaderOptions.setTextureNumViews(numViews)
        sampler = SamplerState()
        if minfilter is not None:
            sampler.setMinfilter(minfilter)
        if magfilter is not None:
            sampler.setMagfilter(magfilter)
        if anisotropicDegree is not None:
            sampler.setAnisotropicDegree(anisotropicDegree)
        texture = TexturePool.load2dTextureArray(texturePattern, readMipmaps, loaderOptions, sampler)
        if not texture and (not okMissing):
            message = 'Could not load 2-D texture array: %s' % texturePattern
            raise IOError(message)
        return texture

    def loadCubeMap(self, texturePattern, readMipmaps=False, okMissing=False, minfilter=None, magfilter=None, anisotropicDegree=None, loaderOptions=None, multiview=None):
        if False:
            i = 10
            return i + 15
        "\n        texturePattern is a string that contains a sequence of one or\n        more hash characters ('#'), which will be filled in with the\n        face index number (0 through 6).  Returns a six-face cube map\n        `Texture` object.\n\n        okMissing should be True to indicate the method should return\n        None if the texture file is not found.  If it is False, the\n        method will raise an exception if the texture file is not\n        found or cannot be loaded.\n\n        If readMipmaps is True, then the filename string must contain\n        two sequences of hash characters; the first group is filled in\n        with the face index number, and the second group with the\n        mipmap index number.\n\n        If multiview is true, it indicates to load a multiview or\n        stereo cube map.  For a stereo cube map, 12 images will be\n        loaded--images numbered 0 - 5 will become the left eye view,\n        and images 6 - 11 will become the right eye view.  In general,\n        the number of images found on disk must be a multiple of six,\n        and each six images will define a new view.\n        "
        assert Loader.notify.debug('Loading cube map: %s' % texturePattern)
        if loaderOptions is None:
            loaderOptions = LoaderOptions()
        else:
            loaderOptions = LoaderOptions(loaderOptions)
        if multiview is not None:
            flags = loaderOptions.getTextureFlags()
            if multiview:
                flags |= LoaderOptions.TFMultiview
            else:
                flags &= ~LoaderOptions.TFMultiview
            loaderOptions.setTextureFlags(flags)
        sampler = SamplerState()
        if minfilter is not None:
            sampler.setMinfilter(minfilter)
        if magfilter is not None:
            sampler.setMagfilter(magfilter)
        if anisotropicDegree is not None:
            sampler.setAnisotropicDegree(anisotropicDegree)
        texture = TexturePool.loadCubeMap(texturePattern, readMipmaps, loaderOptions, sampler)
        if not texture and (not okMissing):
            message = 'Could not load cube map: %s' % texturePattern
            raise IOError(message)
        return texture

    def unloadTexture(self, texture):
        if False:
            i = 10
            return i + 15
        '\n        Removes the previously-loaded texture from the cache, so\n        that when the last reference to it is gone, it will be\n        released.  This also means that the next time the same texture\n        is loaded, it will be re-read from disk (and duplicated in\n        texture memory if there are still outstanding references to\n        it).\n\n        The texture parameter may be the return value of any previous\n        call to loadTexture(), load3DTexture(), or loadCubeMap().\n        '
        assert Loader.notify.debug('Unloading texture: %s' % texture)
        TexturePool.releaseTexture(texture)

    def loadSfx(self, *args, **kw):
        if False:
            print('Hello World!')
        'Loads one or more sound files, specifically designated as a\n        "sound effect" file (that is, uses the sfxManager to load the\n        sound).  There is no distinction between sound effect files\n        and music files other than the particular `AudioManager` used\n        to load the sound file, but this distinction allows the sound\n        effects and/or the music files to be adjusted as a group,\n        independently of the other group.'
        if self.base.sfxManagerList:
            return self.loadSound(self.base.sfxManagerList[0], *args, **kw)
        return None

    def loadMusic(self, *args, **kw):
        if False:
            return 10
        'Loads one or more sound files, specifically designated as a\n        "music" file (that is, uses the musicManager to load the\n        sound).  There is no distinction between sound effect files\n        and music files other than the particular `AudioManager` used\n        to load the sound file, but this distinction allows the sound\n        effects and/or the music files to be adjusted as a group,\n        independently of the other group.'
        if self.base.musicManager:
            return self.loadSound(self.base.musicManager, *args, **kw)
        else:
            return None

    def loadSound(self, manager, soundPath, positional=False, callback=None, extraArgs=[]):
        if False:
            print('Hello World!')
        'Loads one or more sound files, specifying the particular\n        AudioManager that should be used to load them.  The soundPath\n        may be either a single filename, or a list of filenames.  If a\n        callback is specified, the loading happens in the background,\n        just as in loadModel(); otherwise, the loading happens before\n        loadSound() returns.'
        from panda3d.core import AudioLoadRequest
        if not isinstance(soundPath, (tuple, list, set)):
            soundList = [soundPath]
            gotList = False
        else:
            soundList = soundPath
            gotList = True
        if callback is None:
            result = []
            for soundPath in soundList:
                sound = manager.getSound(soundPath, positional)
                result.append(sound)
            if gotList:
                return result
            else:
                return result[0]
        else:
            cb = Loader._Callback(self, len(soundList), gotList, callback, extraArgs)
            for (i, soundPath) in enumerate(soundList):
                request = AudioLoadRequest(manager, soundPath, positional)
                request.setDoneEvent(self.hook)
                self.loader.loadAsync(request)
                cb.requests.add(request)
                cb.requestList.append(request)
                self._requests[request] = (cb, i)
            return cb

    def unloadSfx(self, sfx):
        if False:
            i = 10
            return i + 15
        if sfx:
            if self.base.sfxManagerList:
                self.base.sfxManagerList[0].uncacheSound(sfx.getName())

    def loadShader(self, shaderPath, okMissing=False):
        if False:
            while True:
                i = 10
        shader = ShaderPool.loadShader(shaderPath)
        if not shader and (not okMissing):
            message = 'Could not load shader file: %s' % shaderPath
            raise IOError(message)
        return shader

    def unloadShader(self, shaderPath):
        if False:
            i = 10
            return i + 15
        if shaderPath is not None:
            ShaderPool.releaseShader(shaderPath)

    def asyncFlattenStrong(self, model, inPlace=True, callback=None, extraArgs=[]):
        if False:
            while True:
                i = 10
        ' Performs a model.flattenStrong() operation in a sub-thread\n        (if threading is compiled into Panda).  The model may be a\n        single `.NodePath`, or it may be a list of NodePaths.\n\n        Each model is duplicated and flattened in the sub-thread.\n\n        If inPlace is True, then when the flatten operation completes,\n        the newly flattened copies are automatically dropped into the\n        scene graph, in place the original models.\n\n        If a callback is specified, then it is called after the\n        operation is finished, receiving the flattened model (or a\n        list of flattened models).'
        if isinstance(model, NodePath):
            modelList = [model]
            gotList = False
        else:
            modelList = model
            gotList = True
        if inPlace:
            extraArgs = [gotList, callback, modelList, extraArgs]
            callback = self.__asyncFlattenDone
            gotList = True
        cb = Loader._Callback(self, len(modelList), gotList, callback, extraArgs)
        i = 0
        for model in modelList:
            request = ModelFlattenRequest(model.node())
            request.setDoneEvent(self.hook)
            self.loader.loadAsync(request)
            cb.requests.add(request)
            cb.requestList.append(request)
            self._requests[request] = (cb, i)
            i += 1
        return cb

    def __asyncFlattenDone(self, models, gotList, callback, origModelList, extraArgs):
        if False:
            for i in range(10):
                print('nop')
        ' The asynchronous flatten operation has completed; quietly\n        drop in the new models. '
        self.notify.debug('asyncFlattenDone: %s' % (models,))
        assert len(models) == len(origModelList)
        for (i, model) in enumerate(models):
            origModelList[i].getChildren().detach()
            orig = origModelList[i].node()
            flat = model.node()
            orig.copyAllProperties(flat)
            flat.replaceNode(orig)
        if callback:
            if gotList:
                callback(origModelList, *extraArgs)
            else:
                callback(*origModelList + extraArgs)

    def __gotAsyncObject(self, request):
        if False:
            print('Hello World!')
        "A model or sound file or some such thing has just been\n        loaded asynchronously by the sub-thread.  Add it to the list\n        of loaded objects, and call the appropriate callback when it's\n        time."
        if request not in self._requests:
            return
        (cb, i) = self._requests[request]
        if cb.cancelled() or request.cancelled():
            del self._requests[request]
            return
        cb.requests.discard(request)
        if not cb.requests:
            del self._requests[request]
        result = request.result()
        if isinstance(result, PandaNode):
            result = NodePath(result)
        cb.gotObject(i, result)
    load_model = loadModel
    unload_model = unloadModel
    save_model = saveModel
    load_font = loadFont
    load_texture = loadTexture
    load_3d_texture = load3DTexture
    load_cube_map = loadCubeMap
    unload_texture = unloadTexture
    load_sfx = loadSfx
    load_music = loadMusic
    load_sound = loadSound
    unload_sfx = unloadSfx
    load_shader = loadShader
    unload_shader = unloadShader
    async_flatten_strong = asyncFlattenStrong