""" Nodes the represent ways to access metadata pkg_resources, importlib.resources etc.

"""
from nuitka.Constants import isCompileTimeConstantValue
from nuitka.Options import isStandaloneMode, shallMakeModule
from nuitka.Tracing import inclusion_logger
from nuitka.utils.Importing import importFromCompileTime
from nuitka.utils.Utils import withNoDeprecationWarning
from .AttributeNodes import makeExpressionAttributeLookup
from .ContainerMakingNodes import ExpressionMakeSequenceMixin, makeExpressionMakeList, makeExpressionMakeTuple
from .DictionaryNodes import ExpressionMakeDictMixin, makeExpressionMakeDict
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .ExpressionBasesGenerated import ExpressionImportlibMetadataBackportEntryPointsValueRefBase, ExpressionImportlibMetadataBackportEntryPointValueRefBase, ExpressionImportlibMetadataBackportSelectableGroupsValueRefBase, ExpressionImportlibMetadataDistributionFailedCallBase, ExpressionImportlibMetadataEntryPointsValueRefBase, ExpressionImportlibMetadataEntryPointValueRefBase, ExpressionImportlibMetadataSelectableGroupsValueRefBase
from .HardImportNodesGenerated import ExpressionImportlibMetadataBackportEntryPointsCallBase, ExpressionImportlibMetadataBackportVersionCallBase, ExpressionImportlibMetadataDistributionCallBase, ExpressionImportlibMetadataEntryPointsBefore310CallBase, ExpressionImportlibMetadataEntryPointsSince310CallBase, ExpressionImportlibMetadataVersionCallBase, ExpressionPkgResourcesGetDistributionCallBase, ExpressionPkgResourcesIterEntryPointsCallBase, ExpressionPkgResourcesRequireCallBase
from .KeyValuePairNodes import makeExpressionKeyValuePairConstantKey, makeKeyValuePairExpressionsFromKwArgs

def _getPkgResourcesModule():
    if False:
        while True:
            i = 10
    'Helper for importing pkg_resources from installation at compile time.\n\n    This is not for using the inline copy, but the one from the actual\n    installation of the user. It suppresses warnings and caches the value\n    avoid making more __import__ calls that necessary.\n    '
    return importFromCompileTime('pkg_resources', must_exist=True)

class ExpressionPkgResourcesRequireCall(ExpressionPkgResourcesRequireCallBase):
    kind = 'EXPRESSION_PKG_RESOURCES_REQUIRE_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        require = _getPkgResourcesModule().require
        ResolutionError = _getPkgResourcesModule().ResolutionError
        InvalidRequirement = _getPkgResourcesModule().extern.packaging.requirements.InvalidRequirement
        args = tuple((element.getCompileTimeConstant() for element in self.subnode_requirements))
        try:
            distributions = require(*args)
        except ResolutionError:
            inclusion_logger.warning("Cannot find requirement %s at '%s', expect potential run time problem, unless this is unused code." % (','.join((repr(s) for s in args)), self.source_ref.getAsString()))
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        except (TypeError, InvalidRequirement):
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        except Exception as e:
            inclusion_logger.sysexit("Error, failed to find requirements '%s' at '%s' due to unhandled %s. Please report this bug." % (','.join((repr(s) for s in args)), self.source_ref.getAsString(), repr(e)))
        else:
            result = makeExpressionMakeList(elements=tuple((ExpressionPkgResourcesDistributionValueRef(distribution=distribution, source_ref=self.source_ref) for distribution in distributions)), source_ref=self.source_ref)
            trace_collection.onExceptionRaiseExit(BaseException)
            return (result, 'new_expression', "Compile time predicted 'pkg_resources.require' result")

class ExpressionPkgResourcesGetDistributionCall(ExpressionPkgResourcesGetDistributionCallBase):
    kind = 'EXPRESSION_PKG_RESOURCES_GET_DISTRIBUTION_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            while True:
                i = 10
        get_distribution = _getPkgResourcesModule().get_distribution
        DistributionNotFound = _getPkgResourcesModule().DistributionNotFound
        arg = self.subnode_dist.getCompileTimeConstant()
        try:
            distribution = get_distribution(arg)
        except DistributionNotFound:
            trace_collection.onDistributionUsed(distribution_name=arg, node=self, success=False)
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        except Exception as e:
            inclusion_logger.sysexit("Error, failed to find distribution '%s' at '%s' due to unhandled %s. Please report this bug." % (arg, self.source_ref.getAsString(), repr(e)))
        else:
            trace_collection.onDistributionUsed(distribution_name=arg, node=self, success=True)
            result = ExpressionPkgResourcesDistributionValueRef(distribution=distribution, source_ref=self.source_ref)
            return (result, 'new_expression', "Compile time predicted 'pkg_resources.get_distribution' result")

class ImportlibMetadataVersionCallMixin(object):
    __slots__ = ()

    def _getImportlibMetadataModule(self):
        if False:
            for i in range(10):
                print('nop')
        return importFromCompileTime(self.importlib_metadata_name, must_exist=True)

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            return 10
        version = self._getImportlibMetadataModule().version
        PackageNotFoundError = self._getImportlibMetadataModule().PackageNotFoundError
        arg = self.subnode_distribution_name.getCompileTimeConstant()
        try:
            distribution = version(arg)
        except PackageNotFoundError:
            trace_collection.onDistributionUsed(distribution_name=arg, node=self, success=False)
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        except Exception as e:
            inclusion_logger.sysexit("Error, failed to find distribution '%s' at '%s' due to unhandled %s. Please report this bug." % (arg, self.source_ref.getAsString(), repr(e)))
        else:
            trace_collection.onDistributionUsed(distribution_name=arg, node=self, success=True)
            from .ConstantRefNodes import makeConstantRefNode
            result = makeConstantRefNode(constant=distribution, source_ref=self.source_ref)
            return (result, 'new_expression', "Compile time predicted '%s.version' result" % self.importlib_metadata_name)

class ExpressionImportlibMetadataVersionCall(ImportlibMetadataVersionCallMixin, ExpressionImportlibMetadataVersionCallBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_VERSION_CALL'
    python_version_spec = '>= 0x380'
    importlib_metadata_name = 'importlib.metadata'

class ExpressionImportlibMetadataBackportVersionCall(ImportlibMetadataVersionCallMixin, ExpressionImportlibMetadataBackportVersionCallBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_BACKPORT_VERSION_CALL'
    importlib_metadata_name = 'importlib_metadata'

class ExpressionPkgResourcesDistributionValueRef(ExpressionNoSideEffectsMixin, ExpressionBase):
    kind = 'EXPRESSION_PKG_RESOURCES_DISTRIBUTION_VALUE_REF'
    __slots__ = ('distribution', 'computed_attributes')
    preserved_attributes = ('py_version', 'platform', 'version', 'project_name')

    def __init__(self, distribution, source_ref):
        if False:
            i = 10
            return i + 15
        with withNoDeprecationWarning():
            Distribution = _getPkgResourcesModule().Distribution
            preserved_attributes = self.preserved_attributes
            if not isStandaloneMode():
                preserved_attributes += ('location',)
            distribution = Distribution(**dict(((key, getattr(distribution, key)) for key in preserved_attributes)))
        ExpressionBase.__init__(self, source_ref)
        self.distribution = distribution
        self.computed_attributes = {}

    def finalize(self):
        if False:
            return 10
        del self.distribution

    @staticmethod
    def isKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def getTruthValue():
        if False:
            return 10
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            return 10
        return False

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return (self, None, None)

    def isKnownToHaveAttribute(self, attribute_name):
        if False:
            for i in range(10):
                print('nop')
        if attribute_name not in self.computed_attributes:
            self.computed_attributes[attribute_name] = hasattr(self.distribution, attribute_name)
        return self.computed_attributes[attribute_name]

    def getKnownAttributeValue(self, attribute_name):
        if False:
            while True:
                i = 10
        return getattr(self.distribution, attribute_name)

    def computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection):
        if False:
            while True:
                i = 10
        if self.isKnownToHaveAttribute(attribute_name) and isCompileTimeConstantValue(getattr(self.distribution, attribute_name, None)) and (attribute_name != 'location' or not isStandaloneMode()):
            return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : getattr(self.distribution, attribute_name), description="Attribute '%s' pre-computed." % attribute_name)
        return (lookup_node, None, None)

    def mayRaiseExceptionAttributeLookup(self, exception_type, attribute_name):
        if False:
            while True:
                i = 10
        return not self.isKnownToHaveAttribute(attribute_name)

class ExpressionImportlibMetadataDistributionValueRef(ExpressionNoSideEffectsMixin, ExpressionBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_DISTRIBUTION_VALUE_REF'
    __slots__ = ('distribution', 'original_name', 'computed_attributes')

    def __init__(self, distribution, original_name, source_ref):
        if False:
            print('Hello World!')
        ExpressionBase.__init__(self, source_ref)
        self.distribution = distribution
        self.original_name = original_name
        self.computed_attributes = {}

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'distribution': self.distribution, 'original_name': self.original_name}

    def finalize(self):
        if False:
            return 10
        del self.distribution

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def getTruthValue():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            while True:
                i = 10
        return False

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

class ExpressionPkgResourcesIterEntryPointsCall(ExpressionPkgResourcesIterEntryPointsCallBase):
    kind = 'EXPRESSION_PKG_RESOURCES_ITER_ENTRY_POINTS_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            i = 10
            return i + 15
        iter_entry_points = _getPkgResourcesModule().iter_entry_points
        DistributionNotFound = _getPkgResourcesModule().DistributionNotFound
        group = self.subnode_group.getCompileTimeConstant()
        if self.subnode_name is not None:
            name = self.subnode_name.getCompileTimeConstant()
        else:
            name = None
        try:
            entry_points = tuple(iter_entry_points(group=group, name=name))
        except DistributionNotFound:
            trace_collection.onDistributionUsed(distribution_name=name or group, node=self, success=False)
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        except Exception as e:
            inclusion_logger.sysexit("Error, failed to find distribution '%s' at '%s' due to unhandled %s. Please report this bug." % (name, self.source_ref.getAsString(), repr(e)))
        else:
            trace_collection.onDistributionUsed(distribution_name=name or group, node=self, success=True)
            result = makeExpressionMakeList(elements=tuple((ExpressionPkgResourcesEntryPointValueRef(entry_point=entry_point, source_ref=self.source_ref) for entry_point in entry_points)), source_ref=self.source_ref)
            return (result, 'new_expression', "Compile time predicted 'pkg_resources.iter_entry_points' result")

class ExpressionPkgResourcesEntryPointValueRef(ExpressionNoSideEffectsMixin, ExpressionBase):
    kind = 'EXPRESSION_PKG_RESOURCES_ENTRY_POINT_VALUE_REF'
    __slots__ = ('entry_point', 'computed_attributes')
    preserved_attributes = ('name', 'module_name', 'attrs', 'extras')

    def __init__(self, entry_point, source_ref):
        if False:
            return 10
        with withNoDeprecationWarning():
            EntryPoint = _getPkgResourcesModule().EntryPoint
            entry_point = EntryPoint(**dict(((key, getattr(entry_point, key)) for key in self.preserved_attributes)))
        ExpressionBase.__init__(self, source_ref)
        self.entry_point = entry_point
        self.computed_attributes = {}

    def finalize(self):
        if False:
            return 10
        del self.entry_point
        del self.computed_attributes

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def getTruthValue():
        if False:
            for i in range(10):
                print('nop')
        return True

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

    def isKnownToHaveAttribute(self, attribute_name):
        if False:
            while True:
                i = 10
        if attribute_name not in self.computed_attributes:
            self.computed_attributes[attribute_name] = hasattr(self.entry_point, attribute_name)
        return self.computed_attributes[attribute_name]

    def getKnownAttributeValue(self, attribute_name):
        if False:
            while True:
                i = 10
        return getattr(self.entry_point, attribute_name)

    def computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection):
        if False:
            while True:
                i = 10
        if self.isKnownToHaveAttribute(attribute_name) and isCompileTimeConstantValue(getattr(self.entry_point, attribute_name, None)):
            return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : getattr(self.entry_point, attribute_name), description="Attribute '%s' pre-computed." % attribute_name)
        return (lookup_node, None, None)

class ImportlibMetadataDistributionCallMixin(object):
    __slots__ = ()

    def _getImportlibMetadataModule(self):
        if False:
            for i in range(10):
                print('nop')
        return importFromCompileTime(self.importlib_metadata_name, must_exist=True)

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            i = 10
            return i + 15
        if shallMakeModule():
            return
        distribution_func = self._getImportlibMetadataModule().distribution
        PackageNotFoundError = self._getImportlibMetadataModule().PackageNotFoundError
        arg = self.subnode_distribution_name.getCompileTimeConstant()
        try:
            distribution = distribution_func(arg)
        except PackageNotFoundError:
            return trace_collection.computedExpressionResult(expression=self.makeExpressionImportlibMetadataDistributionFailedCall(), change_tags='new_expression', change_desc="Call to '%s.distribution' failed to resolve." % self.importlib_metadata_name)
        except Exception as e:
            inclusion_logger.sysexit("Error, failed to find distribution '%s' at '%s' due to unhandled %s. Please report this bug." % (arg, self.source_ref.getAsString(), repr(e)))
        else:
            trace_collection.onDistributionUsed(distribution_name=arg, node=self, success=False)
            result = ExpressionImportlibMetadataDistributionValueRef(distribution=distribution, original_name=arg, source_ref=self.source_ref)
            return (result, 'new_expression', "Compile time predicted '%s.distribution' result" % self.importlib_metadata_name)

class ExpressionImportlibMetadataDistributionCall(ImportlibMetadataDistributionCallMixin, ExpressionImportlibMetadataDistributionCallBase):
    """Represents call to importlib.metadata.distribution(distribution_name)"""
    kind = 'EXPRESSION_IMPORTLIB_METADATA_DISTRIBUTION_CALL'
    python_version_spec = '>= 0x380'
    importlib_metadata_name = 'importlib.metadata'

    def makeExpressionImportlibMetadataDistributionFailedCall(self):
        if False:
            for i in range(10):
                print('nop')
        return ExpressionImportlibMetadataDistributionFailedCall(distribution_name=self.subnode_distribution_name, source_ref=self.source_ref)

class ExpressionImportlibMetadataDistributionFailedCallMixin(object):
    __slots__ = ()

    def computeExpression(self, trace_collection):
        if False:
            return 10
        distribution_name = self.subnode_distribution_name.getCompileTimeConstant()
        trace_collection.onDistributionUsed(distribution_name=distribution_name, node=self, success=False)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            while True:
                i = 10
        return True

class ExpressionImportlibMetadataDistributionFailedCall(ExpressionImportlibMetadataDistributionFailedCallMixin, ExpressionImportlibMetadataDistributionFailedCallBase):
    """Represents compile time failed call to importlib.metadata.distribution(distribution_name)"""
    kind = 'EXPRESSION_IMPORTLIB_METADATA_DISTRIBUTION_FAILED_CALL'
    named_children = ('distribution_name',)
    auto_compute_handling = 'final_children'
    python_version_spec = '>= 0x380'
    importlib_metadata_name = 'importlib.metadata'

class ExpressionImportlibMetadataBackportDistributionFailedCall(ExpressionImportlibMetadataDistributionFailedCallMixin, ExpressionImportlibMetadataDistributionFailedCallBase):
    """Represents compile time failed call to importlib_metadata.distribution(distribution_name)"""
    kind = 'EXPRESSION_IMPORTLIB_METADATA_BACKPORT_DISTRIBUTION_FAILED_CALL'
    named_children = ('distribution_name',)
    auto_compute_handling = 'final_children'
    importlib_metadata_name = 'importlib_metadata'

class ExpressionImportlibMetadataBackportDistributionCall(ImportlibMetadataDistributionCallMixin, ExpressionImportlibMetadataDistributionCallBase):
    """Represents call to importlib_metadata.distribution(distribution_name)"""
    kind = 'EXPRESSION_IMPORTLIB_METADATA_BACKPORT_DISTRIBUTION_CALL'
    importlib_metadata_name = 'importlib_metadata'

    def makeExpressionImportlibMetadataDistributionFailedCall(self):
        if False:
            i = 10
            return i + 15
        return ExpressionImportlibMetadataBackportDistributionFailedCall(distribution_name=self.subnode_distribution_name, source_ref=self.source_ref)

def makeExpressionImportlibMetadataMetadataCall(distribution_name, source_ref):
    if False:
        return 10
    return makeExpressionAttributeLookup(expression=ExpressionImportlibMetadataDistributionCall(distribution_name=distribution_name, source_ref=source_ref), attribute_name='metadata', source_ref=source_ref)

def makeExpressionImportlibMetadataBackportMetadataCall(distribution_name, source_ref):
    if False:
        i = 10
        return i + 15
    return makeExpressionAttributeLookup(expression=ExpressionImportlibMetadataBackportDistributionCall(distribution_name=distribution_name, source_ref=source_ref), attribute_name='metadata', source_ref=source_ref)

class ExpressionImportlibMetadataEntryPointValueMixin(object):
    __slots__ = ()
    preserved_attributes = ('name', 'value', 'group')

    def _getImportlibMetadataModule(self):
        if False:
            print('Hello World!')
        return importFromCompileTime(self.importlib_metadata_name, must_exist=True)

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.entry_point
        del self.computed_attributes

    @staticmethod
    def isKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def getTruthValue():
        if False:
            while True:
                i = 10
        return True

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return (self, None, None)

    def isKnownToHaveAttribute(self, attribute_name):
        if False:
            for i in range(10):
                print('nop')
        if attribute_name not in self.computed_attributes:
            self.computed_attributes[attribute_name] = hasattr(self.entry_point, attribute_name)
        return self.computed_attributes[attribute_name]

    def getKnownAttributeValue(self, attribute_name):
        if False:
            i = 10
            return i + 15
        return getattr(self.entry_point, attribute_name)

    def computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection):
        if False:
            i = 10
            return i + 15
        if self.isKnownToHaveAttribute(attribute_name) and isCompileTimeConstantValue(getattr(self.entry_point, attribute_name, None)):
            return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : getattr(self.entry_point, attribute_name), description="Attribute '%s' pre-computed." % attribute_name)
        return (lookup_node, None, None)

class ExpressionImportlibMetadataEntryPointValueRef(ExpressionNoSideEffectsMixin, ExpressionImportlibMetadataEntryPointValueMixin, ExpressionImportlibMetadataEntryPointValueRefBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_ENTRY_POINT_VALUE_REF'
    python_version_spec = '>= 0x380'
    __slots__ = ('entry_point', 'computed_attributes')
    auto_compute_handling = 'final,no_raise'
    importlib_metadata_name = 'importlib.metadata'

    def __init__(self, entry_point, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionImportlibMetadataEntryPointValueRefBase.__init__(self, source_ref)
        EntryPoint = self._getImportlibMetadataModule().EntryPoint
        entry_point = EntryPoint(**dict(((key, getattr(entry_point, key)) for key in self.preserved_attributes)))
        self.entry_point = entry_point
        self.computed_attributes = {}

class ExpressionImportlibMetadataBackportEntryPointValueRef(ExpressionNoSideEffectsMixin, ExpressionImportlibMetadataEntryPointValueMixin, ExpressionImportlibMetadataBackportEntryPointValueRefBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_BACKPORT_ENTRY_POINT_VALUE_REF'
    __slots__ = ('entry_point', 'computed_attributes')
    auto_compute_handling = 'final,no_raise'
    importlib_metadata_name = 'importlib_metadata'

    def __init__(self, entry_point, source_ref):
        if False:
            print('Hello World!')
        ExpressionImportlibMetadataBackportEntryPointValueRefBase.__init__(self, source_ref)
        EntryPoint = self._getImportlibMetadataModule().EntryPoint
        entry_point = EntryPoint(**dict(((key, getattr(entry_point, key)) for key in self.preserved_attributes)))
        self.entry_point = entry_point
        self.computed_attributes = {}

class ExpressionImportlibMetadataSelectableGroupsValueRef(ExpressionMakeDictMixin, ExpressionImportlibMetadataSelectableGroupsValueRefBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_SELECTABLE_GROUPS_VALUE_REF'
    python_version_spec = '>= 0x3a0'
    named_children = ('pairs|tuple',)
    auto_compute_handling = 'final,no_raise'

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionImportlibMetadataBackportSelectableGroupsValueRef(ExpressionMakeDictMixin, ExpressionImportlibMetadataBackportSelectableGroupsValueRefBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_BACKPORT_SELECTABLE_GROUPS_VALUE_REF'
    named_children = ('pairs|tuple',)
    auto_compute_handling = 'final,no_raise'

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return False

class ExpressionImportlibMetadataEntryPointsValueRef(ExpressionMakeSequenceMixin, ExpressionImportlibMetadataBackportEntryPointsValueRefBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_ENTRY_POINTS_VALUE_REF'
    python_version_spec = '>= 0x3a0'
    named_children = ('elements|tuple',)
    auto_compute_handling = 'final,no_raise'

    @staticmethod
    def isKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def getSequenceName():
        if False:
            i = 10
            return i + 15
        'Get name for use in traces'
        return 'importlib.metadata.EntryPoints'

class ExpressionImportlibMetadataBackportEntryPointsValueRef(ExpressionMakeSequenceMixin, ExpressionImportlibMetadataEntryPointsValueRefBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_BACKPORT_ENTRY_POINTS_VALUE_REF'
    named_children = ('elements|tuple',)
    auto_compute_handling = 'final,no_raise'

    @staticmethod
    def getSequenceName():
        if False:
            return 10
        'Get name for use in traces'
        return 'importlib_metadata.EntryPoints'

    @staticmethod
    def isKnownToBeHashable():
        if False:
            while True:
                i = 10
        return False

class ExpressionImportlibMetadataEntryPointsCallMixin(object):
    __slots__ = ()

    def _getImportlibMetadataModule(self):
        if False:
            i = 10
            return i + 15
        return importFromCompileTime(self.importlib_metadata_name, must_exist=True)

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            print('Hello World!')
        metadata_importlib = self._getImportlibMetadataModule()
        constant_args = dict(((param.getKeyCompileTimeConstant(), param.getValueCompileTimeConstant()) for param in self.subnode_params))
        try:
            entry_points_result = metadata_importlib.entry_points(**constant_args)
        except Exception as e:
            inclusion_logger.sysexit("Error, failed to find entrypoints at '%s' due to unhandled %s. Please report this bug." % (self.source_ref.getAsString(), repr(e)))
        else:
            if hasattr(metadata_importlib, 'SelectableGroups') and type(entry_points_result) is metadata_importlib.SelectableGroups:
                pairs = [makeExpressionKeyValuePairConstantKey(key=key, value=self.makeEntryPointsValueRef(elements=tuple((self.makeEntryPointValueRef(entry_point=entry_point, source_ref=self.source_ref) for entry_point in value)), source_ref=self.source_ref)) for (key, value) in entry_points_result.items()]
                result = self.makeSelectableGroupsValueRef(pairs=tuple(pairs), source_ref=self.source_ref)
            elif type(entry_points_result) is dict:
                pairs = [makeExpressionKeyValuePairConstantKey(key=key, value=makeExpressionMakeTuple(elements=tuple((self.makeEntryPointValueRef(entry_point=entry_point, source_ref=self.source_ref) for entry_point in value)), source_ref=self.source_ref)) for (key, value) in entry_points_result.items()]
                result = makeExpressionMakeDict(pairs=tuple(pairs), source_ref=self.source_ref)
            elif hasattr(metadata_importlib, 'EntryPoints') and type(entry_points_result) is metadata_importlib.EntryPoints:
                result = self.makeEntryPointsValueRef(elements=tuple((self.makeEntryPointValueRef(entry_point=entry_point, source_ref=self.source_ref) for entry_point in entry_points_result)), source_ref=self.source_ref)
            else:
                assert False, type(entry_points_result)
            return (result, 'new_expression', "Compile time predicted '%s' result" % self.importlib_metadata_name)

class ExpressionImportlibMetadataEntryPointsBefore310Call(ExpressionImportlibMetadataEntryPointsCallMixin, ExpressionImportlibMetadataEntryPointsBefore310CallBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_ENTRY_POINTS_BEFORE310_CALL'
    python_version_spec = '>= 0x380'
    importlib_metadata_name = 'importlib.metadata'
    makeEntryPointValueRef = ExpressionImportlibMetadataEntryPointValueRef
    subnode_params = ()

def makeExpressionImportlibMetadataEntryPointsSince310Call(params, source_ref):
    if False:
        print('Hello World!')
    return ExpressionImportlibMetadataEntryPointsSince310Call(params=makeKeyValuePairExpressionsFromKwArgs(params), source_ref=source_ref)

class ExpressionImportlibMetadataEntryPointsSince310Call(ExpressionImportlibMetadataEntryPointsCallMixin, ExpressionImportlibMetadataEntryPointsSince310CallBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_ENTRY_POINTS_SINCE310_CALL'
    importlib_metadata_name = 'importlib.metadata'
    makeEntryPointsValueRef = ExpressionImportlibMetadataEntryPointsValueRef
    makeEntryPointValueRef = ExpressionImportlibMetadataEntryPointValueRef
    makeSelectableGroupsValueRef = ExpressionImportlibMetadataSelectableGroupsValueRef

def makeExpressionImportlibMetadataBackportEntryPointsCall(params, source_ref):
    if False:
        print('Hello World!')
    return ExpressionImportlibMetadataBackportEntryPointsCall(params=makeKeyValuePairExpressionsFromKwArgs(params), source_ref=source_ref)

class ExpressionImportlibMetadataBackportEntryPointsCall(ExpressionImportlibMetadataEntryPointsCallMixin, ExpressionImportlibMetadataBackportEntryPointsCallBase):
    kind = 'EXPRESSION_IMPORTLIB_METADATA_BACKPORT_ENTRY_POINTS_CALL'
    importlib_metadata_name = 'importlib_metadata'
    makeEntryPointsValueRef = ExpressionImportlibMetadataBackportEntryPointsValueRef
    makeEntryPointValueRef = ExpressionImportlibMetadataBackportEntryPointValueRef
    makeSelectableGroupsValueRef = ExpressionImportlibMetadataBackportSelectableGroupsValueRef