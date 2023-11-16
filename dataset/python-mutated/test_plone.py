from functools import partial
from ..utils import isort_test
plone_isort_test = partial(isort_test, profile='plone')

def test_plone_code_snippet_one():
    if False:
        i = 10
        return i + 15
    plone_isort_test('# -*- coding: utf-8 -*-\nfrom plone.app.multilingual.testing import PLONE_APP_MULTILINGUAL_PRESET_FIXTURE  # noqa\nfrom plone.app.robotframework.testing import REMOTE_LIBRARY_BUNDLE_FIXTURE\nfrom plone.app.testing import FunctionalTesting\nfrom plone.app.testing import IntegrationTesting\nfrom plone.app.testing import PloneWithPackageLayer\nfrom plone.testing import z2\n\nimport plone.app.multilingualindexes\n\n\nPAMI_FIXTURE = PloneWithPackageLayer(\n    bases=(PLONE_APP_MULTILINGUAL_PRESET_FIXTURE,),\n    name="PAMILayer:Fixture",\n    gs_profile_id="plone.app.multilingualindexes:default",\n    zcml_package=plone.app.multilingualindexes,\n    zcml_filename="configure.zcml",\n    additional_z2_products=["plone.app.multilingualindexes"],\n)\n')

def test_plone_code_snippet_two():
    if False:
        print('Hello World!')
    plone_isort_test('# -*- coding: utf-8 -*-\nfrom Acquisition import aq_base\nfrom App.class_init import InitializeClass\nfrom App.special_dtml import DTMLFile\nfrom BTrees.OOBTree import OOTreeSet\nfrom logging import getLogger\nfrom plone import api\nfrom plone.app.multilingual.events import ITranslationRegisteredEvent\nfrom plone.app.multilingual.interfaces import ITG\nfrom plone.app.multilingual.interfaces import ITranslatable\nfrom plone.app.multilingual.interfaces import ITranslationManager\nfrom plone.app.multilingualindexes.utils import get_configuration\nfrom plone.indexer.interfaces import IIndexableObject\nfrom Products.CMFPlone.utils import safe_hasattr\nfrom Products.DateRecurringIndex.index import DateRecurringIndex\nfrom Products.PluginIndexes.common.UnIndex import UnIndex\nfrom Products.ZCatalog.Catalog import Catalog\nfrom ZODB.POSException import ConflictError\nfrom zope.component import getMultiAdapter\nfrom zope.component import queryAdapter\nfrom zope.globalrequest import getRequest\n\n\nlogger = getLogger(__name__)\n')

def test_plone_code_snippet_three():
    if False:
        return 10
    plone_isort_test('# -*- coding: utf-8 -*-\nfrom plone.app.querystring.interfaces import IQueryModifier\nfrom zope.interface import provider\n\nimport logging\n\n\nlogger = logging.getLogger(__name__)\n\n')