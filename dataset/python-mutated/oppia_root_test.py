"""Tests for the oppia root page."""
from __future__ import annotations
from core.constants import constants
from core.tests import test_utils

class OppiaRootPageTests(test_utils.GenericTestBase):

    def test_oppia_root_page(self) -> None:
        if False:
            return 10
        'Tests access to the unified entry page.'
        for page in constants.PAGES_REGISTERED_WITH_FRONTEND.values():
            if not 'MANUALLY_REGISTERED_WITH_BACKEND' in page:
                response = self.get_html_response('/%s' % page['ROUTE'], expected_status_int=200)
                if 'LIGHTWEIGHT' in page:
                    response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>')
                else:
                    response.mustcontain('<oppia-root></oppia-root>')

class OppiaLightweightRootPageTests(test_utils.GenericTestBase):

    def test_oppia_lightweight_root_page(self) -> None:
        if False:
            print('Hello World!')
        response = self.get_html_response('/', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', '<title>Loading | Oppia</title>')

    def test_oppia_lightweight_root_page_with_rtl_lang_param(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        response = self.get_html_response('/?dir=rtl', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', no='<title>Loading | Oppia</title>')

    def test_oppia_lightweight_root_page_with_ltr_lang_param(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        response = self.get_html_response('/?dir=ltr', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', '<title>Loading | Oppia</title>')

    def test_oppia_lightweight_root_page_with_rtl_dir_cookie(self) -> None:
        if False:
            i = 10
            return i + 15
        self.testapp.set_cookie('dir', 'rtl')
        response = self.get_html_response('/', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', no='<title>Loading | Oppia</title>')

    def test_oppia_lightweight_root_page_with_ltr_dir_cookie(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.testapp.set_cookie('dir', 'ltr')
        response = self.get_html_response('/', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', '<title>Loading | Oppia</title>')

    def test_return_bundle_modifier_precedence(self) -> None:
        if False:
            i = 10
            return i + 15
        self.testapp.set_cookie('dir', 'ltr')
        response = self.get_html_response('/?dir=rtl', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', '<title>Loading | Oppia</title>')
        self.testapp.set_cookie('dir', 'rtl')
        response = self.get_html_response('/?dir=ltr', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', no='<title>Loading | Oppia</title>')

    def test_invalid_bundle_modifier_values(self) -> None:
        if False:
            return 10
        self.testapp.set_cookie('dir', 'new_hacker_in_the_block')
        response = self.get_html_response('/?dir=rtl', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', no='<title>Loading | Oppia</title>')
        self.testapp.set_cookie('dir', 'new_hacker_in_the_block')
        response = self.get_html_response('/?dir=ltr', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', '<title>Loading | Oppia</title>')
        self.testapp.set_cookie('dir', 'new_hacker_in_the_block')
        response = self.get_html_response('/?dir=is_trying_out', expected_status_int=200)
        response.mustcontain('<lightweight-oppia-root></lightweight-oppia-root>', '<title>Loading | Oppia</title>')