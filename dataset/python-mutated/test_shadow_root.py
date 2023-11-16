import pytest
from .support import PyScriptTest

class TestShadowRoot(PyScriptTest):

    @pytest.mark.skip('NEXT: Element interface is gone. Replace with PyDom')
    def test_reachable_shadow_root(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script>\n                // reason to wait for py-script is that it\'s the entry point for\n                // all patches and the MutationObserver, otherwise being this a synchronous\n                // script the constructor gets instantly invoked at the node before\n                // py-script gets a chance to initialize itself.\n                customElements.whenDefined(\'py-script\').then(() => {\n                    customElements.define(\'s-r\', class extends HTMLElement {\n                      constructor() {\n                        super().attachShadow({mode: \'closed\'}).innerHTML =\n                            \'<div id="shadowed">OK</div>\';\n                      }\n                  });\n                });\n            </script>\n            <s-r></s-r>\n            <script type="py">\n                import js\n                js.console.log(Element("shadowed").innerHtml)\n            </script>\n            ')
        assert self.console.log.lines[-1] == 'OK'