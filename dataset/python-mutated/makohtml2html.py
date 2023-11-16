import logging
import os
import mako
from mako.lookup import TemplateLookup
from mako.template import Template
from lxml import etree
_logger = logging.getLogger(__name__)

class makohtml2html(object):

    def __init__(self, html, localcontext):
        if False:
            print('Hello World!')
        self.localcontext = localcontext
        self.html = html

    def format_header(self, html):
        if False:
            i = 10
            return i + 15
        head = html.findall('head')
        header = ''
        for node in head:
            header += etree.tostring(node)
        return header

    def format_footer(self, footer):
        if False:
            while True:
                i = 10
        html_footer = ''
        for node in footer[0].getchildren():
            html_footer += etree.tostring(node)
        return html_footer

    def format_body(self, html):
        if False:
            while True:
                i = 10
        body = html.findall('body')
        body_list = []
        footer = self.format_footer(body[-1].getchildren())
        for b in body[:-1]:
            body_list.append(etree.tostring(b).replace('\t', '').replace('\n', ''))
        html_body = '\n        <script type="text/javascript">\n\n        var indexer = 0;\n        var aryTest = %s ;\n        function nextData()\n            {\n            if(indexer < aryTest.length -1)\n                {\n                indexer += 1;\n                document.forms[0].prev.disabled = false;\n                document.getElementById("openerp_data").innerHTML=aryTest[indexer];\n                document.getElementById("counter").innerHTML= indexer + 1 + \' / \' + aryTest.length;\n                }\n            else\n               {\n                document.forms[0].next.disabled = true;\n               }\n            }\n        function prevData()\n            {\n            if (indexer > 0)\n                {\n                indexer -= 1;\n                document.forms[0].next.disabled = false;\n                document.getElementById("openerp_data").innerHTML=aryTest[indexer];\n                document.getElementById("counter").innerHTML=  indexer + 1 + \' / \' + aryTest.length;\n                }\n            else\n               {\n                document.forms[0].prev.disabled = true;\n               }\n            }\n    </script>\n    </head>\n    <body>\n        <div id="openerp_data">\n            %s\n        </div>\n        <div>\n        %s\n        </div>\n        <br>\n        <form>\n            <table>\n                <tr>\n                    <td td align="left">\n                        <input name = "prev" type="button" value="Previous" onclick="prevData();">\n                    </td>\n                    <td>\n                        <div id = "counter">%s / %s</div>\n                    </td>\n                    <td align="right">\n                        <input name = "next" type="button" value="Next" onclick="nextData();">\n                    </td>\n                </tr>\n            </table>\n        </form>\n    </body></html>' % (body_list, body_list[0], footer, '1', len(body_list))
        return html_body

    def render(self):
        if False:
            return 10
        path = os.path.realpath('addons/base/report')
        temp_lookup = TemplateLookup(directories=[path], output_encoding='utf-8', encoding_errors='replace')
        template = Template(self.html, lookup=temp_lookup)
        self.localcontext.update({'css_path': path})
        final_html = '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">\n                    <html>'
        try:
            html = template.render_unicode(**self.localcontext)
            etree_obj = etree.HTML(html)
            final_html += self.format_header(etree_obj)
            final_html += self.format_body(etree_obj)
            return final_html
        except Exception:
            _logger.exception('report :')

def parseNode(html, localcontext={}):
    if False:
        print('Hello World!')
    r = makohtml2html(html, localcontext)
    return r.render()