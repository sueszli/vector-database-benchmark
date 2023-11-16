import re
from .compat import HTMLParser
from .lib.html2text.html2text import HTML2Text
import click
import requests

class WebViewer(object):
    """Handle viewing of web content within the terminal.

    :type html: :class:`HTMLParser.HTMLParser`
    :param html: An instance of `HTMLParser.HTMLParser`.

    :type html_to_text: :class:`html2text.html2text.HTML2Text`
    :param html_to_text: An instance of `html2text.html2text.HTML2Text`.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        try:
            self.html = HTMLParser.HTMLParser()
        except:
            self.html = HTMLParser
        self.html_to_text = None
        self._init_html_to_text()

    def _init_html_to_text(self):
        if False:
            print('Hello World!')
        'Initialize HTML2Text.'
        self.html_to_text = HTML2Text()
        self.html_to_text.body_width = 0
        self.html_to_text.ignore_images = False
        self.html_to_text.ignore_emphasis = False
        self.html_to_text.ignore_links = False
        self.html_to_text.skip_internal_links = False
        self.html_to_text.inline_links = False
        self.html_to_text.links_each_paragraph = False

    def format_markdown(self, text):
        if False:
            print('Hello World!')
        'Add color to the input markdown using click.style.\n\n        :type text: str\n        :param text: The markdown text.\n\n        :rtype: str\n        :return: The input `text`, formatted.\n        '
        pattern_url_name = '[^]]*'
        pattern_url_link = '[^)]+'
        pattern_url = '([!]*\\[{0}]\\(\\s*{1}\\s*\\))'.format(pattern_url_name, pattern_url_link)
        regex_url = re.compile(pattern_url)
        text = regex_url.sub(click.style('\\1', fg='green'), text)
        pattern_url_ref_name = '[^]]*'
        pattern_url_ref_link = '[^]]+'
        pattern_url_ref = '([!]*\\[{0}]\\[\\s*{1}\\s*\\])'.format(pattern_url_ref_name, pattern_url_ref_link)
        regex_url_ref = re.compile(pattern_url_ref)
        text = regex_url_ref.sub(click.style('\\1', fg='green'), text)
        regex_list = re.compile('(  \\*.*)')
        text = regex_list.sub(click.style('\\1', fg='cyan'), text)
        regex_header = re.compile('(#+) (.*)')
        text = regex_header.sub(click.style('\\2', fg='yellow'), text)
        regex_bold = re.compile('(\\*\\*|__)(.*?)\\1')
        text = regex_bold.sub(click.style('\\2', fg='cyan'), text)
        regex_code = re.compile('(`)(.*?)\\1')
        text = regex_code.sub(click.style('\\1\\2\\1', fg='cyan'), text)
        text = re.sub('(\\s*\\r?\\n\\s*){2,}', '\\n\\n', text)
        return text

    def generate_url_contents(self, url):
        if False:
            return 10
        "Generate the formatted contents of the given item's url.\n\n        Converts the HTML to text using HTML2Text, colors it, then displays\n            the output in a pager.\n\n        :type url: str\n        :param url: The url whose contents to fetch.\n\n        :rtype: str\n        :return: The string representation of the formatted url contents.\n        "
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
            raw_response = requests.get(url, headers=headers)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            contents = 'Error: ' + str(e) + '\n'
            contents += 'Try running hn view # with the --browser/-b flag\n'
            return contents
        text = raw_response.text
        contents = self.html_to_text.handle(text)
        contents = re.sub('[^\\x00-\\x7F]+', '', contents)
        contents = self.format_markdown(contents)
        return contents