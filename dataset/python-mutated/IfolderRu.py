import re
from ..base.simple_downloader import SimpleDownloader

class IfolderRu(SimpleDownloader):
    __name__ = 'IfolderRu'
    __type__ = 'downloader'
    __version__ = '0.44'
    __status__ = 'testing'
    __pattern__ = 'http://(?:www)?(files\\.)?(ifolder\\.ru|metalarea\\.org|rusfolder\\.(com|net|ru))/(files/)?(?P<ID>\\d+)'
    __config__ = [('enabled', 'bool', 'Activated', True), ('use_premium', 'bool', 'Use premium account if available', True), ('fallback', 'bool', 'Fallback to free download if premium fails', True), ('chk_filesize', 'bool', 'Check file size', True), ('max_wait', 'int', 'Reconnect if waiting time is greater than minutes', 10)]
    __description__ = 'Ifolder.ru downloader plugin'
    __license__ = 'GPLv3'
    __authors__ = [('zoidberg', 'zoidberg@mujmail.cz')]
    SIZE_REPLACEMENTS = [('Кб', 'KiB'), ('Мб', 'MiB'), ('Гб', 'GiB')]
    NAME_PATTERN = '(?:<div><span>)?Название:(?:</span>)? <b>(?P<N>.+?)</b><(?:/div|br)>'
    SIZE_PATTERN = '(?:<div><span>)?Размер:(?:</span>)? <b>(?P<S>.+?)</b><(?:/div|br)>'
    OFFLINE_PATTERN = '<p>Файл номер <b>.*?</b> (не найден|удален) !!!</p>'
    SESSION_ID_PATTERN = '<input type="hidden" name="session" value="(.+?)"'
    INTS_SESSION_PATTERN = '\\(\\\'ints_session\\\'\\);\\s*if\\(tag\\){{tag\\.value = "(.+?)";}}'
    HIDDEN_INPUT_PATTERN = "var v = .*?name=\\'(.+?)\\' value=\\'1\\'"
    LINK_FREE_PATTERN = '<a href="(.+?)" class="downloadbutton_files"'
    WRONG_CAPTCHA_PATTERN = '<font color=Red>неверный код,<br>введите еще раз</font><br>'

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.resume_download = self.multi_dl = bool(self.account)
        self.chunk_limit = 1

    def handle_free(self, pyfile):
        if False:
            print('Hello World!')
        id = self.info['pattern']['ID']
        url = f'http://rusfolder.com/{id}'
        self.data = self.load(url)
        self.get_file_info()
        session_id = re.search(self.SESSION_ID_PATTERN, self.data).groups()
        captcha_url = 'http://ints.rusfolder.com/random/images/?session={}'.format(session_id)
        (action, inputs) = self.parse_html_form('id="download-step-one-form"')
        inputs['confirmed_number'] = self.captcha.decrypt(captcha_url, cookies=True)
        inputs['action'] = '1'
        self.log_debug(inputs)
        self.data = self.load(url, post=inputs)
        if self.WRONG_CAPTCHA_PATTERN in self.data:
            self.retry_captcha()
        self.link = re.search(self.LINK_FREE_PATTERN, self.data).group(1)