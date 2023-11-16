from PyQt6 import QtCore
from picard.const import PICARD_URLS
from picard.formats import supported_extensions
from picard.util import versions
from picard.ui import PicardDialog, SingletonDialog
from picard.ui.ui_aboutdialog import Ui_AboutDialog

class AboutDialog(PicardDialog, SingletonDialog):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.ui = Ui_AboutDialog()
        self.ui.setupUi(self)
        self._update_content()

    def _update_content(self):
        if False:
            while True:
                i = 10
        args = versions.as_dict(i18n=True)
        args['third_parties_versions'] = ', '.join([('%s %s' % (versions.version_name(name), value)).replace(' ', '&nbsp;').replace('-', '&#8209;') for (name, value) in versions.as_dict(i18n=True).items() if name != 'version'])
        args['formats'] = ', '.join(map(lambda x: x[1:], supported_extensions()))
        args['copyright_years'] = '2004-2023'
        args['authors_credits'] = ', '.join(['Robert Kaye', 'Lukáš Lalinský', 'Laurent Monin', 'Sambhav Kothari', 'Philipp Wolfer'])
        args['translator_credits'] = _('translator-credits')
        if args['translator_credits'] != 'translator-credits':
            args['translator_credits'] = _('<br/>Translated to LANG by %s') % args['translator_credits'].replace('\n', '<br/>')
        else:
            args['translator_credits'] = ''
        args['icons_credits'] = _('Icons made by Sambhav Kothari <sambhavs.email@gmail.com> and <a href="http://www.flaticon.com/authors/madebyoliver">Madebyoliver</a>, <a href="http://www.flaticon.com/authors/pixel-buddha">Pixel Buddha</a>, <a href="http://www.flaticon.com/authors/nikita-golubev">Nikita Golubev</a>, <a href="http://www.flaticon.com/authors/maxim-basinski">Maxim Basinski</a>, <a href="https://www.flaticon.com/authors/smashicons">Smashicons</a> from <a href="https://www.flaticon.com">www.flaticon.com</a>')

        def strong(s):
            if False:
                i = 10
                return i + 15
            return '<strong>' + s + '</strong>'

        def small(s):
            if False:
                while True:
                    i = 10
            return '<small>' + s + '</small>'

        def url(url, s=None):
            if False:
                i = 10
                return i + 15
            if s is None:
                s = url
            return '<a href="%s">%s</a>' % (url, s)
        text_paragraphs = [strong(_('Version %(version)s')), small('%(third_parties_versions)s'), strong(_('Supported formats')), '%(formats)s', strong(_('Please donate')), _('Thank you for using Picard. Picard relies on the MusicBrainz database, which is operated by the MetaBrainz Foundation with the help of thousands of volunteers. If you like this application please consider donating to the MetaBrainz Foundation to keep the service running.'), url(PICARD_URLS['donate'], _('Donate now!')), strong(_('Credits')), small(_('Copyright © %(copyright_years)s %(authors_credits)s and others') + '%(translator_credits)s'), small('%(icons_credits)s'), strong(_('Official website')), url(PICARD_URLS['home'])]
        self.ui.label.setOpenExternalLinks(True)
        self.ui.label.setText(''.join(('<p align="center">' + p + '</p>' for p in text_paragraphs)) % args)