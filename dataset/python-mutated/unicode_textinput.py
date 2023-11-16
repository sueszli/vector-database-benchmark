from kivy.app import App
from kivy.lang import Builder
from kivy.properties import StringProperty, ObjectProperty
from kivy.core.text import Label as CoreLabel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.spinner import SpinnerOption
from kivy.uix.popup import Popup
import os
Builder.load_string('\n#: import utils kivy\n#: import os os\n#: import Factory kivy.factory.Factory\n<FntSpinnerOption>\n    font_name: self.text if self.text else self.font_name\n\n<Unicode_TextInput>\n    orientation: \'vertical\'\n    txt_input: unicode_txt\n    spnr_fnt: fnt_spnr\n    BoxLayout:\n        size_hint: 1, .05\n        Spinner:\n            id: fnt_spnr\n            text: \'RobotoMono-Regular\'\n            font_name: self.text if self.text else self.font_name\n            values: app.get_font_list\n            option_cls: Factory.FntSpinnerOption\n        Spinner:\n            id: fntsz_spnr\n            text: \'15\'\n            values: map(str, map(sp, range(5,39)))\n    ScrollView:\n        size_hint: 1, .9\n        TextInput:\n            id: unicode_txt\n            background_color: .8811, .8811, .8811, 1\n            foreground_color: 0, 0, 0, 1\n            font_name: fnt_spnr.font_name\n            font_size: sp(fntsz_spnr.text or 0)\n            text: root.unicode_string\n            size_hint: 1, None\n            height: self.minimum_height\n    BoxLayout:\n        size_hint: 1, .05\n        Label:\n            text: \'current font: \' + unicode_txt.font_name\n        Button:\n            size_hint: .15, 1\n            text: \'change Font ...\'\n            valign: \'middle\'\n            halign: \'center\'\n            text_size: self.size\n            on_release: root.show_load()\n\n<LoadDialog>:\n    platform: utils.platform\n    BoxLayout:\n        size: root.size\n        pos: root.pos\n        BoxLayout:\n            orientation: "vertical"\n            size_hint: .2, 1\n            Button:\n                size_hint: 1, .2\n                text: \'User font directory\\n\'\n                valign: \'middle\'\n                halign: \'center\'\n                text_size: self.size\n                on_release:\n                    _platform = root.platform\n                    filechooser.path = (os.path.expanduser(\'~/.fonts\')\n                    if _platform == \'linux\' else \'/system/fonts\'\n                    if _platform == \'android\'\n                    else os.path.expanduser(\'~/Library/Fonts\')\n                    if _platform == \'macosx\'\n                    else os.environ[\'WINDIR\'] +\'\\Fonts\')\n            Button:\n                size_hint: 1, .2\n                text: \'System Font directory\'\n                valign: \'middle\'\n                halign: \'center\'\n                text_size: self.size\n                on_release:\n                    _platform = root.platform\n                    filechooser.path = (\'/usr/share/fonts\'\n                    if _platform == \'linux\' else \'/system/fonts\'\n                    if _platform == \'android\' else os.path.expanduser\n                    (\'/System/Library/Fonts\') if _platform == \'macosx\'\n                    else os.environ[\'WINDIR\'] + "\\Fonts")\n            Label:\n                text: \'BookMarks\'\n        BoxLayout:\n            orientation: "vertical"\n            FileChooserListView:\n                id: filechooser\n                filters: [\'*.ttf\']\n            BoxLayout:\n                size_hint_y: None\n                height: 30\n                Button:\n                    text: "cancel"\n                    on_release: root.cancel()\n                Button:\n                    text: "load"\n                    on_release: filechooser.selection != [] and root.load(filechooser.path, filechooser.selection)\n')

class FntSpinnerOption(SpinnerOption):
    pass

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class Unicode_TextInput(BoxLayout):
    txt_input = ObjectProperty(None)
    unicode_string = StringProperty("Latin-1 supplement: éé çç ßß\n\nList of major languages taken from Google Translate\n____________________________________________________\nTry changing the font to see if the font can render the glyphs you need in your\napplication. Scroll to see all languages in the list.\n\nBasic Latin:    The quick brown fox jumps over the lazy old dog.\nAlbanian:       Kafe të shpejtë dhelpra hedhje mbi qen lazy vjetër.\nالثعلب البني السريع يقفز فوق الكلب القديمة البطيئة.         :Arabic\nAfricans:       Die vinnige bruin jakkals spring oor die lui hond.\nArmenian:       Արագ Brown Fox jumps ավելի ծույլ հին շունը.\nAzerbaijani:    Tez qonur tülkü də tənbəl yaşlı it üzərində atlamalar.\nBasque:         Azkar marroia fox alferrak txakur zaharra baino gehiago jauzi.\nBelarusian:     Хуткі карычневы ліс пераскоквае праз гультаяваты стары сабака.\nBengali:        দ্রুত বাদামী শিয়াল অলস পুরানো কুকুর বেশি\nBulgarian:      Бързата кафява лисица скача над мързелив куче.\nChinese Simpl:  敏捷的棕色狐狸跳过懒惰的老狗。\nCatalan:        La cigonya tocava el saxofon en el vell gos mandrós.\nCroation:       Brzo smeđa lisica skoči preko lijen stari pas.\nCzech:          Rychlá hnědá liška skáče přes líného starého psa.\nDanish:         Den hurtige brune ræv hopper over den dovne gamle hund.\nDutch:          De snelle bruine vos springt over de luie oude hond.\nEstonian:       Kiire pruun rebane hüppab üle laisa vana koer.\nFilipino:       Ang mabilis na brown soro jumps sa ang tamad lumang aso.\nFinnish:        Nopea ruskea kettu hyppää yli laiska vanha koira.\nFrench:         Le renard brun rapide saute par dessus le chien\n                paresseux vieux.\nGalician:       A lixeira raposo marrón ataca o can preguiceiro de idade.\nGregorian:      სწრაფი ყავისფერი მელა jumps გამო ზარმაცი წლის ძაღლი.\nGerman:         Der schnelle braune Fuchs springt über den faulen alten Hund.\nGreek:          Η γρήγορη καφέ αλεπού πηδάει πάνω από το τεμπέλικο\n                γέρικο σκυλί.\nGujrati:        આ ઝડપી ભુરો શિયાળ તે બેકાર જૂના કૂતરા પર કૂદકા.\nGurmukhi:       ਤੇਜ ਭੂਰੇ ਰੰਗ ਦੀ ਲੂੰਬੜੀ ਆਲਸੀ ਬੁੱਢੇ ਕੁੱਤੇ ਦੇ ਉਤੋਂ ਦੀ ਟੱਪਦੀ ਹੈ ।\nHiation Creole: Rapid mawon Rena a so sou chen an parese fin vye granmoun.\nHebrew:         השועל החום הזריז קופץ על הכלב הישן עצלן.\nHindi:          तेज भूरे रंग की लोमड़ी आलसी बूढ़े कुत्ते के उपर से कूदती है ॥\nHungarian:      A gyors barna róka átugorja a lusta vén kutya.\nIcelandic:      The fljótur Brown refur stökk yfir latur gamall hundur.\nIndonesian:     Cepat rubah cokelat melompat atas anjing tua malas.\nIrish:          An sionnach donn tapaidh jumps thar an madra leisciúil d'aois.\nItalian:        The quick brown fox salta sul cane pigro vecchio.\nJapanese:       速い茶色のキツネは、のろまな古いイヌに飛びかかった。\nKannada:        ತ್ವರಿತ ಕಂದು ನರಿ ಆಲೂಗಡ್ಡೆ ಹಳೆಯ ಶ್ವಾನ ಮೇಲೆ ಜಿಗಿತಗಳು.\nKorean:         무궁화 게으른 옛 피었습니다.\nLatin:          Vivamus adipiscing orci et rutrum tincidunt super vetus canis.\nLatvian:        Ātra brūna lapsa lec pāri slinkam vecs suns.\nLithuanian:     Greita ruda lapė šokinėja per tingus senas šuo.\nMacedonian:     Брзата кафена лисица скокови над мрзливи стариот пес.\nMalay:          Fox coklat cepat melompat atas anjing lama malas.\nMaltese:        Il-volpi kannella malajr jumps fuq il-kelb qodma għażżien.\nNorweigian:     Den raske brune reven hopper over den late gamle hunden.\nPersian:        روباه قهوه ای سریع روی سگ تنبل قدیمی میپرد.\nPolish:         Szybki brązowy lis przeskoczył nad leniwym psem życia.\nPortugese:      A ligeira raposa marrom ataca o cão preguiçoso de idade.\nRomanian:       Rapidă maro vulpea sare peste cainele lenes vechi.\nRussian:        Быстрая коричневая лисица перепрыгивает ленивого старого пса.\nSerniam:        Брза смеђа лисица прескаче лењог пса старог.\nSlovak:         Rýchla hnedá líška skáče cez lenivého starého psa.\nSlovenian:      Kožuščku hudobnega nad leni starega psa.\nSpanish:        La cigüeña tocaba el saxofón en el viejo perro perezoso.\nSwahili:        Haraka brown fox anaruka juu ya mbwa wavivu zamani.\nSwedish:        Den snabba bruna räven hoppar över den lata gammal hund.\nTamil:          விரைவான பிரவுன் ஃபாக்ஸ் சோம்பேறி பழைய நாய் மீது\n                தொடரப்படுகிறது\nTelugu:         శీఘ్ర బ్రౌన్ ఫాక్స్ సోమరితనం పాత కుక్క కంటే హెచ్చుతగ్గుల.\nThai:           สีน้ำตาลอย่างรวดเร็วจิ้งจอกกระโดดมากกว่าสุนัขเก่าที่ขี้เกียจ\nTurkish:        Hızlı kahverengi tilki tembel köpeğin üstünden atlar.\nUkrainian:       Швидкий коричневий лис перестрибує через лінивий старий пес.\nUrdu:           فوری بھوری لومڑی سست بوڑھے کتے پر کودتا.\nVietnamese:     Các con cáo nâu nhanh chóng nhảy qua con chó lười biếng cũ.\nWelsh:          Mae'r cyflym frown llwynog neidio dros y ci hen ddiog.\nYiddish:        דער גיך ברוין פוקס דזשאַמפּס איבער די פויל אַלט הונט.")

    def dismiss_popup(self):
        if False:
            return 10
        self._popup.dismiss()

    def load(self, _path, _fname):
        if False:
            print('Hello World!')
        self.txt_input.font_name = _fname[0]
        _f_name = _fname[0][_fname[0].rfind(os.sep) + 1:]
        self.spnr_fnt.text = _f_name[:_f_name.rfind('.')]
        self._popup.dismiss()

    def show_load(self):
        if False:
            for i in range(10):
                print('nop')
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title='load file', content=content, size_hint=(0.9, 0.9))
        self._popup.open()
from kivy.utils import reify

class unicode_app(App):

    def build(self):
        if False:
            print('Hello World!')
        return Unicode_TextInput()

    @reify
    def get_font_list(self):
        if False:
            print('Hello World!')
        'Get a list of all the fonts available on this system.\n        '
        fonts_path = CoreLabel.get_system_fonts_dir()
        flist = []
        for fdir in fonts_path:
            for fpath in sorted(os.listdir(fdir)):
                if fpath.endswith('.ttf'):
                    flist.append(fpath[:-4])
        return sorted(flist)
if __name__ == '__main__':
    unicode_app().run()