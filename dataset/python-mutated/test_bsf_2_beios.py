"""
Tests the conversion code for the lang_uk NER dataset
"""
import unittest
from stanza.utils.datasets.ner.convert_bsf_to_beios import convert_bsf, parse_bsf, BsfInfo
import pytest
pytestmark = [pytest.mark.travis, pytest.mark.pipeline]

class TestBsf2Beios(unittest.TestCase):

    def test_empty_markup(self):
        if False:
            while True:
                i = 10
        res = convert_bsf('', '')
        self.assertEqual('', res)

    def test_1line_markup(self):
        if False:
            for i in range(10):
                print('nop')
        data = 'тележурналіст Василь'
        bsf_markup = 'T1\tPERS 14 20\tВасиль'
        expected = 'тележурналіст O\nВасиль S-PERS'
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_1line_follow_markup(self):
        if False:
            while True:
                i = 10
        data = 'тележурналіст Василь .'
        bsf_markup = 'T1\tPERS 14 20\tВасиль'
        expected = 'тележурналіст O\nВасиль S-PERS\n. O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_1line_2tok_markup(self):
        if False:
            i = 10
            return i + 15
        data = 'тележурналіст Василь Нагірний .'
        bsf_markup = 'T1\tPERS 14 29\tВасиль Нагірний'
        expected = 'тележурналіст O\nВасиль B-PERS\nНагірний E-PERS\n. O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_1line_Long_tok_markup(self):
        if False:
            print('Hello World!')
        data = 'А в музеї Гуцульщини і Покуття можна '
        bsf_markup = 'T12\tORG 4 30\tмузеї Гуцульщини і Покуття'
        expected = 'А O\nв O\nмузеї B-ORG\nГуцульщини I-ORG\nі I-ORG\nПокуття E-ORG\nможна O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_2line_2tok_markup(self):
        if False:
            while True:
                i = 10
        data = 'тележурналіст Василь Нагірний .\nВ івано-франківському видавництві «Лілея НВ» вийшла друком'
        bsf_markup = 'T1\tPERS 14 29\tВасиль Нагірний\nT2\tORG 67 75\tЛілея НВ'
        expected = 'тележурналіст O\nВасиль B-PERS\nНагірний E-PERS\n. O\n\n\nВ O\nівано-франківському O\nвидавництві O\n« O\nЛілея B-ORG\nНВ E-ORG\n» O\nвийшла O\nдруком O'
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

    def test_real_markup(self):
        if False:
            for i in range(10):
                print('nop')
        data = "Через напіввоєнний стан в Україні та збільшення телефонних терористичних погроз українці купуватимуть sim-карти тільки за паспортами .\nПро це повідомив начальник управління зв'язків зі ЗМІ адміністрації Держспецзв'язку Віталій Кукса .\nВін зауважив , що днями відомство опублікує проект змін до правил надання телекомунікаційних послуг , де будуть прописані норми ідентифікації громадян .\nАбонентів , які на сьогодні вже мають sim-карту , за словами Віталія Кукси , реєструватимуть , коли ті звертатимуться в службу підтримки свого оператора мобільного зв'язку .\nОднак мобільні оператори побоюються , що таке нововведення помітно зменшить продаж стартових пакетів , адже спеціалізовані магазини є лише у містах .\nВідтак купити сімку в невеликих населених пунктах буде неможливо .\nКрім того , нова процедура ідентифікації абонентів вимагатиме від операторів мобільного зв'язку додаткових витрат .\n- Близько 90 % українських абонентів - це абоненти передоплати .\nЯкщо мова буде йти навіть про поетапну їх ідентифікацію , зробити це буде складно , довго і дорого .\nМобільним операторам доведеться йти на чималі витрати , пов'язані з укладанням і зберіганням договорів , веденням баз даних , - розповіла « Економічній правді » начальник відділу зв'язків з громадськістю « МТС-Україна » Вікторія Рубан .\n"
        bsf_markup = "T1\tLOC 26 33\tУкраїні\nT2\tORG 203 218\tДержспецзв'язку\nT3\tPERS 219 232\tВіталій Кукса\nT4\tPERS 449 462\tВіталія Кукси\nT5\tORG 1201 1219\tЕкономічній правді\nT6\tORG 1267 1278\tМТС-Україна\nT7\tPERS 1281 1295\tВікторія Рубан\n"
        expected = "Через O\nнапіввоєнний O\nстан O\nв O\nУкраїні S-LOC\nта O\nзбільшення O\nтелефонних O\nтерористичних O\nпогроз O\nукраїнці O\nкупуватимуть O\nsim-карти O\nтільки O\nза O\nпаспортами O\n. O\n\n\nПро O\nце O\nповідомив O\nначальник O\nуправління O\nзв'язків O\nзі O\nЗМІ O\nадміністрації O\nДержспецзв'язку S-ORG\nВіталій B-PERS\nКукса E-PERS\n. O\n\n\nВін O\nзауважив O\n, O\nщо O\nднями O\nвідомство O\nопублікує O\nпроект O\nзмін O\nдо O\nправил O\nнадання O\nтелекомунікаційних O\nпослуг O\n, O\nде O\nбудуть O\nпрописані O\nнорми O\nідентифікації O\nгромадян O\n. O\n\n\nАбонентів O\n, O\nякі O\nна O\nсьогодні O\nвже O\nмають O\nsim-карту O\n, O\nза O\nсловами O\nВіталія B-PERS\nКукси E-PERS\n, O\nреєструватимуть O\n, O\nколи O\nті O\nзвертатимуться O\nв O\nслужбу O\nпідтримки O\nсвого O\nоператора O\nмобільного O\nзв'язку O\n. O\n\n\nОднак O\nмобільні O\nоператори O\nпобоюються O\n, O\nщо O\nтаке O\nнововведення O\nпомітно O\nзменшить O\nпродаж O\nстартових O\nпакетів O\n, O\nадже O\nспеціалізовані O\nмагазини O\nє O\nлише O\nу O\nмістах O\n. O\n\n\nВідтак O\nкупити O\nсімку O\nв O\nневеликих O\nнаселених O\nпунктах O\nбуде O\nнеможливо O\n. O\n\n\nКрім O\nтого O\n, O\nнова O\nпроцедура O\nідентифікації O\nабонентів O\nвимагатиме O\nвід O\nоператорів O\nмобільного O\nзв'язку O\nдодаткових O\nвитрат O\n. O\n\n\n- O\nБлизько O\n90 O\n% O\nукраїнських O\nабонентів O\n- O\nце O\nабоненти O\nпередоплати O\n. O\n\n\nЯкщо O\nмова O\nбуде O\nйти O\nнавіть O\nпро O\nпоетапну O\nїх O\nідентифікацію O\n, O\nзробити O\nце O\nбуде O\nскладно O\n, O\nдовго O\nі O\nдорого O\n. O\n\n\nМобільним O\nоператорам O\nдоведеться O\nйти O\nна O\nчималі O\nвитрати O\n, O\nпов'язані O\nз O\nукладанням O\nі O\nзберіганням O\nдоговорів O\n, O\nведенням O\nбаз O\nданих O\n, O\n- O\nрозповіла O\n« O\nЕкономічній B-ORG\nправді E-ORG\n» O\nначальник O\nвідділу O\nзв'язків O\nз O\nгромадськістю O\n« O\nМТС-Україна S-ORG\n» O\nВікторія B-PERS\nРубан E-PERS\n. O"
        self.assertEqual(expected, convert_bsf(data, bsf_markup))

class TestBsf(unittest.TestCase):

    def test_empty_bsf(self):
        if False:
            print('Hello World!')
        self.assertEqual(parse_bsf(''), [])

    def test_empty2_bsf(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parse_bsf(' \n \n'), [])

    def test_1line_bsf(self):
        if False:
            print('Hello World!')
        bsf = 'T1\tPERS 103 118\tВасиль Нагірний'
        res = parse_bsf(bsf)
        expected = BsfInfo('T1', 'PERS', 103, 118, 'Василь Нагірний')
        self.assertEqual(len(res), 1)
        self.assertEqual(res, [expected])

    def test_2line_bsf(self):
        if False:
            print('Hello World!')
        bsf = 'T9\tPERS 778 783\tКарла\nT10\tMISC 814 819\tміста'
        res = parse_bsf(bsf)
        expected = [BsfInfo('T9', 'PERS', 778, 783, 'Карла'), BsfInfo('T10', 'MISC', 814, 819, 'міста')]
        self.assertEqual(len(res), 2)
        self.assertEqual(res, expected)

    def test_multiline_bsf(self):
        if False:
            for i in range(10):
                print('nop')
        bsf = 'T3\tPERS 220 235\tАндрієм Кіщуком\nT4\tMISC 251 285\tА .\nKubler .\nСвітло і тіні маестро\nT5\tPERS 363 369\tКіблер'
        res = parse_bsf(bsf)
        expected = [BsfInfo('T3', 'PERS', 220, 235, 'Андрієм Кіщуком'), BsfInfo('T4', 'MISC', 251, 285, 'А .\nKubler .\nСвітло і тіні маестро'), BsfInfo('T5', 'PERS', 363, 369, 'Кіблер')]
        self.assertEqual(len(res), len(expected))
        self.assertEqual(res, expected)
if __name__ == '__main__':
    unittest.main()