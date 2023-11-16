from test.picardtestcase import PicardTestCase
from picard import util
from picard.const.sys import IS_WIN
show_latin2ascii_coverage = False
compatibility_from = 'ĲĳſǇǈǉǊǋǌǱǲǳﬀﬁﬂﬃﬄﬅﬆＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ℀℁ℂ℅℆ℊℋℌℍℎℐℑℒℓℕ№ℙℚℛℜℝ℡ℤℨℬℭℯℰℱℳℴℹ℻ⅅⅆⅇⅈⅉ㍱㍲㍳㍴㍵㍶㍷㍺㎀㎁㎃㎄㎅㎆㎇㎈㎉㎊㎋㎎㎏㎐㎑㎒㎓㎔㎙㎚㎜㎝㎞㎩㎪㎫㎬㎭㎰㎱㎳㎴㎵㎷㎸㎹㎺㎻㎽㎾㎿㏂㏃㏄㏅㏇㏈㏉㏊㏋㏌㏍㏎㏏㏐㏑㏒㏓㏔㏕㏖㏗㏘㏙㏚㏛㏜㏝⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅬⅭⅮⅯⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻⅼⅽⅾⅿ⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛０１２３４５６７８９\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u205f＂＇﹣－․‥…‼⁇⁈⁉︐︓︔︕︖︙︰︵︶︷︸﹇﹈﹐﹒﹔﹕﹖﹗﹙﹚﹛﹜﹟﹠﹡﹢﹤﹥﹦﹨﹩﹪﹫！＃＄％＆（）＊＋，．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～⩴⩵⩶'
compatibility_to = 'IJijsLJLjljNJNjnjDZDzdzfffiflffifflststABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyza/ca/sCc/oc/ugHHHhIILlNNoPQRRRTELZZBCeEFMoiFAXDdeijhPadaAUbaroVpcdmIUpAnAmAkAKBMBGBcalkcalpFnFmgkgHzkHzMHzGHzTHzfmnmmmcmkmPakPaMPaGParadpsnsmspVnVmVkVMVpWnWmWkWMWa.m.BqcccdCo.dBGyhaHPinKKKMktlmlnloglxmbmilmolPHp.m.PPMPRsrSvWb(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)(l)(m)(n)(o)(p)(q)(r)(s)(t)(u)(v)(w)(x)(y)(z)IIIIIIIVVVIVIIVIIIIXXXIXIILCDMiiiiiiivvviviiviiiixxxixiilcdm(1)(2)(3)(4)(5)(6)(7)(8)(9)(10)(11)(12)(13)(14)(15)(16)(17)(18)(19)(20)1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.0123456789          "\'--......!!???!!?,:;!?.....(){}[],.;:?!(){}#&*+<>=\\$%@!#$%&()*+,./:;<=>?@[\\]^_`{|}~::======'
compatibility_from += 'ᴀᴄᴅᴇᴊᴋᴍᴏᴘᴛᴜᴠᴡᴢ〇\xa0\u3000'
compatibility_to += 'ACDEJKMOPTUVWZ0  '
punctuation_from = '‘’‚‛“”„‟′〝〞«»‹›\xad‐‒–—―‖⁄⁅⁆⁎〈〉《》〔〕〘〙〚〛−∕∖∣∥≪≫⦅⦆•\u200b'
punctuation_to = '\'\'\'\'""""\'""<<>><>-----||/[]*<><<>>[][][]-/\\|||<<>>(())-'
combinations_from = 'ÆÐØÞßæðøþĐđĦħıĸŁłŊŋŒœŦŧƀƁƂƃƇƈƉƊƋƌƐƑƒƓƕƖƗƘƙƚƝƞƢƣƤƥƫƬƭƮƲƳƴƵƶǤǥȡȤȥȴȵȶȷȸȹȺȻȼȽȾȿɀɃɄɆɇɈɉɌɍɎɏɓɕɖɗɛɟɠɡɢɦɧɨɪɫɬɭɱɲɳɴɼɽɾʀʂʈʉʋʏʐʑʙʛʜʝʟʠʣʥʦʪʫᴃᴆᴌᵫᵬᵭᵮᵯᵰᵱᵲᵳᵴᵵᵶᵺᵻᵽᵾᶀᶁᶂᶃᶄᶅᶆᶇᶈᶉᶊᶌᶍᶎᶏᶑᶒᶓᶖᶙẜẝẞỺỻỼỽỾỿ©®₠₢₣₤₧₺₹℞、。×÷·ẟƄƅƾ'
combinations_to = 'AEDOETHssaedoethDdHhiqLlNnOEoeTtbBBbCcDDDdEFfGhvIIKklNnGHghPptTtTVYyZzGgdZzlntjdbqpACcLTszBUEeJjRrYybcddejggGhhiIlllmnnNrrrRstuvYzzBGHjLqdzdztslslzBDLuebdfmnprrstzthIpUbdfgklmnprsvxzadeeiussSSLLllVvYy(C)(R)CECrFr.L.PtsTLRsRx,.x/.ddHhts'
ascii_chars = ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'

class CompatibilityTest(PicardTestCase):

    def test_correct(self):
        if False:
            print('Hello World!')
        self.maxDiff = None
        self.assertEqual(util.textencoding.unicode_simplify_compatibility(compatibility_from), compatibility_to)
        self.assertEqual(util.textencoding.unicode_simplify_compatibility(punctuation_from), punctuation_from)
        self.assertEqual(util.textencoding.unicode_simplify_compatibility(combinations_from), combinations_from)
        self.assertEqual(util.textencoding.unicode_simplify_compatibility(ascii_chars), ascii_chars)

    def test_pathsave(self):
        if False:
            while True:
                i = 10
        self.assertEqual(util.textencoding.unicode_simplify_compatibility('／', pathsave=True), '_')

    def test_incorrect(self):
        if False:
            i = 10
            return i + 15
        pass

class PunctuationTest(PicardTestCase):

    def test_correct(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = None
        self.assertEqual(util.textencoding.unicode_simplify_punctuation(compatibility_from), compatibility_from)
        self.assertEqual(util.textencoding.unicode_simplify_punctuation(punctuation_from), punctuation_to)
        self.assertEqual(util.textencoding.unicode_simplify_punctuation(combinations_from), combinations_from)
        self.assertEqual(util.textencoding.unicode_simplify_punctuation(ascii_chars), ascii_chars)

    def test_pathsave(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(util.textencoding.unicode_simplify_punctuation('∕∖', True), '__' if IS_WIN else '_\\')
        self.assertEqual(util.textencoding.unicode_simplify_punctuation('/\\∕∖', True), '/\\__' if IS_WIN else '/\\_\\')

    def test_pathsave_win_compat(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(util.textencoding.unicode_simplify_punctuation('∕∖', True, True), '__')
        self.assertEqual(util.textencoding.unicode_simplify_punctuation('/\\∕∖', True, True), '/\\__')

    def test_incorrect(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class CombinationsTest(PicardTestCase):

    def test_correct(self):
        if False:
            print('Hello World!')
        self.maxDiff = None
        self.assertEqual(util.textencoding.unicode_simplify_combinations(combinations_from), combinations_to)
        self.assertEqual(util.textencoding.unicode_simplify_combinations(compatibility_from), compatibility_from)
        self.assertEqual(util.textencoding.unicode_simplify_combinations(punctuation_from), punctuation_from)
        self.assertEqual(util.textencoding.unicode_simplify_combinations(ascii_chars), ascii_chars)

    def test_pathsave(self):
        if False:
            while True:
                i = 10
        self.assertEqual(util.textencoding.unicode_simplify_combinations('8½', True), '8 1_2')
        self.assertEqual(util.textencoding.unicode_simplify_combinations('8/\\½', True), '8/\\ 1_2')

    def test_incorrect(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class AsciiPunctTest(PicardTestCase):

    def test_correct(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(util.textencoding.asciipunct('‘Test’'), "'Test'")
        self.assertEqual(util.textencoding.asciipunct('“Test”'), '"Test"')
        self.assertEqual(util.textencoding.asciipunct('1′6″'), '1\'6"')
        self.assertEqual(util.textencoding.asciipunct('…'), '...')
        self.assertEqual(util.textencoding.asciipunct('․'), '.')
        self.assertEqual(util.textencoding.asciipunct('‥'), '..')

    def test_incorrect(self):
        if False:
            print('Hello World!')
        pass

class UnaccentTest(PicardTestCase):

    def test_correct(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(util.textencoding.unaccent('Lukáš'), 'Lukas')
        self.assertEqual(util.textencoding.unaccent('Björk'), 'Bjork')
        self.assertEqual(util.textencoding.unaccent('小室哲哉'), '小室哲哉')

    def test_incorrect(self):
        if False:
            print('Hello World!')
        self.assertNotEqual(util.textencoding.unaccent('Björk'), 'Björk')
        self.assertNotEqual(util.textencoding.unaccent('小室哲哉'), 'Tetsuya Komuro')
        self.assertNotEqual(util.textencoding.unaccent('Trentemøller'), 'Trentemoller')
        self.assertNotEqual(util.textencoding.unaccent('Ænima'), 'AEnima')
        self.assertNotEqual(util.textencoding.unaccent('ænima'), 'aenima')

class ReplaceNonAsciiTest(PicardTestCase):

    def test_correct(self):
        if False:
            return 10
        self.assertEqual(util.textencoding.replace_non_ascii('Lukáš'), 'Lukas')
        self.assertEqual(util.textencoding.replace_non_ascii('Björk'), 'Bjork')
        self.assertEqual(util.textencoding.replace_non_ascii('Trentemøller'), 'Trentemoeller')
        self.assertEqual(util.textencoding.replace_non_ascii('Ænima'), 'AEnima')
        self.assertEqual(util.textencoding.replace_non_ascii('ænima'), 'aenima')
        self.assertEqual(util.textencoding.replace_non_ascii('小室哲哉'), '____')
        self.assertEqual(util.textencoding.replace_non_ascii('ᴀᴄᴇ'), 'ACE')
        self.assertEqual(util.textencoding.replace_non_ascii('Ａｂｃ'), 'Abc')
        self.assertEqual(util.textencoding.replace_non_ascii('500㎏,2㎓'), '500kg,2GHz')
        self.assertEqual(util.textencoding.replace_non_ascii('⒜⒝⒞'), '(a)(b)(c)')
        self.assertEqual(util.textencoding.replace_non_ascii('ⅯⅯⅩⅣ'), 'MMXIV')
        self.assertEqual(util.textencoding.replace_non_ascii('ⅿⅿⅹⅳ'), 'mmxiv')
        self.assertEqual(util.textencoding.replace_non_ascii('⑴⑵⑶'), '(1)(2)(3)')
        self.assertEqual(util.textencoding.replace_non_ascii('⒈ ⒉ ⒊'), '1. 2. 3.')
        self.assertEqual(util.textencoding.replace_non_ascii('１２３'), '123')
        self.assertEqual(util.textencoding.replace_non_ascii('∖⁄∕／'), '\\///')

    def test_pathsave(self):
        if False:
            i = 10
            return i + 15
        expected = '____/8 1_2\\' if IS_WIN else '\\___/8 1_2\\'
        self.assertEqual(util.textencoding.replace_non_ascii('∖⁄∕／/8½\\', pathsave=True), expected)

    def test_win_compat(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(util.textencoding.replace_non_ascii('∖⁄∕／/8½\\', pathsave=True, win_compat=True), '____/8 1_2\\')

    def test_incorrect(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(util.textencoding.replace_non_ascii('Lukáš'), 'Lukáš')
        self.assertNotEqual(util.textencoding.replace_non_ascii('Lukáš'), 'Luk____')
if show_latin2ascii_coverage:
    latin_1 = 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ'
    latin_a = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽž'
    latin_b = 'ƀƁƂƃƄƅƆƇƈƉƊƋƌƍƎƏƐƑƒƓƔƕƖƗƘƙƚƛƜƝƞƟƠơƢƣƤƥƦƧƨƩƪƫƬƭƮƯưƱƲƳƴƵƶƷƸƹƺƻƼƽƾƿǀǁǂǃǄǅǆǇǈǉǊǋǌǍǎǏǐǑǒǓǔǕǖǗǘǙǚǛǜǝǞǟǠǡǢǣǤǥǦǧǨǩǪǫǬǭǮǯǰǱǲǳǴǵǶǷǸǹǺǻǼǽǾǿȀȁȂȃȄȅȆȇȈȉȊȋȌȍȎȏȐȑȒȓȔȕȖȗȘșȚțȜȝȞȟȠȡȢȣȤȥȦȧȨȩȪȫȬȭȮȯȰȱȲȳȴȵȶȷȸȹȺȻȼȽȾȿɀɁɂɃɄɅɆɇɈɉɊɋɌɍɎɏ'
    ipa_ext = 'ɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɩɪɫɬɭɮɯɰɱɲɳɴɵɶɷɸɹɺɻɼɽɾɿʀʁʂʃʄʅʆʇʈʉʊʋʌʍʎʏʐʑʒʓʔʕʖʗʘʙʚʛʜʝʞʟʠʡʢʣʤʥʦʧʨʩʪʫʬʭʮʯ'
    phonetic = 'ᴀᴁᴂᴃᴄᴅᴆᴇᴈᴉᴊᴋᴌᴍᴎᴏᴐᴑᴒᴓᴔᴕᴖᴗᴘᴙᴚᴛᴜᴝᴞᴟᴠᴡᴢᴣᴤᴥᴦᴧᴨᴩᴪᴫᴬᴭᴮᴯᴰᴱᴲᴳᴴᴵᴶᴷᴸᴹᴺᴻᴼᴽᴾᴿᵀᵁᵂᵃᵄᵅᵆᵇᵈᵉᵊᵋᵌᵍᵎᵏᵐᵑᵒᵓᵔᵕᵖᵗᵘᵙᵚᵛᵜᵝᵞᵟᵠᵡᵢᵣᵤᵥᵦᵧᵨᵩᵪᵫᵬᵭᵮᵯᵰᵱᵲᵳᵴᵵᵶᵷᵸᵹᵺᵻᵼᵽᵾᵿᶀᶁᶂᶃᶄᶅᶆᶇᶈᶉᶊᶋᶌᶍᶎᶏᶐᶑᶒᶓᶔᶕᶖᶗᶘᶙᶚᶛᶜᶝᶞᶟᶠᶡᶢᶣᶤᶥᶦᶧᶨᶩᶪᶫᶬᶭᶮᶯᶰᶱᶲᶳᶴᶵᶶᶷᶸᶹᶺᶻᶼᶽᶾᶿ'
    latin_ext_add = 'ḀḁḂḃḄḅḆḇḈḉḊḋḌḍḎḏḐḑḒḓḔḕḖḗḘḙḚḛḜḝḞḟḠḡḢḣḤḤḦḧḨḩḪḫḬḭḮḯḰḱḲḳḴḵḶḷḸḹḺḻḼḽḾḿṀṁṂṃṄṅṆṇṈṉṊṋṌṍṎṏṐṑṒṓṔṕṖṗṘṙṚṛṜṝṞṟṠṡṢṣṤṥṦṧṨṩṪṫṬṭṮṯṰṱṲṳṴṵṶṷṸṹṺṻṼṽṾṿẀẁẂẃẄẅẆẇẈẉẊẋẌẍẎẏẐẑẒẓẔẕẖẗẘẙẚẛẜẝẞẟẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹỺỻỼỽỾỿ'
    letter_like = '℀℁ℂ℃℄℅℆ℇ℈℉ℊℋℌℍℎℏℐℑℒℓ℔ℕ№℗℘ℙℚℛℜℝ℞℟℠℡™℣ℤ℥Ω℧ℨ℩KÅℬℭ℮ℯℰℱℲℳℴℵℶℷℸℹ℺℻ℼℽℾℿ⅀⅁⅂⅃⅄ⅅⅆⅇⅈⅉ⅊⅋⅌⅍ⅎ⅏'
    enclosed = '⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ⓪⓫⓬⓭⓮⓯'
    print('The following lines show the coverage of Latin characters conversion to ascii.')
    print('Underscores are characters which currently do not have an ASCII representation.')
    print()
    print('latin-1:       ', util.textencoding.replace_non_ascii(latin_1))
    print('latin-1:       ', util.textencoding.replace_non_ascii(latin_1))
    print('latin-a:       ', util.textencoding.replace_non_ascii(latin_a))
    print('latin-b:       ', util.textencoding.replace_non_ascii(latin_b))
    print('ipa-ext:       ', util.textencoding.replace_non_ascii(ipa_ext))
    print('phonetic:      ', util.textencoding.replace_non_ascii(phonetic))
    print('latin-ext-add: ', util.textencoding.replace_non_ascii(latin_ext_add))
    print('letter-like:   ', util.textencoding.replace_non_ascii(letter_like))
    print('enclosed:      ', util.textencoding.replace_non_ascii(enclosed))
    print()