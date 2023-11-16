from __future__ import absolute_import, unicode_literals
raw_prefixes = 'rR'
bytes_prefixes = 'bB'
string_prefixes = 'fFuU' + bytes_prefixes
char_prefixes = 'cC'
any_string_prefix = raw_prefixes + string_prefixes + char_prefixes
IDENT = 'IDENT'

def make_lexicon():
    if False:
        i = 10
        return i + 15
    from ..Plex import Str, Any, AnyBut, AnyChar, Rep, Rep1, Opt, Bol, Eol, Eof, TEXT, IGNORE, Method, State, Lexicon, Range
    nonzero_digit = Any('123456789')
    digit = Any('0123456789')
    bindigit = Any('01')
    octdigit = Any('01234567')
    hexdigit = Any('0123456789ABCDEFabcdef')
    indentation = Bol + Rep(Any(' \t'))
    unicode_start_character = Any(unicode_start_ch_any) | Range(unicode_start_ch_range)
    unicode_continuation_character = unicode_start_character | Any(unicode_continuation_ch_any) | Range(unicode_continuation_ch_range)

    def underscore_digits(d):
        if False:
            while True:
                i = 10
        return Rep1(d) + Rep(Str('_') + Rep1(d))

    def prefixed_digits(prefix, digits):
        if False:
            print('Hello World!')
        return prefix + Opt(Str('_')) + underscore_digits(digits)
    decimal = underscore_digits(digit)
    dot = Str('.')
    exponent = Any('Ee') + Opt(Any('+-')) + decimal
    decimal_fract = decimal + dot + Opt(decimal) | dot + decimal
    name = unicode_start_character + Rep(unicode_continuation_character)
    intconst = prefixed_digits(nonzero_digit, digit) | Str('0') + (prefixed_digits(Any('Xx'), hexdigit) | prefixed_digits(Any('Oo'), octdigit) | prefixed_digits(Any('Bb'), bindigit)) | underscore_digits(Str('0')) | Rep1(digit)
    intsuffix = Opt(Any('Uu')) + Opt(Any('Ll')) + Opt(Any('Ll')) | Opt(Any('Ll')) + Opt(Any('Ll')) + Opt(Any('Uu'))
    intliteral = intconst + intsuffix
    fltconst = decimal_fract + Opt(exponent) | decimal + exponent
    imagconst = (intconst | fltconst) + Any('jJ')
    beginstring = Opt(Rep(Any(string_prefixes + raw_prefixes)) | Any(char_prefixes)) + (Str("'") | Str('"') | Str("'''") | Str('"""'))
    two_oct = octdigit + octdigit
    three_oct = octdigit + octdigit + octdigit
    two_hex = hexdigit + hexdigit
    four_hex = two_hex + two_hex
    escapeseq = Str('\\') + (two_oct | three_oct | Str('N{') + Rep(AnyBut('}')) + Str('}') | Str('u') + four_hex | Str('x') + two_hex | Str('U') + four_hex + four_hex | AnyChar)
    bra = Any('([{')
    ket = Any(')]}')
    ellipsis = Str('...')
    punct = Any(':,;+-*/|&<>=.%`~^?!@')
    diphthong = Str('==', '<>', '!=', '<=', '>=', '<<', '>>', '**', '//', '+=', '-=', '*=', '/=', '%=', '|=', '^=', '&=', '<<=', '>>=', '**=', '//=', '->', '@=', '&&', '||', ':=')
    spaces = Rep1(Any(' \t\x0c'))
    escaped_newline = Str('\\\n')
    lineterm = Eol + Opt(Str('\n'))
    comment = Str('#') + Rep(AnyBut('\n'))
    return Lexicon([(name, Method('normalize_ident')), (intliteral, Method('strip_underscores', symbol='INT')), (fltconst, Method('strip_underscores', symbol='FLOAT')), (imagconst, Method('strip_underscores', symbol='IMAG')), (ellipsis | punct | diphthong, TEXT), (bra, Method('open_bracket_action')), (ket, Method('close_bracket_action')), (lineterm, Method('newline_action')), (beginstring, Method('begin_string_action')), (comment, IGNORE), (spaces, IGNORE), (escaped_newline, IGNORE), State('INDENT', [(comment + lineterm, Method('commentline')), (Opt(spaces) + Opt(comment) + lineterm, IGNORE), (indentation, Method('indentation_action')), (Eof, Method('eof_action'))]), State('SQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('\'"\n\\')), 'CHARS'), (Str('"'), 'CHARS'), (Str('\n'), Method('unclosed_string_action')), (Str("'"), Method('end_string_action')), (Eof, 'EOF')]), State('DQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('"\n\\')), 'CHARS'), (Str("'"), 'CHARS'), (Str('\n'), Method('unclosed_string_action')), (Str('"'), Method('end_string_action')), (Eof, 'EOF')]), State('TSQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('\'"\n\\')), 'CHARS'), (Any('\'"'), 'CHARS'), (Str('\n'), 'NEWLINE'), (Str("'''"), Method('end_string_action')), (Eof, 'EOF')]), State('TDQ_STRING', [(escapeseq, 'ESCAPE'), (Rep1(AnyBut('"\'\n\\')), 'CHARS'), (Any('\'"'), 'CHARS'), (Str('\n'), 'NEWLINE'), (Str('"""'), Method('end_string_action')), (Eof, 'EOF')]), (Eof, Method('eof_action'))])
unicode_start_ch_any = u'_ªµºˬˮͿΆΌՙەۿܐޱߺࠚࠤࠨऽॐলঽৎৼਫ਼ઽૐૹଽୱஃஜௐఽ\u0c5dಀಽഽൎලาຄລາຽໆༀဿၡႎჇჍቘዀៗៜᢪᪧᳺὙὛὝιⁱⁿℂℇℕℤΩℨⅎⴧⴭⵯ\ua7d3ꣻꧏꩺꪱꫀꫂיִמּﹱﹳﹷﹹﹻﹽ𐠈𐠼𐨀𐼧\U00011075𑅄𑅇𑅶𑇚𑇜𑊈𑌽𑍐𑓇𑙄𑚸𑤉𑤿𑥁𑧡𑧣𑨀𑨺𑩐𑪝𑱀𑵆𑶘\U00011f02𑾰𖽐𖿣\U0001b132\U0001b155𝒢𝒻𝕆𞅎𞥋𞸤𞸧𞸹𞸻𞹂𞹇𞹉𞹋𞹔𞹗𞹙𞹛𞹝𞹟𞹤𞹾'
unicode_start_ch_range = u'AZazÀÖØöøˁˆˑˠˤͰʹͶͷͻͽΈΊΎΡΣϵϷҁҊԯԱՖՠֈאתׯײؠيٮٯٱۓۥۦۮۯۺۼܒܯݍޥߊߪߴߵࠀࠕࡀࡘࡠࡪ\u0870\u0887\u0889\u088eࢠ\u08c9ऄहक़ॡॱঀঅঌএঐওনপরশহড়ঢ়য়ৡৰৱਅਊਏਐਓਨਪਰਲਲ਼ਵਸ਼ਸਹਖ਼ੜੲੴઅઍએઑઓનપરલળવહૠૡଅଌଏଐଓନପରଲଳଵହଡ଼ଢ଼ୟୡஅஊஎஐஒகஙசஞடணதநபமஹఅఌఎఐఒనపహౘౚౠౡಅಌಎಐಒನಪಳವಹ\u0cddೞೠೡೱೲഄഌഎഐഒഺൔൖൟൡൺൿඅඖකනඳරවෆกะเๆກຂຆຊຌຣວະເໄໜໟཀཇཉཬྈྌကဪၐၕၚၝၥၦၮၰၵႁႠჅაჺჼቈቊቍቐቖቚቝበኈኊኍነኰኲኵኸኾዂዅወዖዘጐጒጕጘፚᎀᎏᎠᏵᏸᏽᐁᙬᙯᙿᚁᚚᚠᛪᛮᛸᜀᜑ\u171fᜱᝀᝑᝠᝬᝮᝰកឳᠠᡸᢀᢨᢰᣵᤀᤞᥐᥭᥰᥴᦀᦫᦰᧉᨀᨖᨠᩔᬅᬳᭅ\u1b4cᮃᮠᮮᮯᮺᯥᰀᰣᱍᱏᱚᱽᲀᲈᲐᲺᲽᲿᳩᳬᳮᳳᳵᳶᴀᶿḀἕἘἝἠὅὈὍὐὗὟώᾀᾴᾶᾼῂῄῆῌῐΐῖΊῠῬῲῴῶῼₐₜℊℓ℘ℝKℹℼℿⅅⅉⅠↈⰀⳤⳫⳮⳲⳳⴀⴥⴰⵧⶀⶖⶠⶦⶨⶮⶰⶶⶸⶾⷀⷆⷈⷎⷐⷖⷘⷞ々〇〡〩〱〵〸〼ぁゖゝゟァヺーヿㄅㄯㄱㆎㆠㆿㇰㇿ㐀䶿一ꒌꓐꓽꔀꘌꘐꘟꘪꘫꙀꙮꙿꚝꚠꛯꜗꜟꜢꞈꞋꟊ\ua7d0\ua7d1\ua7d5\ua7d9\ua7f2ꠁꠃꠅꠇꠊꠌꠢꡀꡳꢂꢳꣲꣷꣽꣾꤊꤥꤰꥆꥠꥼꦄꦲꧠꧤꧦꧯꧺꧾꨀꨨꩀꩂꩄꩋꩠꩶꩾꪯꪵꪶꪹꪽꫛꫝꫠꫪꫲꫴꬁꬆꬉꬎꬑꬖꬠꬦꬨꬮꬰꭚꭜꭩꭰꯢ가힣ힰퟆퟋퟻ豈舘並龎ﬀﬆﬓﬗײַﬨשׁזּטּלּנּסּףּפּצּﮱﯓﱝﱤﴽﵐﶏﶒﷇﷰﷹﹿﻼＡＺａｚｦﾝﾠﾾￂￇￊￏￒￗￚￜ𐀀𐀋𐀍𐀦𐀨𐀺𐀼𐀽𐀿𐁍𐁐𐁝𐂀𐃺𐅀𐅴𐊀𐊜𐊠𐋐𐌀𐌟𐌭𐍊𐍐𐍵𐎀𐎝𐎠𐏃𐏈𐏏𐏑𐏕𐐀𐒝𐒰𐓓𐓘𐓻𐔀𐔧𐔰𐕣\U00010570\U0001057a\U0001057c\U0001058a\U0001058c\U00010592\U00010594\U00010595\U00010597\U000105a1\U000105a3\U000105b1\U000105b3\U000105b9\U000105bb\U000105bc𐘀𐜶𐝀𐝕𐝠𐝧\U00010780\U00010785\U00010787\U000107b0\U000107b2\U000107ba𐠀𐠅𐠊𐠵𐠷𐠸𐠿𐡕𐡠𐡶𐢀𐢞𐣠𐣲𐣴𐣵𐤀𐤕𐤠𐤹𐦀𐦷𐦾𐦿𐨐𐨓𐨕𐨗𐨙𐨵𐩠𐩼𐪀𐪜𐫀𐫇𐫉𐫤𐬀𐬵𐭀𐭕𐭠𐭲𐮀𐮑𐰀𐱈𐲀𐲲𐳀𐳲𐴀𐴣𐺀𐺩𐺰𐺱𐼀𐼜𐼰𐽅\U00010f70\U00010f81𐾰𐿄𐿠𐿶𑀃𑀷\U00011071\U00011072𑂃𑂯𑃐𑃨𑄃𑄦𑅐𑅲𑆃𑆲𑇁𑇄𑈀𑈑𑈓𑈫\U0001123f\U00011240𑊀𑊆𑊊𑊍𑊏𑊝𑊟𑊨𑊰𑋞𑌅𑌌𑌏𑌐𑌓𑌨𑌪𑌰𑌲𑌳𑌵𑌹𑍝𑍡𑐀𑐴𑑇𑑊𑑟𑑡𑒀𑒯𑓄𑓅𑖀𑖮𑗘𑗛𑘀𑘯𑚀𑚪𑜀𑜚\U00011740\U00011746𑠀𑠫𑢠𑣟𑣿𑤆𑤌𑤓𑤕𑤖𑤘𑤯𑦠𑦧𑦪𑧐𑨋𑨲𑩜𑪉\U00011ab0𑫸𑰀𑰈𑰊𑰮𑱲𑲏𑴀𑴆𑴈𑴉𑴋𑴰𑵠𑵥𑵧𑵨𑵪𑶉𑻠𑻲\U00011f04\U00011f10\U00011f12\U00011f33𒀀𒎙𒐀𒑮𒒀𒕃\U00012f90\U00012ff0𓀀\U0001342f\U00013441\U00013446𔐀𔙆𖠀𖨸𖩀𖩞\U00016a70\U00016abe𖫐𖫭𖬀𖬯𖭀𖭃𖭣𖭷𖭽𖮏𖹀𖹿𖼀𖽊𖾓𖾟𖿠𖿡𗀀𘟷𘠀𘳕𘴀𘴈\U0001aff0\U0001aff3\U0001aff5\U0001affb\U0001affd\U0001affe𛀀\U0001b122𛅐𛅒𛅤𛅧𛅰𛋻𛰀𛱪𛱰𛱼𛲀𛲈𛲐𛲙𝐀𝑔𝑖𝒜𝒞𝒟𝒥𝒦𝒩𝒬𝒮𝒹𝒽𝓃𝓅𝔅𝔇𝔊𝔍𝔔𝔖𝔜𝔞𝔹𝔻𝔾𝕀𝕄𝕊𝕐𝕒𝚥𝚨𝛀𝛂𝛚𝛜𝛺𝛼𝜔𝜖𝜴𝜶𝝎𝝐𝝮𝝰𝞈𝞊𝞨𝞪𝟂𝟄𝟋\U0001df00\U0001df1e\U0001df25\U0001df2a\U0001e030\U0001e06d𞄀𞄬𞄷𞄽\U0001e290\U0001e2ad𞋀𞋫\U0001e4d0\U0001e4eb\U0001e7e0\U0001e7e6\U0001e7e8\U0001e7eb\U0001e7ed\U0001e7ee\U0001e7f0\U0001e7fe𞠀𞣄𞤀𞥃𞸀𞸃𞸅𞸟𞸡𞸢𞸩𞸲𞸴𞸷𞹍𞹏𞹑𞹒𞹡𞹢𞹧𞹪𞹬𞹲𞹴𞹷𞹹𞹼𞺀𞺉𞺋𞺛𞺡𞺣𞺥𞺩𞺫𞺻𠀀\U0002a6df𪜀\U0002b739𫝀𫠝𫠠𬺡𬺰𮯠丽𪘀𰀀𱍊'
unicode_continuation_ch_any = u'··়ׇֿٰܑ߽ৗ਼৾ੑੵ઼଼ஂௗ\u0c3c಼\u0cf3ൗ්ූัັ༹༵༷࿆᳭ᢩ៝᳴⁔⵿⃡꙯ꠂ꠆ꠋ꠬ꧥꩃﬞꪰ꫁＿𐨿𐇽𐋠\U000110c2𑅳𑈾\U00011241𑍗𑑞𑥀𑧤𑩇𑴺𑵇\U00011f03\U00013440𖽏𖿤𝩵𝪄\U0001e08f\U0001e2ae'
unicode_continuation_ch_range = u'09ֽׁׂًؚ֑ׅ̀ͯ҃҇ׄؐ٩۪ۭۖۜ۟ۤۧۨ۰۹ܰ݊ަް߀߉࡙࡛߫߳ࠖ࠙ࠛࠣࠥࠧࠩ࠭\u0898\u089f\u08caࣣ࣡ःऺ़ाॏ॑ॗॢॣ०९ঁঃাৄেৈো্ৢৣ০৯ਁਃਾੂੇੈੋ੍੦ੱઁઃાૅેૉો્ૢૣ૦૯ૺ૿ଁଃାୄେୈୋ୍୕ୗୢୣ୦୯ாூெைொ்௦௯ఀఄాౄెైొ్ౕౖౢౣ౦౯ಁಃಾೄೆೈೊ್ೕೖೢೣ೦೯ഀഃ഻഼ാൄെൈൊ്ൢൣ൦൯ඁඃාුෘෟ෦෯ෲෳำฺ็๎๐๙ຳຼ່\u0ece໐໙༘༙༠༩༾༿྄ཱ྆྇ྍྗྙྼါှ၀၉ၖၙၞၠၢၤၧၭၱၴႂႍႏႝ፝፟፩፱ᜒ\u1715ᜲ᜴ᝒᝓᝲᝳ឴៓០៩᠋᠍\u180f᠙ᤠᤫᤰ᤻᥆᥏᧐᧚ᨗᨛᩕᩞ᩠᩿᩼᪉᪐᪙᪽ᪿ᪰\u1aceᬀᬄ᬴᭄᭐᭙᭫᭳ᮀᮂᮡᮭ᮰᮹᯦᯳ᰤ᰷᱀᱉᱐᱙᳔᳨᳐᳒᳷᷿᳹᷀‿⁀⃥゙゚〪〯⃐⃜⃰⳯⳱ⷠⷿ꘠꘩ꙴ꙽ꚞꚟ꛰꛱ꠣꠧꢀꢁꢴꣅ꣐꣙꣠꣱ꣿ꤉ꤦ꤭ꥇ꥓ꦀꦃ꦳꧀꧐꧙꧰꧹ꨩꨶꩌꩍ꩐꩙ꩻꩽꪴꪲꪷꪸꪾ꪿ꫫꫯꫵ꫶ꯣꯪ꯬꯭꯰꯹︀️︠︯︳︴﹍﹏０９ﾞﾟ𐍶𐍺𐒠𐒩𐨁𐨃𐨅𐨆𐨌𐨺𐫦𐨏𐨸𐫥𐴤𐴧𐴰𐴹𐺫𐺬\U00010efd\U00010eff𐽆𐽐\U00010f82\U00010f85𑀀𑀂𑀸𑁆𑁦\U00011070\U00011073\U00011074𑁿𑂂𑂰𑂺𑃰𑃹𑄀𑄂𑄧𑄴𑄶𑄿𑅅𑅆𑆀𑆂𑆳𑇀𑇉𑇌𑇎𑇙𑈬𑈷𑋟𑋪𑋰𑋹𑌀𑌃𑌻𑌼𑌾𑍄𑍇𑍈𑍋𑍍𑍢𑍣𑍦𑍬𑍰𑍴𑐵𑑆𑑐𑑙𑒰𑓃𑓐𑓙𑖯𑖵𑖸𑗀𑗜𑗝𑘰𑙀𑙐𑙙𑚫𑚷𑛀𑛉𑜝𑜫𑜰𑜹𑠬𑠺𑣠𑣩𑤰𑤵𑤷𑤸𑤻𑤾𑥂𑥃𑥐𑥙𑧑𑧗𑧚𑧠𑨁𑨊𑨳𑨹𑨻𑨾𑩑𑩛𑪊𑪙𑰯𑰶𑰸𑰿𑱐𑱙𑲒𑲧𑲩𑲶𑴱𑴶𑴼𑴽𑴿𑵅𑵐𑵙𑶊𑶎𑶐𑶑𑶓𑶗𑶠𑶩𑻳𑻶\U00011f00\U00011f01\U00011f34\U00011f3a\U00011f3e\U00011f42\U00011f50\U00011f59\U00013447\U00013455𖩠𖩩\U00016ac0\U00016ac9𖫰𖫴𖬰𖬶𖭐𖭙𖽑𖾇𖾏𖾒𖿰𖿱𛲝𛲞\U0001cf00\U0001cf2d\U0001cf30\U0001cf46𝅩𝅥𝅲𝅻𝆂𝆋𝅭𝆅𝆪𝆭𝉂𝉄𝟎𝟿𝨀𝨶𝨻𝩬𝪛𝪟𝪡𝪯𞀀𞀆𞀈𞀘𞀛𞀡𞀣𞀤𞀦𞀪𞄰𞄶𞅀𞅉𞋬𞋹\U0001e4ec\U0001e4f9𞥊𞣐𞣖𞥄𞥐𞥙🯰🯹'