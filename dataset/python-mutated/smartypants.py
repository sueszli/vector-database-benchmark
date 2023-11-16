__author__ = 'Chad Miller <smartypantspy@chad.org>, Kovid Goyal <kovid at kovidgoyal.net>'
__description__ = 'Smart-quotes, smart-ellipses, and smart-dashes for weblog entries in pyblosxom'
'\n==============\nsmartypants.py\n==============\n\n----------------------------\nSmartyPants ported to Python\n----------------------------\n\nPorted by `Chad Miller`_\nCopyright (c) 2004, 2007 Chad Miller\n\noriginal `SmartyPants`_ by `John Gruber`_\nCopyright (c) 2003 John Gruber\n\n\nSynopsis\n========\n\nA smart-quotes plugin for Pyblosxom_.\n\nThe priginal "SmartyPants" is a free web publishing plug-in for Movable Type,\nBlosxom, and BBEdit that easily translates plain ASCII punctuation characters\ninto "smart" typographic punctuation HTML entities.\n\nThis software, *smartypants.py*, endeavours to be a functional port of\nSmartyPants to Python, for use with Pyblosxom_.\n\n\nDescription\n===========\n\nSmartyPants can perform the following transformations:\n\n- Straight quotes ( " and \' ) into "curly" quote HTML entities\n- Backticks-style quotes (\\`\\`like this\'\') into "curly" quote HTML entities\n- Dashes (``--`` and ``---``) into en- and em-dash entities\n- Three consecutive dots (``...`` or ``. . .``) into an ellipsis entity\n\nThis means you can write, edit, and save your posts using plain old\nASCII straight quotes, plain dashes, and plain dots, but your published\nposts (and final HTML output) will appear with smart quotes, em-dashes,\nand proper ellipses.\n\nSmartyPants does not modify characters within ``<pre>``, ``<code>``, ``<kbd>``,\n``<math>`` or ``<script>`` tag blocks. Typically, these tags are used to\ndisplay text where smart quotes and other "smart punctuation" would not be\nappropriate, such as source code or example markup.\n\n\nBackslash Escapes\n=================\n\nIf you need to use literal straight quotes (or plain hyphens and\nperiods), SmartyPants accepts the following backslash escape sequences\nto force non-smart punctuation. It does so by transforming the escape\nsequence into a decimal-encoded HTML entity:\n\n(FIXME:  table here.)\n\n.. comment    It sucks that there\'s a disconnect between the visual layout and table markup when special characters are involved.\n.. comment ======  =====  =========\n.. comment Escape  Value  Character\n.. comment ======  =====  =========\n.. comment \\\\\\\\\\\\\\\\    &#92;  \\\\\\\\\n.. comment \\\\\\\\"     &#34;  "\n.. comment \\\\\\\\\'     &#39;  \'\n.. comment \\\\\\\\.     &#46;  .\n.. comment \\\\\\\\-     &#45;  \\-\n.. comment \\\\\\\\`     &#96;  \\`\n.. comment ======  =====  =========\n\nThis is useful, for example, when you want to use straight quotes as\nfoot and inch marks: 6\'2" tall; a 17" iMac.\n\nOptions\n=======\n\nFor Pyblosxom users, the ``smartypants_attributes`` attribute is where you\nspecify configuration options.\n\nNumeric values are the easiest way to configure SmartyPants\' behavior:\n\n"0"\n    Suppress all transformations. (Do nothing.)\n"1"\n    Performs default SmartyPants transformations: quotes (including\n    \\`\\`backticks\'\' -style), em-dashes, and ellipses. "``--``" (dash dash)\n    is used to signify an em-dash; there is no support for en-dashes.\n\n"2"\n    Same as smarty_pants="1", except that it uses the old-school typewriter\n    shorthand for dashes:  "``--``" (dash dash) for en-dashes, "``---``"\n    (dash dash dash)\n    for em-dashes.\n\n"3"\n    Same as smarty_pants="2", but inverts the shorthand for dashes:\n    "``--``" (dash dash) for em-dashes, and "``---``" (dash dash dash) for\n    en-dashes.\n\n"-1"\n    Stupefy mode. Reverses the SmartyPants transformation process, turning\n    the HTML entities produced by SmartyPants into their ASCII equivalents.\n    E.g.  "&#8220;" is turned into a simple double-quote ("), "&#8212;" is\n    turned into two dashes, etc.\n\n\nThe following single-character attribute values can be combined to toggle\nindividual transformations from within the smarty_pants attribute. For\nexample, to educate normal quotes and em-dashes, but not ellipses or\n\\`\\`backticks\'\' -style quotes:\n\n``py[\'smartypants_attributes\'] = "1"``\n\n"q"\n    Educates normal quote characters: (") and (\').\n\n"b"\n    Educates \\`\\`backticks\'\' -style double quotes.\n\n"B"\n    Educates \\`\\`backticks\'\' -style double quotes and \\`single\' quotes.\n\n"d"\n    Educates em-dashes.\n\n"D"\n    Educates em-dashes and en-dashes, using old-school typewriter shorthand:\n    (dash dash) for en-dashes, (dash dash dash) for em-dashes.\n\n"i"\n    Educates em-dashes and en-dashes, using inverted old-school typewriter\n    shorthand: (dash dash) for em-dashes, (dash dash dash) for en-dashes.\n\n"e"\n    Educates ellipses.\n\n"w"\n    Translates any instance of ``&quot;`` into a normal double-quote character.\n    This should be of no interest to most people, but of particular interest\n    to anyone who writes their posts using Dreamweaver, as Dreamweaver\n    inexplicably uses this entity to represent a literal double-quote\n    character. SmartyPants only educates normal quotes, not entities (because\n    ordinarily, entities are used for the explicit purpose of representing the\n    specific character they represent). The "w" option must be used in\n    conjunction with one (or both) of the other quote options ("q" or "b").\n    Thus, if you wish to apply all SmartyPants transformations (quotes, en-\n    and em-dashes, and ellipses) and also translate ``&quot;`` entities into\n    regular quotes so SmartyPants can educate them, you should pass the\n    following to the smarty_pants attribute:\n\nThe ``smartypants_forbidden_flavours`` list contains pyblosxom flavours for\nwhich no Smarty Pants rendering will occur.\n\n\nCaveats\n=======\n\nWhy You Might Not Want to Use Smart Quotes in Your Weblog\n---------------------------------------------------------\n\nFor one thing, you might not care.\n\nMost normal, mentally stable individuals do not take notice of proper\ntypographic punctuation. Many design and typography nerds, however, break\nout in a nasty rash when they encounter, say, a restaurant sign that uses\na straight apostrophe to spell "Joe\'s".\n\nIf you\'re the sort of person who just doesn\'t care, you might well want to\ncontinue not caring. Using straight quotes -- and sticking to the 7-bit\nASCII character set in general -- is certainly a simpler way to live.\n\nEven if you I *do* care about accurate typography, you still might want to\nthink twice before educating the quote characters in your weblog. One side\neffect of publishing curly quote HTML entities is that it makes your\nweblog a bit harder for others to quote from using copy-and-paste. What\nhappens is that when someone copies text from your blog, the copied text\ncontains the 8-bit curly quote characters (as well as the 8-bit characters\nfor em-dashes and ellipses, if you use these options). These characters\nare not standard across different text encoding methods, which is why they\nneed to be encoded as HTML entities.\n\nPeople copying text from your weblog, however, may not notice that you\'re\nusing curly quotes, and they\'ll go ahead and paste the unencoded 8-bit\ncharacters copied from their browser into an email message or their own\nweblog. When pasted as raw "smart quotes", these characters are likely to\nget mangled beyond recognition.\n\nThat said, my own opinion is that any decent text editor or email client\nmakes it easy to stupefy smart quote characters into their 7-bit\nequivalents, and I don\'t consider it my problem if you\'re using an\nindecent text editor or email client.\n\n\nAlgorithmic Shortcomings\n------------------------\n\nOne situation in which quotes will get curled the wrong way is when\napostrophes are used at the start of leading contractions. For example:\n\n``\'Twas the night before Christmas.``\n\nIn the case above, SmartyPants will turn the apostrophe into an opening\nsingle-quote, when in fact it should be a closing one. I don\'t think\nthis problem can be solved in the general case -- every word processor\nI\'ve tried gets this wrong as well. In such cases, it\'s best to use the\nproper HTML entity for closing single-quotes (``&#8217;``) by hand.\n\n\nBugs\n====\n\nTo file bug reports or feature requests (other than topics listed in the\nCaveats section above) please send email to: mailto:smartypantspy@chad.org\n\nIf the bug involves quotes being curled the wrong way, please send example\ntext to illustrate.\n\nTo Do list\n----------\n\n- Provide a function for use within templates to quote anything at all.\n\n\nVersion History\n===============\n\n1.5_1.6: Fri, 27 Jul 2007 07:06:40 -0400\n    - Fixed bug where blocks of precious unalterable text was instead\n      interpreted.  Thanks to Le Roux and Dirk van Oosterbosch.\n\n1.5_1.5: Sat, 13 Aug 2005 15:50:24 -0400\n    - Fix bogus magical quotation when there is no hint that the\n      user wants it, e.g., in "21st century".  Thanks to Nathan Hamblen.\n    - Be smarter about quotes before terminating numbers in an en-dash\'ed\n      range.\n\n1.5_1.4: Thu, 10 Feb 2005 20:24:36 -0500\n    - Fix a date-processing bug, as reported by jacob childress.\n    - Begin a test-suite for ensuring correct output.\n    - Removed import of "string", since I didn\'t really need it.\n      (This was my first every Python program.  Sue me!)\n\n1.5_1.3: Wed, 15 Sep 2004 18:25:58 -0400\n    - Abort processing if the flavour is in forbidden-list.  Default of\n      [ "rss" ]   (Idea of Wolfgang SCHNERRING.)\n    - Remove stray virgules from en-dashes.  Patch by Wolfgang SCHNERRING.\n\n1.5_1.2: Mon, 24 May 2004 08:14:54 -0400\n    - Some single quotes weren\'t replaced properly.  Diff-tesuji played\n      by Benjamin GEIGER.\n\n1.5_1.1: Sun, 14 Mar 2004 14:38:28 -0500\n    - Support upcoming pyblosxom 0.9 plugin verification feature.\n\n1.5_1.0: Tue, 09 Mar 2004 08:08:35 -0500\n    - Initial release\n\nVersion Information\n-------------------\n\nVersion numbers will track the SmartyPants_ version numbers, with the addition\nof an underscore and the smartypants.py version on the end.\n\nNew versions will be available at `http://wiki.chad.org/SmartyPantsPy`_\n\n.. _http://wiki.chad.org/SmartyPantsPy: http://wiki.chad.org/SmartyPantsPy\n\nAuthors\n=======\n\n`John Gruber`_ did all of the hard work of writing this software in Perl for\n`Movable Type`_ and almost all of this useful documentation.  `Chad Miller`_\nported it to Python to use with Pyblosxom_.\n\n\nAdditional Credits\n==================\n\nPortions of the SmartyPants original work are based on Brad Choate\'s nifty\nMTRegex plug-in.  `Brad Choate`_ also contributed a few bits of source code to\nthis plug-in.  Brad Choate is a fine hacker indeed.\n\n`Jeremy Hedley`_ and `Charles Wiltgen`_ deserve mention for exemplary beta\ntesting of the original SmartyPants.\n\n`Rael Dornfest`_ ported SmartyPants to Blosxom.\n\n.. _Brad Choate: http://bradchoate.com/\n.. _Jeremy Hedley: http://antipixel.com/\n.. _Charles Wiltgen: http://playbacktime.com/\n.. _Rael Dornfest: http://raelity.org/\n\n\nCopyright and License\n=====================\n\nSmartyPants_ license::\n\n    Copyright (c) 2003 John Gruber\n    (https://daringfireball.net/)\n    All rights reserved.\n\n    Redistribution and use in source and binary forms, with or without\n    modification, are permitted provided that the following conditions are\n    met:\n\n    *   Redistributions of source code must retain the above copyright\n        notice, this list of conditions and the following disclaimer.\n\n    *   Redistributions in binary form must reproduce the above copyright\n        notice, this list of conditions and the following disclaimer in\n        the documentation and/or other materials provided with the\n        distribution.\n\n    *   Neither the name "SmartyPants" nor the names of its contributors\n        may be used to endorse or promote products derived from this\n        software without specific prior written permission.\n\n    This software is provided by the copyright holders and contributors "as\n    is" and any express or implied warranties, including, but not limited\n    to, the implied warranties of merchantability and fitness for a\n    particular purpose are disclaimed. In no event shall the copyright\n    owner or contributors be liable for any direct, indirect, incidental,\n    special, exemplary, or consequential damages (including, but not\n    limited to, procurement of substitute goods or services; loss of use,\n    data, or profits; or business interruption) however caused and on any\n    theory of liability, whether in contract, strict liability, or tort\n    (including negligence or otherwise) arising in any way out of the use\n    of this software, even if advised of the possibility of such damage.\n\n\nsmartypants.py license::\n\n    smartypants.py is a derivative work of SmartyPants.\n\n    Redistribution and use in source and binary forms, with or without\n    modification, are permitted provided that the following conditions are\n    met:\n\n    *   Redistributions of source code must retain the above copyright\n        notice, this list of conditions and the following disclaimer.\n\n    *   Redistributions in binary form must reproduce the above copyright\n        notice, this list of conditions and the following disclaimer in\n        the documentation and/or other materials provided with the\n        distribution.\n\n    This software is provided by the copyright holders and contributors "as\n    is" and any express or implied warranties, including, but not limited\n    to, the implied warranties of merchantability and fitness for a\n    particular purpose are disclaimed. In no event shall the copyright\n    owner or contributors be liable for any direct, indirect, incidental,\n    special, exemplary, or consequential damages (including, but not\n    limited to, procurement of substitute goods or services; loss of use,\n    data, or profits; or business interruption) however caused and on any\n    theory of liability, whether in contract, strict liability, or tort\n    (including negligence or otherwise) arising in any way out of the use\n    of this software, even if advised of the possibility of such damage.\n\n\n\n.. _John Gruber: https://daringfireball.net/\n.. _Chad Miller: http://web.chad.org/\n\n.. _Pyblosxom: http://roughingit.subtlehints.net/pyblosxom\n.. _SmartyPants: https://daringfireball.net/projects/smartypants/\n.. _Movable Type: http://www.movabletype.org/\n\n'
import re
tags_to_skip_regex = re.compile('<(/)?(style|pre|code|kbd|script|math)[^>]*>', re.I)
self_closing_regex = re.compile('/\\s*>$')

def parse_attr(attr):
    if False:
        while True:
            i = 10
    do_dashes = do_backticks = do_quotes = do_ellipses = do_stupefy = 0
    if attr == '1':
        do_quotes = 1
        do_backticks = 1
        do_dashes = 1
        do_ellipses = 1
    elif attr == '2':
        do_quotes = 1
        do_backticks = 1
        do_dashes = 2
        do_ellipses = 1
    elif attr == '3':
        do_quotes = 1
        do_backticks = 1
        do_dashes = 3
        do_ellipses = 1
    elif attr == '-1':
        do_stupefy = 1
    else:
        for c in attr:
            if c == 'q':
                do_quotes = 1
            elif c == 'b':
                do_backticks = 1
            elif c == 'B':
                do_backticks = 2
            elif c == 'd':
                do_dashes = 1
            elif c == 'D':
                do_dashes = 2
            elif c == 'i':
                do_dashes = 3
            elif c == 'e':
                do_ellipses = 1
            else:
                pass
    return (do_dashes, do_backticks, do_quotes, do_ellipses, do_stupefy)

def smartyPants(text, attr='1'):
    if False:
        print('Hello World!')
    if attr == '0':
        return text
    (do_dashes, do_backticks, do_quotes, do_ellipses, do_stupefy) = parse_attr(attr)
    dashes_func = {1: educateDashes, 2: educateDashesOldSchool, 3: educateDashesOldSchoolInverted}.get(do_dashes, lambda x: x)
    backticks_func = {1: educateBackticks, 2: lambda x: educateSingleBackticks(educateBackticks(x))}.get(do_backticks, lambda x: x)
    ellipses_func = {1: educateEllipses}.get(do_ellipses, lambda x: x)
    stupefy_func = {1: stupefyEntities}.get(do_stupefy, lambda x: x)
    skipped_tag_stack = []
    tokens = _tokenize(text)
    result = []
    in_pre = False
    prev_token_last_char = ''
    for cur_token in tokens:
        if cur_token[0] == 'tag':
            result.append(cur_token[1])
            skip_match = tags_to_skip_regex.match(cur_token[1])
            if skip_match is not None:
                is_self_closing = self_closing_regex.search(skip_match.group()) is not None
                if not is_self_closing:
                    if not skip_match.group(1):
                        skipped_tag_stack.append(skip_match.group(2).lower())
                        in_pre = True
                    else:
                        if len(skipped_tag_stack) > 0:
                            if skip_match.group(2).lower() == skipped_tag_stack[-1]:
                                skipped_tag_stack.pop()
                            else:
                                pass
                        if len(skipped_tag_stack) == 0:
                            in_pre = False
        else:
            t = cur_token[1]
            last_char = t[-1:]
            if not in_pre:
                t = processEscapes(t)
                t = re.sub('&quot;', '"', t)
                t = dashes_func(t)
                t = ellipses_func(t)
                t = backticks_func(t)
                if do_quotes != 0:
                    if t == "'":
                        if re.match('\\S', prev_token_last_char):
                            t = '&#8217;'
                        else:
                            t = '&#8216;'
                    elif t == '"':
                        if re.match('\\S', prev_token_last_char):
                            t = '&#8221;'
                        else:
                            t = '&#8220;'
                    else:
                        t = educateQuotes(t)
                t = stupefy_func(t)
            prev_token_last_char = last_char
            result.append(t)
    return ''.join(result)

def educateQuotes(text):
    if False:
        return 10
    '\n    Parameter:  String.\n\n    Returns:    The string, with "educated" curly quote HTML entities.\n\n    Example input:  "Isn\'t this fun?"\n    Example output: &#8220;Isn&#8217;t this fun?&#8221;\n    '
    punct_class = '[!"#\\$\\%\'()*+,-.\\/:;<=>?\\@\\[\\\\\\]\\^_`{|}~]'
    text = re.sub(f"^'(?={punct_class}\\\\B)", '&#8217;', text)
    text = re.sub(f'^"(?={punct_class}\\\\B)', '&#8221;', text)
    text = re.sub('"\'(?=\\w)', '&#8220;&#8216;', text)
    text = re.sub('\'"(?=\\w)', '&#8216;&#8220;', text)
    text = re.sub('""(?=\\w)', '&#8220;&#8220;', text)
    text = re.sub("''(?=\\w)", '&#8216;&#8216;', text)
    text = re.sub('\\"\\\'', '&#8221;&#8217;', text)
    text = re.sub('\\\'\\"', '&#8217;&#8221;', text)
    text = re.sub('""', '&#8221;&#8221;', text)
    text = re.sub("''", '&#8217;&#8217;', text)
    text = re.sub("(\\W|^)'(?=\\d{2}s)", '\\1&#8217;', text)
    text = re.sub('(\\W|^)([-0-9.]+\\s*)\'(\\s*[-0-9.]+)"', '\\1\\2&#8242;\\3&#8243;', text)
    text = re.sub('(?<=\\W)"(?=\\w)', '&#8220;', text)
    text = re.sub("(?<=\\W)'(?=\\w)", '&#8216;', text)
    text = re.sub('(?<=\\w)"(?=\\W)', '&#8221;', text)
    text = re.sub("(?<=\\w)'(?=\\W)", '&#8217;', text)
    close_class = '[^\\ \\t\\r\\n\\[\\{\\(\\-]'
    dec_dashes = '&#8211;|&#8212;'
    opening_single_quotes_regex = re.compile("\n            (\n                \\s          |   # a whitespace char, or\n                &nbsp;      |   # a non-breaking space entity, or\n                --          |   # dashes, or\n                &[mn]dash;  |   # named dash entities\n                {}          |   # or decimal entities\n                &\\#x201[34];    # or hex\n            )\n            '                 # the quote\n            (?=\\w)            # followed by a word character\n            ".format(dec_dashes), re.VERBOSE)
    text = opening_single_quotes_regex.sub('\\1&#8216;', text)
    closing_single_quotes_regex = re.compile("\n            ({})\n            '\n            (?!\\s | s\\b | \\d)\n            ".format(close_class), re.VERBOSE)
    text = closing_single_quotes_regex.sub('\\1&#8217;', text)
    closing_single_quotes_regex = re.compile("\n            ({})\n            '\n            (\\s | s\\b)\n            ".format(close_class), re.VERBOSE)
    text = closing_single_quotes_regex.sub('\\1&#8217;\\2', text)
    text = re.sub("'", '&#8216;', text)
    opening_double_quotes_regex = re.compile('\n            (\n                \\s          |   # a whitespace char, or\n                &nbsp;      |   # a non-breaking space entity, or\n                --          |   # dashes, or\n                &[mn]dash;  |   # named dash entities\n                {}          |   # or decimal entities\n                &\\#x201[34];    # or hex\n            )\n            "                 # the quote\n            (?=\\w)            # followed by a word character\n            '.format(dec_dashes), re.VERBOSE)
    text = opening_double_quotes_regex.sub('\\1&#8220;', text)
    closing_double_quotes_regex = re.compile('\n            #({})?   # character that indicates the quote should be closing\n            "\n            (?=\\s)\n            '.format(close_class), re.VERBOSE)
    text = closing_double_quotes_regex.sub('&#8221;', text)
    closing_double_quotes_regex = re.compile('\n            ({})   # character that indicates the quote should be closing\n            "\n            '.format(close_class), re.VERBOSE)
    text = closing_double_quotes_regex.sub('\\1&#8221;', text)
    if text.endswith('-"'):
        text = text[:-1] + '&#8221;'
    text = re.sub('"', '&#8220;', text)
    return text

def educateBackticks(text):
    if False:
        i = 10
        return i + 15
    "\n    Parameter:  String.\n    Returns:    The string, with ``backticks'' -style double quotes\n                translated into HTML curly quote entities.\n    Example input:  ``Isn't this fun?''\n    Example output: &#8220;Isn't this fun?&#8221;\n    "
    text = re.sub('``', '&#8220;', text)
    text = re.sub("''", '&#8221;', text)
    return text

def educateSingleBackticks(text):
    if False:
        return 10
    "\n    Parameter:  String.\n    Returns:    The string, with `backticks' -style single quotes\n                translated into HTML curly quote entities.\n\n    Example input:  `Isn't this fun?'\n    Example output: &#8216;Isn&#8217;t this fun?&#8217;\n    "
    text = re.sub('`', '&#8216;', text)
    text = re.sub("'", '&#8217;', text)
    return text

def educateDashes(text):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parameter:  String.\n\n    Returns:    The string, with each instance of "--" translated to\n                an em-dash HTML entity.\n    '
    text = re.sub('---', '&#8211;', text)
    text = re.sub('--', '&#8212;', text)
    return text

def educateDashesOldSchool(text):
    if False:
        return 10
    '\n    Parameter:  String.\n\n    Returns:    The string, with each instance of "--" translated to\n                an en-dash HTML entity, and each "---" translated to\n                an em-dash HTML entity.\n    '
    text = re.sub('---', '&#8212;', text)
    text = re.sub('--', '&#8211;', text)
    return text

def educateDashesOldSchoolInverted(text):
    if False:
        i = 10
        return i + 15
    '\n    Parameter:  String.\n\n    Returns:    The string, with each instance of "--" translated to\n                an em-dash HTML entity, and each "---" translated to\n                an en-dash HTML entity. Two reasons why: First, unlike the\n                en- and em-dash syntax supported by\n                EducateDashesOldSchool(), it\'s compatible with existing\n                entries written before SmartyPants 1.1, back when "--" was\n                only used for em-dashes.  Second, em-dashes are more\n                common than en-dashes, and so it sort of makes sense that\n                the shortcut should be shorter to type. (Thanks to Aaron\n                Swartz for the idea.)\n    '
    text = re.sub('---', '&#8211;', text)
    text = re.sub('--', '&#8212;', text)
    return text

def educateEllipses(text):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parameter:  String.\n    Returns:    The string, with each instance of "..." translated to\n                an ellipsis HTML entity.\n\n    Example input:  Huh...?\n    Example output: Huh&#8230;?\n    '
    text = re.sub('\\.\\.\\.', '&#8230;', text)
    text = re.sub('\\. \\. \\.', '&#8230;', text)
    return text

def stupefyEntities(text):
    if False:
        i = 10
        return i + 15
    '\n    Parameter:  String.\n    Returns:    The string, with each SmartyPants HTML entity translated to\n                its ASCII counterpart.\n\n    Example input:  &#8220;Hello &#8212; world.&#8221;\n    Example output: "Hello -- world."\n    '
    text = re.sub('&#8211;', '-', text)
    text = re.sub('&#8212;', '--', text)
    text = re.sub('&#8216;', "'", text)
    text = re.sub('&#8217;', "'", text)
    text = re.sub('&#8220;', '"', text)
    text = re.sub('&#8221;', '"', text)
    text = re.sub('&#8230;', '...', text)
    return text

def processEscapes(text):
    if False:
        i = 10
        return i + 15
    '\n    Parameter:  String.\n    Returns:    The string, with after processing the following backslash\n                escape sequences. This is useful if you want to force a "dumb"\n                quote or other character to appear.\n\n                Escape  Value\n                ------  -----\n                \\\\      &#92;\n                \\"      &#34;\n                \\\'      &#39;\n                \\.      &#46;\n                \\-      &#45;\n                \\`      &#96;\n    '
    text = re.sub('\\\\\\\\', '&#92;', text)
    text = re.sub('\\\\"', '&#34;', text)
    text = re.sub("\\\\'", '&#39;', text)
    text = re.sub('\\\\\\.', '&#46;', text)
    text = re.sub('\\\\-', '&#45;', text)
    text = re.sub('\\\\`', '&#96;', text)
    return text

def _tokenize(html):
    if False:
        print('Hello World!')
    '\n    Parameter:  String containing HTML markup.\n    Returns:    Reference to an array of the tokens comprising the input\n                string. Each token is either a tag (possibly with nested,\n                tags contained therein, such as <a href="<MTFoo>">, or a\n                run of text between tags. Each element of the array is a\n                two-element array; the first is either \'tag\' or \'text\';\n                the second is the actual value.\n\n    Based on the _tokenize() subroutine from Brad Choate\'s MTRegex plugin.\n        <http://www.bradchoate.com/past/mtregex.php>\n    '
    tokens = []
    tag_soup = re.compile('([^<]*)(<[^>]*>)')
    token_match = tag_soup.search(html)
    previous_end = 0
    while token_match is not None:
        if token_match.group(1):
            tokens.append(['text', token_match.group(1)])
        tokens.append(['tag', token_match.group(2)])
        previous_end = token_match.end()
        token_match = tag_soup.search(html, token_match.end())
    if previous_end < len(html):
        tokens.append(['text', html[previous_end:]])
    return tokens

def run_tests(return_tests=False):
    if False:
        i = 10
        return i + 15
    import unittest
    sp = smartyPants

    class TestSmartypantsAllAttributes(unittest.TestCase):

        def test_dates(self):
            if False:
                return 10
            self.assertEqual(sp("one two '60s"), 'one two &#8217;60s')
            self.assertEqual(sp("1440-80's"), '1440-80&#8217;s')
            self.assertEqual(sp("1440-'80s"), '1440-&#8217;80s')
            self.assertEqual(sp("1440---'80s"), '1440&#8211;&#8217;80s')
            self.assertEqual(sp('1960s'), '1960s')
            self.assertEqual(sp("1960's"), '1960&#8217;s')
            self.assertEqual(sp("one two '60s"), 'one two &#8217;60s')
            self.assertEqual(sp("'60s"), '&#8217;60s')

        def test_measurements(self):
            if False:
                while True:
                    i = 10
            ae = self.assertEqual
            ae(sp('one two 1.1\'2.2"'), 'one two 1.1&#8242;2.2&#8243;')
            ae(sp('1\' 2"'), '1&#8242; 2&#8243;')

        def test_skip_tags(self):
            if False:
                while True:
                    i = 10
            self.assertEqual(sp('<script type="text/javascript">\n<!--\nvar href = "http://www.google.com";\nvar linktext = "google";\ndocument.write(\'<a href="\' + href + \'">\' + linktext + "</a>");\n//-->\n</script>'), '<script type="text/javascript">\n<!--\nvar href = "http://www.google.com";\nvar linktext = "google";\ndocument.write(\'<a href="\' + href + \'">\' + linktext + "</a>");\n//-->\n</script>')
            self.assertEqual(sp("<p>He said &quot;Let's write some code.&quot; This code here <code>if True:\n\tprint &quot;Okay&quot;</code> is python code.</p>"), '<p>He said &#8220;Let&#8217;s write some code.&#8221; This code here <code>if True:\n\tprint &quot;Okay&quot;</code> is python code.</p>')
            self.assertEqual(sp("<script/><p>It's ok</p>"), '<script/><p>It&#8217;s ok</p>')

        def test_ordinal_numbers(self):
            if False:
                i = 10
                return i + 15
            self.assertEqual(sp('21st century'), '21st century')
            self.assertEqual(sp('3rd'), '3rd')

        def test_educated_quotes(self):
            if False:
                print('Hello World!')
            self.assertEqual(sp('"Isn\'t this fun?"'), '&#8220;Isn&#8217;t this fun?&#8221;')
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(TestSmartypantsAllAttributes)
    if return_tests:
        return tests
    unittest.TextTestRunner(verbosity=4).run(tests)
if __name__ == '__main__':
    run_tests()