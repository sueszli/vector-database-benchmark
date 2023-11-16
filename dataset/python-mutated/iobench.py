import itertools
import os
import platform
import re
import sys
import time
from optparse import OptionParser
out = sys.stdout
TEXT_ENCODING = 'utf8'
NEWLINES = 'lf'
try:
    xrange
except NameError:
    xrange = range

def text_open(fn, mode, encoding=None):
    if False:
        print('Hello World!')
    try:
        return open(fn, mode, encoding=encoding or TEXT_ENCODING)
    except TypeError:
        if 'r' in mode:
            mode += 'U'
        return open(fn, mode)

def get_file_sizes():
    if False:
        print('Hello World!')
    for s in ['20 KiB', '400 KiB', '10 MiB']:
        (size, unit) = s.split()
        size = int(size) * {'KiB': 1024, 'MiB': 1024 ** 2}[unit]
        yield (s.replace(' ', ''), size)

def get_binary_files():
    if False:
        for i in range(10):
            print('nop')
    return ((name + '.bin', size) for (name, size) in get_file_sizes())

def get_text_files():
    if False:
        while True:
            i = 10
    return (('%s-%s-%s.txt' % (name, TEXT_ENCODING, NEWLINES), size) for (name, size) in get_file_sizes())

def with_open_mode(mode):
    if False:
        print('Hello World!')

    def decorate(f):
        if False:
            i = 10
            return i + 15
        f.file_open_mode = mode
        return f
    return decorate

def with_sizes(*sizes):
    if False:
        for i in range(10):
            print('nop')

    def decorate(f):
        if False:
            i = 10
            return i + 15
        f.file_sizes = sizes
        return f
    return decorate

@with_open_mode('r')
@with_sizes('medium')
def read_bytewise(f):
    if False:
        return 10
    ' read one unit at a time '
    f.seek(0)
    while f.read(1):
        pass

@with_open_mode('r')
@with_sizes('medium')
def read_small_chunks(f):
    if False:
        return 10
    ' read 20 units at a time '
    f.seek(0)
    while f.read(20):
        pass

@with_open_mode('r')
@with_sizes('medium')
def read_big_chunks(f):
    if False:
        print('Hello World!')
    ' read 4096 units at a time '
    f.seek(0)
    while f.read(4096):
        pass

@with_open_mode('r')
@with_sizes('small', 'medium', 'large')
def read_whole_file(f):
    if False:
        for i in range(10):
            print('nop')
    ' read whole contents at once '
    f.seek(0)
    while f.read():
        pass

@with_open_mode('rt')
@with_sizes('medium')
def read_lines(f):
    if False:
        for i in range(10):
            print('nop')
    ' read one line at a time '
    f.seek(0)
    for line in f:
        pass

@with_open_mode('r')
@with_sizes('medium')
def seek_forward_bytewise(f):
    if False:
        i = 10
        return i + 15
    ' seek forward one unit at a time '
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    for i in xrange(0, size - 1):
        f.seek(i, 0)

@with_open_mode('r')
@with_sizes('medium')
def seek_forward_blockwise(f):
    if False:
        print('Hello World!')
    ' seek forward 1000 units at a time '
    f.seek(0, 2)
    size = f.tell()
    f.seek(0, 0)
    for i in xrange(0, size - 1, 1000):
        f.seek(i, 0)

@with_open_mode('rb')
@with_sizes('medium')
def read_seek_bytewise(f):
    if False:
        for i in range(10):
            print('nop')
    ' alternate read & seek one unit '
    f.seek(0)
    while f.read(1):
        f.seek(1, 1)

@with_open_mode('rb')
@with_sizes('medium')
def read_seek_blockwise(f):
    if False:
        i = 10
        return i + 15
    ' alternate read & seek 1000 units '
    f.seek(0)
    while f.read(1000):
        f.seek(1000, 1)

@with_open_mode('w')
@with_sizes('small')
def write_bytewise(f, source):
    if False:
        print('Hello World!')
    ' write one unit at a time '
    for i in xrange(0, len(source)):
        f.write(source[i:i + 1])

@with_open_mode('w')
@with_sizes('medium')
def write_small_chunks(f, source):
    if False:
        print('Hello World!')
    ' write 20 units at a time '
    for i in xrange(0, len(source), 20):
        f.write(source[i:i + 20])

@with_open_mode('w')
@with_sizes('medium')
def write_medium_chunks(f, source):
    if False:
        return 10
    ' write 4096 units at a time '
    for i in xrange(0, len(source), 4096):
        f.write(source[i:i + 4096])

@with_open_mode('w')
@with_sizes('large')
def write_large_chunks(f, source):
    if False:
        return 10
    ' write 1e6 units at a time '
    for i in xrange(0, len(source), 1000000):
        f.write(source[i:i + 1000000])

@with_open_mode('w+')
@with_sizes('small')
def modify_bytewise(f, source):
    if False:
        i = 10
        return i + 15
    ' modify one unit at a time '
    f.seek(0)
    for i in xrange(0, len(source)):
        f.write(source[i:i + 1])

@with_open_mode('w+')
@with_sizes('medium')
def modify_small_chunks(f, source):
    if False:
        print('Hello World!')
    ' modify 20 units at a time '
    f.seek(0)
    for i in xrange(0, len(source), 20):
        f.write(source[i:i + 20])

@with_open_mode('w+')
@with_sizes('medium')
def modify_medium_chunks(f, source):
    if False:
        print('Hello World!')
    ' modify 4096 units at a time '
    f.seek(0)
    for i in xrange(0, len(source), 4096):
        f.write(source[i:i + 4096])

@with_open_mode('wb+')
@with_sizes('medium')
def modify_seek_forward_bytewise(f, source):
    if False:
        return 10
    ' alternate write & seek one unit '
    f.seek(0)
    for i in xrange(0, len(source), 2):
        f.write(source[i:i + 1])
        f.seek(i + 2)

@with_open_mode('wb+')
@with_sizes('medium')
def modify_seek_forward_blockwise(f, source):
    if False:
        print('Hello World!')
    ' alternate write & seek 1000 units '
    f.seek(0)
    for i in xrange(0, len(source), 2000):
        f.write(source[i:i + 1000])
        f.seek(i + 2000)

@with_open_mode('wb+')
@with_sizes('medium')
def read_modify_bytewise(f, source):
    if False:
        while True:
            i = 10
    ' alternate read & write one unit '
    f.seek(0)
    for i in xrange(0, len(source), 2):
        f.read(1)
        f.write(source[i + 1:i + 2])

@with_open_mode('wb+')
@with_sizes('medium')
def read_modify_blockwise(f, source):
    if False:
        i = 10
        return i + 15
    ' alternate read & write 1000 units '
    f.seek(0)
    for i in xrange(0, len(source), 2000):
        f.read(1000)
        f.write(source[i + 1000:i + 2000])
read_tests = [read_bytewise, read_small_chunks, read_lines, read_big_chunks, None, read_whole_file, None, seek_forward_bytewise, seek_forward_blockwise, read_seek_bytewise, read_seek_blockwise]
write_tests = [write_bytewise, write_small_chunks, write_medium_chunks, write_large_chunks]
modify_tests = [modify_bytewise, modify_small_chunks, modify_medium_chunks, None, modify_seek_forward_bytewise, modify_seek_forward_blockwise, read_modify_bytewise, read_modify_blockwise]

def run_during(duration, func):
    if False:
        for i in range(10):
            print('nop')
    _t = time.time
    n = 0
    start = os.times()
    start_timestamp = _t()
    real_start = start[4] or start_timestamp
    while True:
        func()
        n += 1
        if _t() - start_timestamp > duration:
            break
    end = os.times()
    real = (end[4] if start[4] else time.time()) - real_start
    return (n, real, sum(end[0:2]) - sum(start[0:2]))

def warm_cache(filename):
    if False:
        print('Hello World!')
    with open(filename, 'rb') as f:
        f.read()

def run_all_tests(options):
    if False:
        for i in range(10):
            print('nop')

    def print_label(filename, func):
        if False:
            print('Hello World!')
        name = re.split('[-.]', filename)[0]
        out.write(('[%s] %s... ' % (name.center(7), func.__doc__.strip())).ljust(52))
        out.flush()

    def print_results(size, n, real, cpu):
        if False:
            while True:
                i = 10
        bw = n * float(size) / 1024 ** 2 / real
        bw = ('%4d MiB/s' if bw > 100 else '%.3g MiB/s') % bw
        out.write(bw.rjust(12) + '\n')
        if cpu < 0.9 * real:
            out.write('   warning: test above used only %d%% CPU, result may be flawed!\n' % (100.0 * cpu / real))

    def run_one_test(name, size, open_func, test_func, *args):
        if False:
            i = 10
            return i + 15
        mode = test_func.file_open_mode
        print_label(name, test_func)
        if 'w' not in mode or '+' in mode:
            warm_cache(name)
        with open_func(name) as f:
            (n, real, cpu) = run_during(1.5, lambda : test_func(f, *args))
        print_results(size, n, real, cpu)

    def run_test_family(tests, mode_filter, files, open_func, *make_args):
        if False:
            for i in range(10):
                print('nop')
        for test_func in tests:
            if test_func is None:
                out.write('\n')
                continue
            if mode_filter in test_func.file_open_mode:
                continue
            for s in test_func.file_sizes:
                (name, size) = files[size_names[s]]
                args = tuple((f(name, size) for f in make_args))
                run_one_test(name, size, open_func, test_func, *args)
    size_names = {'small': 0, 'medium': 1, 'large': 2}
    print('Python %s' % sys.version)
    if sys.version_info < (3, 3):
        if sys.maxunicode > 65535:
            text = 'UCS-4 (wide build)'
        else:
            text = 'UTF-16 (narrow build)'
    else:
        text = 'PEP 393'
    print('Unicode: %s' % text)
    print(platform.platform())
    binary_files = list(get_binary_files())
    text_files = list(get_text_files())
    if 'b' in options:
        print('Binary unit = one byte')
    if 't' in options:
        print('Text unit = one character (%s-decoded)' % TEXT_ENCODING)
    if 'b' in options and 'r' in options:
        print('\n** Binary input **\n')
        run_test_family(read_tests, 't', binary_files, lambda fn: open(fn, 'rb'))
    if 't' in options and 'r' in options:
        print('\n** Text input **\n')
        run_test_family(read_tests, 'b', text_files, lambda fn: text_open(fn, 'r'))
    if 'b' in options and 'w' in options:
        print('\n** Binary append **\n')

        def make_test_source(name, size):
            if False:
                i = 10
                return i + 15
            with open(name, 'rb') as f:
                return f.read()
        run_test_family(write_tests, 't', binary_files, lambda fn: open(os.devnull, 'wb'), make_test_source)
    if 't' in options and 'w' in options:
        print('\n** Text append **\n')

        def make_test_source(name, size):
            if False:
                for i in range(10):
                    print('nop')
            with text_open(name, 'r') as f:
                return f.read()
        run_test_family(write_tests, 'b', text_files, lambda fn: text_open(os.devnull, 'w'), make_test_source)
    if 'b' in options and 'w' in options:
        print('\n** Binary overwrite **\n')

        def make_test_source(name, size):
            if False:
                print('Hello World!')
            with open(name, 'rb') as f:
                return f.read()
        run_test_family(modify_tests, 't', binary_files, lambda fn: open(fn, 'r+b'), make_test_source)
    if 't' in options and 'w' in options:
        print('\n** Text overwrite **\n')

        def make_test_source(name, size):
            if False:
                return 10
            with text_open(name, 'r') as f:
                return f.read()
        run_test_family(modify_tests, 'b', text_files, lambda fn: text_open(fn, 'r+'), make_test_source)

def prepare_files():
    if False:
        i = 10
        return i + 15
    print('Preparing files...')
    for (name, size) in get_binary_files():
        if os.path.isfile(name) and os.path.getsize(name) == size:
            continue
        with open(name, 'wb') as f:
            f.write(os.urandom(size))
    chunk = []
    with text_open(__file__, 'r', encoding='utf8') as f:
        for line in f:
            if line.startswith('# <iobench text chunk marker>'):
                break
        else:
            raise RuntimeError("Couldn't find chunk marker in %s !" % __file__)
        if NEWLINES == 'all':
            it = itertools.cycle(['\n', '\r', '\r\n'])
        else:
            it = itertools.repeat({'cr': '\r', 'lf': '\n', 'crlf': '\r\n'}[NEWLINES])
        chunk = ''.join((line.replace('\n', next(it)) for line in f))
        if isinstance(chunk, bytes):
            chunk = chunk.decode('utf8')
        chunk = chunk.encode(TEXT_ENCODING)
    for (name, size) in get_text_files():
        if os.path.isfile(name) and os.path.getsize(name) == size:
            continue
        head = chunk * (size // len(chunk))
        tail = chunk[:size % len(chunk)]
        while True:
            try:
                tail.decode(TEXT_ENCODING)
                break
            except UnicodeDecodeError:
                tail = tail[:-1]
        with open(name, 'wb') as f:
            f.write(head)
            f.write(tail)

def main():
    if False:
        return 10
    global TEXT_ENCODING, NEWLINES
    usage = 'usage: %prog [-h|--help] [options]'
    parser = OptionParser(usage=usage)
    parser.add_option('-b', '--binary', action='store_true', dest='binary', default=False, help='run binary I/O tests')
    parser.add_option('-t', '--text', action='store_true', dest='text', default=False, help='run text I/O tests')
    parser.add_option('-r', '--read', action='store_true', dest='read', default=False, help='run read tests')
    parser.add_option('-w', '--write', action='store_true', dest='write', default=False, help='run write & modify tests')
    parser.add_option('-E', '--encoding', action='store', dest='encoding', default=None, help='encoding for text tests (default: %s)' % TEXT_ENCODING)
    parser.add_option('-N', '--newlines', action='store', dest='newlines', default='lf', help='line endings for text tests (one of: {lf (default), cr, crlf, all})')
    parser.add_option('-m', '--io-module', action='store', dest='io_module', default=None, help='io module to test (default: builtin open())')
    (options, args) = parser.parse_args()
    if args:
        parser.error('unexpected arguments')
    NEWLINES = options.newlines.lower()
    if NEWLINES not in ('lf', 'cr', 'crlf', 'all'):
        parser.error("invalid 'newlines' option: %r" % NEWLINES)
    test_options = ''
    if options.read:
        test_options += 'r'
    if options.write:
        test_options += 'w'
    elif not options.read:
        test_options += 'rw'
    if options.text:
        test_options += 't'
    if options.binary:
        test_options += 'b'
    elif not options.text:
        test_options += 'tb'
    if options.encoding:
        TEXT_ENCODING = options.encoding
    if options.io_module:
        globals()['open'] = __import__(options.io_module, {}, {}, ['open']).open
    prepare_files()
    run_all_tests(test_options)
if __name__ == '__main__':
    main()
'\n1.\nGáttir allar,\náðr gangi fram,\num skoðask skyli,\num skyggnast skyli,\nþví at óvíst er at vita,\nhvar óvinir\nsitja á fleti fyrir.\n\n2.\nGefendr heilir!\nGestr er inn kominn,\nhvar skal sitja sjá?\nMjök er bráðr,\nsá er á bröndum skal\nsíns of freista frama.\n\n3.\nElds er þörf,\nþeims inn er kominn\nok á kné kalinn;\nmatar ok váða\ner manni þörf,\nþeim er hefr um fjall farit.\n\n4.\nVatns er þörf,\nþeim er til verðar kemr,\nþerru ok þjóðlaðar,\ngóðs of æðis,\nef sér geta mætti,\norðs ok endrþögu.\n\n5.\nVits er þörf,\nþeim er víða ratar;\ndælt er heima hvat;\nat augabragði verðr,\nsá er ekki kann\nok með snotrum sitr.\n\n6.\nAt hyggjandi sinni\nskyli-t maðr hræsinn vera,\nheldr gætinn at geði;\nþá er horskr ok þögull\nkemr heimisgarða til,\nsjaldan verðr víti vörum,\nþví at óbrigðra vin\nfær maðr aldregi\nen mannvit mikit.\n\n7.\nInn vari gestr,\ner til verðar kemr,\nþunnu hljóði þegir,\neyrum hlýðir,\nen augum skoðar;\nsvá nýsisk fróðra hverr fyrir.\n\n8.\nHinn er sæll,\ner sér of getr\nlof ok líknstafi;\nódælla er við þat,\ner maðr eiga skal\nannars brjóstum í.\n'
"\nC'est revenir tard, je le sens, sur un sujet trop rebattu et déjà presque oublié. Mon état, qui ne me permet plus aucun travail suivi, mon aversion pour le genre polémique, ont causé ma lenteur à écrire et ma répugnance à publier. J'aurais même tout à fait supprimé ces Lettres, ou plutôt je lie les aurais point écrites, s'il n'eût été question que de moi : Mais ma patrie ne m'est pas tellement devenue étrangère que je puisse voir tranquillement opprimer ses citoyens, surtout lorsqu'ils n'ont compromis leurs droits qu'en défendant ma cause. Je serais le dernier des hommes si dans une telle occasion j'écoutais un sentiment qui n'est plus ni douceur ni patience, mais faiblesse et lâcheté, dans celui qu'il empêche de remplir son devoir.\nRien de moins important pour le public, j'en conviens, que la matière de ces lettres. La constitution d'une petite République, le sort d'un petit particulier, l'exposé de quelques injustices, la réfutation de quelques sophismes ; tout cela n'a rien en soi d'assez considérable pour mériter beaucoup de lecteurs : mais si mes sujets sont petits mes objets sont grands, et dignes de l'attention de tout honnête homme. Laissons Genève à sa place, et Rousseau dans sa dépression ; mais la religion, mais la liberté, la justice ! voilà, qui que vous soyez, ce qui n'est pas au-dessous de vous.\nQu'on ne cherche pas même ici dans le style le dédommagement de l'aridité de la matière. Ceux que quelques traits heureux de ma plume ont si fort irrités trouveront de quoi s'apaiser dans ces lettres, L'honneur de défendre un opprimé eût enflammé mon coeur si j'avais parlé pour un autre. Réduit au triste emploi de me défendre moi-même, j'ai dû me borner à raisonner ; m'échauffer eût été m'avilir. J'aurai donc trouvé grâce en ce point devant ceux qui s'imaginent qu'il est essentiel à la vérité d'être dite froidement ; opinion que pourtant j'ai peine à comprendre. Lorsqu'une vive persuasion nous anime, le moyen d'employer un langage glacé ? Quand Archimède tout transporté courait nu dans les rues de Syracuse, en avait-il moins trouvé la vérité parce qu'il se passionnait pour elle ? Tout au contraire, celui qui la sent ne peut s'abstenir de l'adorer ; celui qui demeure froid ne l'a pas vue.\nQuoi qu'il en soit, je prie les lecteurs de vouloir bien mettre à part mon beau style, et d'examiner seulement si je raisonne bien ou mal ; car enfin, de cela seul qu'un auteur s'exprime en bons termes, je ne vois pas comment il peut s'ensuivre que cet auteur ne sait ce qu'il dit.\n"