"""
Kazakh Transliteration:
    Cyrillic Kazakh --> Latin Kazakh


"""
import argparse
import os
from re import M
import string
import sys
from stanza.models.common.utils import open_read_text, get_tqdm
tqdm = get_tqdm()
"\nThis dictionary isn't used in the code, just put this here in case you want to implement it more\nefficiently and in case the need to look up the unicode encodings for these letters might arise.\nSome letters are mapped to multiple latin letters, for these, I separated the unicde with a '%' delimiter\nbetween the two unicode characters.\n"
alph_map = {'А': 'A', 'а': 'a', 'Ә': 'Ä', 'ә': 'ä', 'Б': 'B', 'б': 'b', 'В': 'V', 'в': 'v', 'Г': 'G', 'г': 'g', 'Ғ': 'Ğ', 'ғ': 'ğ', 'Д': 'D', 'д': 'd', 'Е': 'E', 'е': 'e', 'Ё': 'İ%o', 'ё': 'i%o', 'Ж': 'J', 'ж': 'j', 'З': 'Z', 'з': 'z', 'И': 'İ', 'и': 'i', 'Й': 'İ', 'й': 'i', 'К': 'K', 'к': 'k', 'Қ': 'Q', 'қ': 'q', 'Л': 'L', 'л': 'l', 'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n', 'Ң': 'Ñ', 'ң': 'ñ', 'О': 'O', 'о': 'o', 'Ө': 'Ö', 'ө': 'ö', 'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r', 'С': 'S', 'с': 's', 'Т': 'T', 'т': 't', 'У': 'U', 'у': 'u', 'Ұ': 'Ū', 'ұ': 'ū', 'Ү': 'Ü', 'ү': 'ü', 'Ф': 'F', 'ф': 'f', 'Х': 'H', 'х': 'h', 'Һ': 'H', 'һ': 'h', 'Ц': 'C', 'ц': 'c', 'Ч': 'Ç', 'ч': 'ç', 'Ш': 'Ş', 'ш': 'ş', 'Щ': 'Ş%ç', 'щ': 'ş%ç', 'Ъ': '', 'ъ': '', 'Ы': 'Y', 'ы': 'y', 'І': 'İ', 'і': 'i', 'Ь': '', 'ь': '', 'Э': 'E', 'э': 'e', 'Ю': 'İ%u', 'ю': 'i%u', 'Я': 'İ%a', 'я': 'i%a'}
kazakh_alph = 'АаӘәБбВвГгҒғДдЕеЁёЖжЗзИиЙйКкҚқЛлМмНнҢңОоӨөПпРрСсТтУуҰұҮүФфХхҺһЦцЧчШшЩщЪъЫыІіЬьЭэЮюЯя'
latin_alph = 'AaÄäBbVvGgĞğDdEeİoioJjZzİiİiKkQqLlMmNnÑñOoÖöPpRrSsTtUuŪūÜüFfHhHhCcÇçŞşŞçşçYyİiEeİuiuİaia'
mult_mapping = 'ЁёЩщЮюЯя'
empty_mapping = 'ЪъЬь'
"\nϵ : Ukrainian letter for 'ё'\nə : Russian utf-8 encoding for Kazakh 'ә'\nó : 2016 Kazakh Latin adopted this instead of 'ö'\nã : 1 occurrence in the dataset -- mapped to 'a'\n\n"
russian_alph = 'ϵəóã'
russian_counterpart = 'ioäaöa'

def create_dic(source_alph, target_alph, mult_mapping, empty_mapping):
    if False:
        i = 10
        return i + 15
    res = {}
    idx = 0
    for i in range(len(source_alph)):
        l_s = source_alph[i]
        if l_s in mult_mapping:
            res[l_s] = target_alph[idx] + target_alph[idx + 1]
            idx += 1
        elif l_s in empty_mapping:
            res[l_s] = ''
            idx -= 1
        else:
            res[l_s] = target_alph[idx]
        idx += 1
    res['ϵ'] = 'io'
    res['ə'] = 'ä'
    res['ó'] = 'ö'
    res['ã'] = 'a'
    print(res)
    return res
supp_alph = 'IWwXx0123456789–«»—'

def transliterate(source):
    if False:
        print('Hello World!')
    output = ''
    tr_dict = create_dic(kazakh_alph, latin_alph, mult_mapping, empty_mapping)
    punc = string.punctuation
    white_spc = string.whitespace
    for c in source:
        if c in punc or c in white_spc:
            output += c
        elif c in latin_alph or c in supp_alph:
            output += c
        elif c in tr_dict:
            output += tr_dict[c]
        else:
            print(f'Transliteration Error: {c}')
    return output
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, nargs='+', help='Files to process')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to output results')
    args = parser.parse_args()
    tr_dict = create_dic(kazakh_alph, latin_alph, mult_mapping, empty_mapping)
    for filename in tqdm(args.input_file):
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            (directory, basename) = os.path.split(filename)
            output_name = os.path.join(args.output_dir, basename)
            if output_name.endswith('.xz'):
                output_name = output_name[:-3]
            output_name = output_name + '.trans'
        else:
            output_name = filename + '.trans'
        tqdm.write('Transliterating %s to %s' % (filename, output_name))
        with open_read_text(filename) as f_in:
            data = f_in.read()
        with open(output_name, 'w') as f_out:
            punc = string.punctuation
            white_spc = string.whitespace
            for c in tqdm(data, leave=False):
                if c in tr_dict:
                    f_out.write(tr_dict[c])
                else:
                    f_out.write(c)
    print('Process Completed Successfully!')