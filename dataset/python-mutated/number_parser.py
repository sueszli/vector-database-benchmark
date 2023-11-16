import os
import re
import sys
import config
import typing
G_spat = re.compile('^\\w+\\.(cc|com|net|me|club|jp|tv|xyz|biz|wiki|info|tw|us|de)@|^22-sht\\.me|^(fhd|hd|sd|1080p|720p|4K)(-|_)|(-|_)(fhd|hd|sd|1080p|720p|4K|x264|x265|uncensored|hack|leak)', re.IGNORECASE)

def get_number(debug: bool, file_path: str) -> str:
    if False:
        return 10
    '\n    从文件路径中提取番号 from number_parser import get_number\n    >>> get_number(False, "/Users/Guest/AV_Data_Capture/snis-829.mp4")\n    \'snis-829\'\n    >>> get_number(False, "/Users/Guest/AV_Data_Capture/snis-829-C.mp4")\n    \'snis-829\'\n    >>> get_number(False, "/Users/Guest/AV_Data_Capture/[脸肿字幕组][PoRO]牝教師4～穢された教壇～ 「生意気ドジっ娘女教師・美結～高飛車ハメ堕ち2濁金」[720p][x264_aac].mp4")\n    \'牝教師4～穢された教壇～ 「生意気ドジっ娘女教師・美結～高飛車ハメ堕ち2濁金」\'\n    >>> get_number(False, "C:¥Users¥Guest¥snis-829.mp4")\n    \'snis-829\'\n    >>> get_number(False, "C:¥Users¥Guest¥snis-829-C.mp4")\n    \'snis-829\'\n    >>> get_number(False, "./snis-829.mp4")\n    \'snis-829\'\n    >>> get_number(False, "./snis-829-C.mp4")\n    \'snis-829\'\n    >>> get_number(False, ".¥snis-829.mp4")\n    \'snis-829\'\n    >>> get_number(False, ".¥snis-829-C.mp4")\n    \'snis-829\'\n    >>> get_number(False, "snis-829.mp4")\n    \'snis-829\'\n    >>> get_number(False, "snis-829-C.mp4")\n    \'snis-829\'\n    '
    filepath = os.path.basename(file_path)
    try:
        if config.getInstance().number_regexs().split().__len__() > 0:
            for regex in config.getInstance().number_regexs().split():
                try:
                    if re.search(regex, filepath):
                        return re.search(regex, filepath).group()
                except Exception as e:
                    print(f'[-]custom regex exception: {e} [{regex}]')
        file_number = get_number_by_dict(filepath)
        if file_number:
            return file_number
        elif '字幕组' in filepath or 'SUB' in filepath.upper() or re.match('[\\u30a0-\\u30ff]+', filepath):
            filepath = G_spat.sub('', filepath)
            filepath = re.sub('\\[.*?\\]', '', filepath)
            filepath = filepath.replace('.chs', '').replace('.cht', '')
            file_number = str(re.findall('(.+?)\\.', filepath)).strip(" [']")
            return file_number
        elif '-' in filepath or '_' in filepath:
            filepath = G_spat.sub('', filepath)
            filename = str(re.sub('\\[\\d{4}-\\d{1,2}-\\d{1,2}\\] - ', '', filepath))
            lower_check = filename.lower()
            if 'fc2' in lower_check:
                filename = lower_check.replace('--', '-').replace('_', '-').upper()
            filename = re.sub('[-_]cd\\d{1,2}', '', filename, flags=re.IGNORECASE)
            if not re.search('-|_', filename):
                return str(re.search('\\w+', filename[:filename.find('.')], re.A).group())
            file_number = os.path.splitext(filename)
            filename = re.search('[\\w\\-_]+', filename, re.A)
            if filename:
                file_number = str(filename.group())
            else:
                file_number = file_number[0]
            new_file_number = file_number
            if re.search('-c', file_number, flags=re.IGNORECASE):
                new_file_number = re.sub('(-|_)c$', '', file_number, flags=re.IGNORECASE)
            elif re.search('-u$', file_number, flags=re.IGNORECASE):
                new_file_number = re.sub('(-|_)u$', '', file_number, flags=re.IGNORECASE)
            elif re.search('-uc$', file_number, flags=re.IGNORECASE):
                new_file_number = re.sub('(-|_)uc$', '', file_number, flags=re.IGNORECASE)
            elif re.search('\\d+ch$', file_number, flags=re.I):
                new_file_number = file_number[:-2]
            return new_file_number.upper()
        else:
            oumei = re.search('[a-zA-Z]+\\.\\d{2}\\.\\d{2}\\.\\d{2}', filepath)
            if oumei:
                return oumei.group()
            try:
                return str(re.findall('(.+?)\\.', str(re.search('([^<>/\\\\|:""\\*\\?]+)\\.\\w+$', filepath).group()))).strip("['']").replace('_', '-')
            except:
                return str(re.search('(.+?)\\.', filepath)[0])
    except Exception as e:
        if debug:
            print(f'[-]Number Parser exception: {e} [{file_path}]')
        return None
G_TAKE_NUM_RULES = {'tokyo.*hot': lambda x: str(re.search('(cz|gedo|k|n|red-|se)\\d{2,4}', x, re.I).group()), 'carib': lambda x: str(re.search('\\d{6}(-|_)\\d{3}', x, re.I).group()).replace('_', '-'), '1pon|mura|paco': lambda x: str(re.search('\\d{6}(-|_)\\d{3}', x, re.I).group()).replace('-', '_'), '10mu': lambda x: str(re.search('\\d{6}(-|_)\\d{2}', x, re.I).group()).replace('-', '_'), 'x-art': lambda x: str(re.search('x-art\\.\\d{2}\\.\\d{2}\\.\\d{2}', x, re.I).group()), 'xxx-av': lambda x: ''.join(['xxx-av-', re.findall('xxx-av[^\\d]*(\\d{3,5})[^\\d]*', x, re.I)[0]]), 'heydouga': lambda x: 'heydouga-' + '-'.join(re.findall('(\\d{4})[\\-_](\\d{3,4})[^\\d]*', x, re.I)[0]), 'heyzo': lambda x: 'HEYZO-' + re.findall('heyzo[^\\d]*(\\d{4})', x, re.I)[0], 'mdbk': lambda x: str(re.search('mdbk(-|_)(\\d{4})', x, re.I).group()), 'mdtm': lambda x: str(re.search('mdtm(-|_)(\\d{4})', x, re.I).group()), 'caribpr': lambda x: str(re.search('\\d{6}(-|_)\\d{3}', x, re.I).group()).replace('_', '-')}

def get_number_by_dict(filename: str) -> typing.Optional[str]:
    if False:
        return 10
    try:
        for (k, v) in G_TAKE_NUM_RULES.items():
            if re.search(k, filename, re.I):
                return v(filename)
    except:
        pass
    return None

class Cache_uncensored_conf:
    prefix = None

    def is_empty(self):
        if False:
            while True:
                i = 10
        return bool(self.prefix is None)

    def set(self, v: list):
        if False:
            return 10
        if not v or not len(v) or (not len(v[0])):
            raise ValueError('input prefix list empty or None')
        s = v[0]
        if len(v) > 1:
            for i in v[1:]:
                s += f'|{i}.+'
        self.prefix = re.compile(s, re.I)

    def check(self, number):
        if False:
            print('Hello World!')
        if self.prefix is None:
            raise ValueError('No init re compile')
        return self.prefix.match(number)
G_cache_uncensored_conf = Cache_uncensored_conf()

def is_uncensored(number) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if re.match('[\\d-]{4,}|\\d{6}_\\d{2,3}|(cz|gedo|k|n|red-|se)\\d{2,4}|heyzo.+|xxx-av-.+|heydouga-.+|x-art\\.\\d{2}\\.\\d{2}\\.\\d{2}', number, re.I):
        return True
    if G_cache_uncensored_conf.is_empty():
        G_cache_uncensored_conf.set(config.getInstance().get_uncensored().split(','))
    return bool(G_cache_uncensored_conf.check(number))
if __name__ == '__main__':
    test_use_cases = ('MEYD-594-C.mp4', 'SSIS-001_C.mp4', 'SSIS100-C.mp4', 'SSIS101_C.mp4', 'ssni984.mp4', 'ssni666.mp4', 'SDDE-625_uncensored_C.mp4', 'SDDE-625_uncensored_leak_C.mp4', 'SDDE-625_uncensored_leak_C_cd1.mp4', 'Tokyo Hot n9001 FHD.mp4', 'TokyoHot-n1287-HD SP2006 .mp4', 'caribean-020317_001.nfo', '257138_3xplanet_1Pondo_080521_001.mp4', 'ADV-R0624-CD3.wmv', 'XXX-AV   22061-CD5.iso', 'xxx-av 20589.mp4', 'Muramura-102114_145-HD.wmv', 'heydouga-4102-023-CD2.iso', 'HeyDOuGa4236-1048 Ai Qiu - .mp4', 'pacopacomama-093021_539-FHD.mkv', 'sbw99.cc@heyzo_hd_2636_full.mp4', 'hhd800.com@STARS-566-HD.mp4', 'jav20s8.com@GIGL-677_4K.mp4', 'sbw99.cc@iesp-653-4K.mp4', '4K-ABP-358_C.mkv', 'n1012-CD1.wmv', '[]n1012-CD2.wmv', 'rctd-460ch.mp4', 'rctd-461CH-CD2.mp4', 'rctd-461-Cd3-C.mp4', 'rctd-461-C-cD4.mp4', 'MD-123.ts', 'MDSR-0001-ep2.ts', 'MKY-NS-001.mp4')

    def evprint(evstr):
        if False:
            while True:
                i = 10
        code = compile(evstr, '<string>', 'eval')
        print("{1:>20} # '{0}'".format(evstr[18:-2], eval(code)))
    for t in test_use_cases:
        evprint(f'get_number(True, "{t}")')
    if len(sys.argv) <= 1 or not re.search('^[A-Z]:?', sys.argv[1], re.IGNORECASE):
        sys.exit(0)
    import subprocess
    ES_search_path = 'ALL disks'
    if sys.argv[1] == 'ALL':
        if sys.platform == 'win32':
            ES_prog_path = 'es.exe'
            ES_cmdline = f'{ES_prog_path} -name size:gigantic ext:mp4;avi;rmvb;wmv;mov;mkv;flv;ts;webm;iso;mpg;m4v'
            out_bytes = subprocess.check_output(ES_cmdline.split(' '))
            out_text = out_bytes.decode('gb18030')
            out_list = out_text.splitlines()
        elif sys.platform in ('linux', 'darwin'):
            ES_prog_path = 'locate' if sys.platform == 'linux' else 'glocate'
            ES_cmdline = "{} -b -i --regex '\\.mp4$|\\.avi$|\\.rmvb$|\\.wmv$|\\.mov$|\\.mkv$|\\.webm$|\\.iso$|\\.mpg$|\\.m4v$'".format(ES_prog_path)
            out_bytes = subprocess.check_output(ES_cmdline.split(' '))
            out_text = out_bytes.decode('utf-8')
            out_list = [os.path.basename(line) for line in out_text.splitlines()]
        else:
            print('[-]Unsupported platform! Please run on OS Windows/Linux/MacOSX. Exit.')
            sys.exit(1)
    else:
        if sys.platform != 'win32':
            print('[!]Usage: python3 ./number_parser.py ALL')
            sys.exit(0)
        ES_prog_path = 'es.exe'
        if os.path.isdir(sys.argv[1]):
            ES_search_path = sys.argv[1]
        else:
            ES_search_path = sys.argv[1][0] + ':/'
            if not os.path.isdir(ES_search_path):
                ES_search_path = 'C:/'
            ES_search_path = os.path.normcase(ES_search_path)
        ES_cmdline = f'{ES_prog_path} -path {ES_search_path} -name size:gigantic ext:mp4;avi;rmvb;wmv;mov;mkv;webm;iso;mpg;m4v'
        out_bytes = subprocess.check_output(ES_cmdline.split(' '))
        out_text = out_bytes.decode('gb18030')
        out_list = out_text.splitlines()
    print(f'\n[!]{ES_prog_path} is searching {ES_search_path} for movies as number parser test cases...')
    print(f'[+]Find {len(out_list)} Movies.')
    for filename in out_list:
        try:
            n = get_number(True, filename)
            if n:
                print('  [{0}] {2}# {1}'.format(n, filename, '#无码' if is_uncensored(n) else ''))
            else:
                print(f'[-]Number return None. # {filename}')
        except Exception as e:
            print(f'[-]Number Parser exception: {e} [{filename}]')
    sys.exit(0)