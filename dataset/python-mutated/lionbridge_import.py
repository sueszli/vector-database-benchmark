import argparse
import io
import os
import os.path
cura_files = {'cura', 'fdmprinter.def.json', 'fdmextruder.def.json'}
uranium_files = {'uranium'}

def lionbridge_import(source: str) -> None:
    if False:
        while True:
            i = 10
    "Imports translation files from Lionbridge.\n\n    Lionbridge has a bit of a weird export feature. It exports it to the same\n    file type as what we imported, so that's a .pot file. However this .pot file\n    only contains the translations, so the header is completely empty. We need\n    to merge those translations into our existing files so that the header is\n    preserved.\n    "
    print('Importing from:', source)
    print('Importing to Cura:', destination_cura())
    print('Importing to Uranium:', destination_uranium())
    for language in (directory for directory in os.listdir(source) if os.path.isdir(os.path.join(source, directory))):
        print('================ Processing language:', language, '================')
        directory = os.path.join(source, language)
        for file_pot in (file for file in os.listdir(directory) if file.endswith('.pot')):
            source_file = file_pot[:-4]
            if source_file in cura_files:
                destination_file = os.path.join(destination_cura(), language.replace('-', '_'), source_file + '.po')
                print('Merging', source_file, '(Cura) into', destination_file)
            elif source_file in uranium_files:
                destination_file = os.path.join(destination_uranium(), language.replace('-', '_'), source_file + '.po')
                print('Merging', source_file, '(Uranium) into', destination_file)
            else:
                raise Exception('Unknown file: ' + source_file + '... Is this Cura or Uranium?')
            with io.open(os.path.join(directory, file_pot), encoding='utf8') as f:
                source_str = f.read()
            with io.open(destination_file, encoding='utf8') as f:
                destination_str = f.read()
            result = merge(source_str, destination_str)
            with io.open(destination_file, 'w', encoding='utf8') as f:
                f.write(result)

def destination_cura() -> str:
    if False:
        return 10
    'Gets the destination path to copy the translations for Cura to.\n\n    :return: Destination path for Cura.\n    '
    return os.path.abspath(os.path.join(__file__, '..', '..', 'resources', 'i18n'))

def destination_uranium() -> str:
    if False:
        i = 10
        return i + 15
    'Gets the destination path to copy the translations for Uranium to.\n\n    :return: Destination path for Uranium.\n    '
    try:
        import UM
    except ImportError:
        relative_path = os.path.join(__file__, '..', '..', '..', 'Uranium', 'resources', 'i18n', 'uranium.pot')
        absolute_path = os.path.abspath(relative_path)
        if os.path.exists(absolute_path):
            absolute_path = os.path.abspath(os.path.join(absolute_path, '..'))
            print('Uranium is at:', absolute_path)
            return absolute_path
        else:
            raise Exception("Can't find Uranium. Please put UM on the PYTHONPATH or put the Uranium folder next to the Cura folder. Looked for: " + absolute_path)
    return os.path.abspath(os.path.join(UM.__file__, '..', '..', 'resources', 'i18n'))

def merge(source: str, destination: str) -> str:
    if False:
        while True:
            i = 10
    'Merges translations from the source file into the destination file if they\n\n    were missing in the destination file.\n    :param source: The contents of the source .po file.\n    :param destination: The contents of the destination .po file.\n    '
    result_lines = []
    last_destination = {'msgctxt': '""\n', 'msgid': '""\n', 'msgstr': '""\n', 'msgid_plural': '""\n'}
    current_state = 'none'
    for line in destination.split('\n'):
        if line.startswith('msgctxt "'):
            current_state = 'msgctxt'
            line = line[8:]
            last_destination[current_state] = ''
        elif line.startswith('msgid "'):
            current_state = 'msgid'
            line = line[6:]
            last_destination[current_state] = ''
        elif line.startswith('msgstr "'):
            current_state = 'msgstr'
            line = line[7:]
            last_destination[current_state] = ''
        elif line.startswith('msgid_plural "'):
            current_state = 'msgid_plural'
            line = line[13:]
            last_destination[current_state] = ''
        if line.startswith('"') and line.endswith('"'):
            last_destination[current_state] += line + '\n'
        else:
            if last_destination['msgstr'] == '""\n' and last_destination['msgid'] != '""\n':
                last_destination['msgstr'] = find_translation(source, last_destination['msgctxt'], last_destination['msgid'])
            if last_destination['msgctxt'] != '""\n' or last_destination['msgid'] != '""\n' or last_destination['msgid_plural'] != '""\n' or (last_destination['msgstr'] != '""\n'):
                if last_destination['msgctxt'] != '""\n':
                    result_lines.append('msgctxt {msgctxt}'.format(msgctxt=last_destination['msgctxt'][:-1]))
                result_lines.append('msgid {msgid}'.format(msgid=last_destination['msgid'][:-1]))
                if last_destination['msgid_plural'] != '""\n':
                    result_lines.append('msgid_plural {msgid_plural}'.format(msgid_plural=last_destination['msgid_plural'][:-1]))
                else:
                    result_lines.append('msgstr {msgstr}'.format(msgstr=last_destination['msgstr'][:-1]))
            last_destination = {'msgctxt': '""\n', 'msgid': '""\n', 'msgstr': '""\n', 'msgid_plural': '""\n'}
            result_lines.append(line)
    return '\n'.join(result_lines)

def find_translation(source: str, msgctxt: str, msgid: str) -> str:
    if False:
        while True:
            i = 10
    'Finds a translation in the source file.\n\n    :param source: The contents of the source .po file.\n    :param msgctxt: The ctxt of the translation to find.\n    :param msgid: The id of the translation to find.\n    '
    last_source = {'msgctxt': '""\n', 'msgid': '""\n', 'msgstr': '""\n', 'msgid_plural': '""\n'}
    current_state = 'none'
    for line in source.split('\n'):
        if line.startswith('msgctxt "'):
            current_state = 'msgctxt'
            line = line[8:]
            last_source[current_state] = ''
        elif line.startswith('msgid "'):
            current_state = 'msgid'
            line = line[6:]
            last_source[current_state] = ''
        elif line.startswith('msgstr "'):
            current_state = 'msgstr'
            line = line[7:]
            last_source[current_state] = ''
        elif line.startswith('msgid_plural "'):
            current_state = 'msgid_plural'
            line = line[13:]
            last_source[current_state] = ''
        if line.startswith('"') and line.endswith('"'):
            last_source[current_state] += line + '\n'
        else:
            source_ctxt = ''.join((line.strip()[1:-1] for line in last_source['msgctxt'].split('\n')))
            source_id = ''.join((line.strip()[1:-1] for line in last_source['msgid'].split('\n')))
            dest_ctxt = ''.join((line.strip()[1:-1] for line in msgctxt.split('\n')))
            dest_id = ''.join((line.strip()[1:-1] for line in msgid.split('\n')))
            if source_ctxt == dest_ctxt and source_id == dest_id:
                if last_source['msgstr'] == '""\n' and last_source['msgid_plural'] == '""\n':
                    print('!!! Empty translation for {' + dest_ctxt + '}', dest_id, '!!!')
                return last_source['msgstr']
            last_source = {'msgctxt': '""\n', 'msgid': '""\n', 'msgstr': '""\n', 'msgid_plural': '""\n'}
    print('!!! Missing translation for {' + msgctxt.strip() + '}', msgid.strip(), '!!!')
    return '""\n'
if __name__ == '__main__':
    print('Usage instructions:\n\n1. In Smartling, in the Cura project go to the "Files" tab.\n2. Select all four .pot files.\n3. In the expando above the file list, choose "Download Selected".\n4. In the pop-up, select:\n   - Current translations\n   - Select all languages\n   - Organize files: Folders for languages.\n5. Download that and extract the .zip archive somewhere.\n6. Start this script, with the location you extracted to as a parameter, e.g.:\n   python3 /path/to/lionbridge_import.py /home/username/Desktop/cura_translations\n')
    argparser = argparse.ArgumentParser(description='Import translation files from Lionbridge.')
    argparser.add_argument('source')
    args = argparser.parse_args()
    lionbridge_import(args.source)