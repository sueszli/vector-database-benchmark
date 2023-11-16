import argparse
from typing import List
'\n    Takes in one of the po files in resources/i18n/[LANG_CODE]/cura.po and updates it with translations from a \n    new po file without changing the translation ordering. \n    This script should be used when we get a po file that has updated translations but is no longer correctly ordered \n    so the merge becomes messy.\n    \n    If you are importing files from lionbridge/smartling use lionbridge_import.py.\n    \n    Note: This does NOT include new strings, it only UPDATES existing strings   \n'

class Msg:

    def __init__(self, msgctxt: str='', msgid: str='', msgstr: str='') -> None:
        if False:
            while True:
                i = 10
        self.msgctxt = msgctxt
        self.msgid = msgid
        self.msgstr = msgstr

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.msgctxt + self.msgid + self.msgstr

def parsePOFile(filename: str) -> List[Msg]:
    if False:
        return 10
    messages = []
    with open(filename) as f:
        iterator = iter(f.readlines())
        for line in iterator:
            if line.startswith('msgctxt'):
                msg = Msg()
                msg.msgctxt = line
                while True:
                    line = next(iterator)
                    if line.startswith('msgid'):
                        msg.msgid = line
                        break
                while True:
                    line = next(iterator)
                    if line == '\n':
                        break
                    if line.startswith('msgstr'):
                        msg.msgstr = line
                    else:
                        msg.msgstr += line
                messages.append(msg)
        return messages

def getDifferentMessages(messages_original: List[Msg], messages_new: List[Msg]) -> List[Msg]:
    if False:
        for i in range(10):
            print('nop')
    different_messages = []
    for m_new in messages_new:
        for m_original in messages_original:
            if m_new.msgstr != m_original.msgstr and m_new.msgid == m_original.msgid and (m_new.msgctxt == m_original.msgctxt) and (m_new.msgid != 'msgid ""\n'):
                different_messages.append(m_new)
    return different_messages

def updatePOFile(input_filename: str, output_filename: str, messages: List[Msg]) -> None:
    if False:
        while True:
            i = 10
    with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
        iterator = iter(input_file.readlines())
        for line in iterator:
            output_file.write(line)
            if line.startswith('msgctxt'):
                msgctxt = line
                msgid = next(iterator)
                output_file.write(msgid)
                message = list(filter(lambda m: m.msgctxt == msgctxt and m.msgid == msgid, messages))
                if message and message[0]:
                    output_file.write(message[0].msgstr)
                    while True:
                        line = next(iterator)
                        if line == '\n':
                            output_file.write(line)
                            break
if __name__ == '__main__':
    print('********************************************************************************************************************')
    print("This creates a new file 'updated.po' that is a copy of original_file with any changed translations from updated_file")
    print('This does not change the order of translations')
    print('This does not include new translations, only existing changed translations')
    print('Do not use this to import lionbridge/smarting translations')
    print('********************************************************************************************************************')
    parser = argparse.ArgumentParser(description='Update po file with translations from new po file. This ')
    parser.add_argument('original_file', type=str, help='Input .po file inside resources/i18n/[LANG]/')
    parser.add_argument('updated_file', type=str, help='Input .po file with updated translations added')
    args = parser.parse_args()
    messages_updated = parsePOFile(args.updated_file)
    messages_original = parsePOFile(args.original_file)
    different_messages = getDifferentMessages(messages_original, messages_updated)
    updatePOFile(args.original_file, 'updated.po', different_messages)