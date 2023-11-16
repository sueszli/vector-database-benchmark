import os
import re
from typing import TYPE_CHECKING
from jrnl.exception import JrnlException
from jrnl.messages import Message
from jrnl.messages import MsgStyle
from jrnl.messages import MsgText
from jrnl.output import print_msg
from jrnl.plugins.text_exporter import TextExporter
if TYPE_CHECKING:
    from jrnl.journals import Entry
    from jrnl.journals import Journal

class YAMLExporter(TextExporter):
    """This Exporter converts entries and journals into Markdown formatted text with
    YAML front matter."""
    names = ['yaml']
    extension = 'md'

    @classmethod
    def export_entry(cls, entry: 'Entry', to_multifile: bool=True) -> str:
        if False:
            print('Hello World!')
        'Returns a markdown representation of an entry, with YAML front matter.'
        if to_multifile is False:
            raise JrnlException(Message(MsgText.YamlMustBeDirectory, MsgStyle.ERROR))
        date_str = entry.date.strftime(entry.journal.config['timeformat'])
        body_wrapper = '\n' if entry.body else ''
        body = body_wrapper + entry.body
        tagsymbols = entry.journal.config['tagsymbols']
        multi_tag_regex = re.compile(f'(?u)^\\s*([{tagsymbols}][-+*#/\\w]+\\s*)+$')
        'Increase heading levels in body text'
        newbody = ''
        heading = '#'
        previous_line = ''
        warn_on_heading_level = False
        for line in body.splitlines(True):
            if re.match('^#+ ', line):
                'ATX style headings'
                newbody = newbody + previous_line + heading + line
                if re.match('^#######+ ', heading + line):
                    warn_on_heading_level = True
                line = ''
            elif re.match('^=+$', line.rstrip()) and (not re.match('^$', previous_line.strip())):
                'Setext style H1'
                newbody = newbody + heading + '# ' + previous_line
                line = ''
            elif re.match('^-+$', line.rstrip()) and (not re.match('^$', previous_line.strip())):
                'Setext style H2'
                newbody = newbody + heading + '## ' + previous_line
                line = ''
            elif multi_tag_regex.match(line):
                'Tag only lines'
                line = ''
            else:
                newbody = newbody + previous_line
            previous_line = line
        newbody = newbody + previous_line
        if previous_line not in ['\r', '\n', '\r\n', '\n\r']:
            newbody = newbody + os.linesep
        spacebody = '\t'
        for line in newbody.splitlines(True):
            spacebody = spacebody + '\t' + line
        if warn_on_heading_level is True:
            print_msg(Message(MsgText.HeadingsPastH6, MsgStyle.WARNING, {'date': date_str, 'title': entry.title}))
        dayone_attributes = ''
        if hasattr(entry, 'uuid'):
            dayone_attributes += 'uuid: ' + entry.uuid + '\n'
        if hasattr(entry, 'creator_device_agent') or hasattr(entry, 'creator_generation_date') or hasattr(entry, 'creator_host_name') or hasattr(entry, 'creator_os_agent') or hasattr(entry, 'creator_software_agent'):
            dayone_attributes += 'creator:\n'
            if hasattr(entry, 'creator_device_agent'):
                dayone_attributes += f'    device agent: {entry.creator_device_agent}\n'
            if hasattr(entry, 'creator_generation_date'):
                dayone_attributes += '    generation date: {}\n'.format(str(entry.creator_generation_date))
            if hasattr(entry, 'creator_host_name'):
                dayone_attributes += f'    host name: {entry.creator_host_name}\n'
            if hasattr(entry, 'creator_os_agent'):
                dayone_attributes += f'    os agent: {entry.creator_os_agent}\n'
            if hasattr(entry, 'creator_software_agent'):
                dayone_attributes += f'    software agent: {entry.creator_software_agent}\n'
        return '{start}\ntitle: {title}\ndate: {date}\nstarred: {starred}\ntags: {tags}\n{dayone}body: |{body}{end}'.format(start='---', date=date_str, title=entry.title, starred=entry.starred, tags=', '.join([tag[1:] for tag in entry.tags]), dayone=dayone_attributes, body=spacebody, end='...')

    @classmethod
    def export_journal(cls, journal: 'Journal'):
        if False:
            return 10
        'Returns an error, as YAML export requires a directory as a target.'
        raise JrnlException(Message(MsgText.YamlMustBeDirectory, MsgStyle.ERROR))