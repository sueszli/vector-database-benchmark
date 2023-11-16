import traceback
import sys
MESSENGERS_LIST = list()
_indent = ''
flowgraph_error = None
flowgraph_error_file = None

def register_messenger(messenger):
    if False:
        return 10
    '\n    Append the given messenger to the list of messengers.\n\n    Args:\n        messenger: a method that takes a string\n    '
    MESSENGERS_LIST.append(messenger)

def set_indent(level=0):
    if False:
        print('Hello World!')
    global _indent
    _indent = '    ' * level

def send(message):
    if False:
        while True:
            i = 10
    '\n    Give the message to each of the messengers.\n\n    Args:\n        message: a message string\n    '
    for messenger in MESSENGERS_LIST:
        messenger(_indent + message)
register_messenger(sys.stdout.write)

def send_init(platform):
    if False:
        while True:
            i = 10
    msg = '<<< Welcome to {config.name} {config.version} >>>\n\nBlock paths:\n\t{paths}\n'
    send(msg.format(config=platform.config, paths='\n\t'.join(platform.config.block_paths)))

def send_xml_errors_if_any(xml_failures):
    if False:
        while True:
            i = 10
    if xml_failures:
        send('\nXML parser: Found {0} erroneous XML file{1} while loading the block tree (see "Help/Parser errors" for details)\n'.format(len(xml_failures), 's' if len(xml_failures) > 1 else ''))

def send_start_load(file_path):
    if False:
        i = 10
        return i + 15
    send('\nLoading: "%s"\n' % file_path)

def send_error_msg_load(error):
    if False:
        for i in range(10):
            print('nop')
    send('>>> Error: %s\n' % error)

def send_error_load(error):
    if False:
        for i in range(10):
            print('nop')
    send_error_msg_load(error)
    traceback.print_exc()

def send_end_load():
    if False:
        print('Hello World!')
    send('>>> Done\n')

def send_fail_load(error):
    if False:
        i = 10
        return i + 15
    send('Error: %s\n>>> Failure\n' % error)
    traceback.print_exc()

def send_start_gen(file_path):
    if False:
        return 10
    send('\nGenerating: "%s"\n' % file_path)

def send_auto_gen(file_path):
    if False:
        return 10
    send('>>> Generating: "%s"\n' % file_path)

def send_fail_gen(error):
    if False:
        return 10
    send('Generate Error: %s\n>>> Failure\n' % error)
    traceback.print_exc()

def send_start_exec(file_path):
    if False:
        while True:
            i = 10
    send('\nExecuting: %s\n' % file_path)

def send_verbose_exec(verbose):
    if False:
        return 10
    send(verbose)

def send_end_exec(code=0):
    if False:
        for i in range(10):
            print('nop')
    send('\n>>> Done%s\n' % (' (return code %s)' % code if code else ''))

def send_fail_save(file_path):
    if False:
        while True:
            i = 10
    send('>>> Error: Cannot save: %s\n' % file_path)

def send_fail_connection(msg=''):
    if False:
        for i in range(10):
            print('nop')
    send('>>> Error: Cannot create connection.\n' + ('\t{}\n'.format(msg) if msg else ''))

def send_fail_load_preferences(prefs_file_path):
    if False:
        return 10
    send('>>> Error: Cannot load preferences file: "%s"\n' % prefs_file_path)

def send_fail_save_preferences(prefs_file_path):
    if False:
        for i in range(10):
            print('nop')
    send('>>> Error: Cannot save preferences file: "%s"\n' % prefs_file_path)

def send_warning(warning):
    if False:
        print('Hello World!')
    send('>>> Warning: %s\n' % warning)

def send_flowgraph_error_report(flowgraph):
    if False:
        i = 10
        return i + 15
    ' verbose error report for flowgraphs '
    error_list = flowgraph.get_error_messages()
    if not error_list:
        return
    send('*' * 50 + '\n')
    summary_msg = '{} errors from flowgraph:\n'.format(len(error_list))
    send(summary_msg)
    for err in error_list:
        send(err)
    send('\n' + '*' * 50 + '\n')