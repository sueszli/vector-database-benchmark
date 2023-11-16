import xml
import yaml

def subprocess_shell_cmd():
    if False:
        print('Hello World!')
    subprocess.Popen('/bin/ls *', shell=True)

def yaml_load():
    if False:
        while True:
            i = 10
    temp_str = yaml.dump({'a': '1', 'b': '2'})
    y = yaml.load(temp_str)

def xml_sax_make_parser():
    if False:
        while True:
            i = 10
    xml.sax.make_parser()