import xml
import yaml

def subprocess_shell_cmd():
    if False:
        i = 10
        return i + 15
    subprocess.Popen('/bin/ls *', shell=True)
    subprocess.Popen('/bin/ls *', shell=True)

def yaml_load():
    if False:
        while True:
            i = 10
    temp_str = yaml.dump({'a': '1', 'b': '2'})
    y = yaml.load(temp_str)
    y = yaml.load(temp_str)

def xml_sax_make_parser():
    if False:
        return 10
    xml.sax.make_parser()
    xml.sax.make_parser()