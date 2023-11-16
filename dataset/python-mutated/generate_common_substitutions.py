DEBUG = False
if __name__ == '__main__' and (not DEBUG):
    print('Suppressing output of generate_common_substitutions.py')

def debug(*args, **kwargs):
    if False:
        while True:
            i = 10
    if DEBUG:
        print(*args, **kwargs)
import os
import sys
sys.path.append(os.path.join(os.path.split(__file__)[0], '../generators'))
from device_infos import brick_infos, bricklet_infos
lang = 'en'
ipcon_common = {'en': '\n>>>intro\nThis is the description of the |ref_api_bindings| for the IP Connection.\nThe IP Connection manages the communication between the API bindings and the\n:ref:`Brick Daemon <brickd>` or a :ref:`WIFI <wifi_extension>`/:ref:`Ethernet\n<ethernet_extension>` Extension. Before :ref:`Bricks <primer_bricks>` and\n:ref:`Bricklets <primer_bricklets>` can be controlled using their API an\nIP Connection has to be created and its TCP/IP connection has to be established.\n\nAn |ref_install_guide| for the |bindings_name| API bindings is part of their\ngeneral description.\n<<<intro\n', 'de': '\n>>>intro\nDies ist die Beschreibung der |ref_api_bindings| für die IP Connection.\nDie IP Connection kümmert sich um die Kommunikation zwischen einem\n:ref:`Brick Daemon <brickd>` oder einer\n:ref:`WIFI <wifi_extension>`/:ref:`Ethernet <ethernet_extension>` Extension.\nBevor :ref:`Bricks <primer_bricks>` und :ref:`Bricklets <primer_bricklets>` über\nderen API angesprochen werden können muss eine IP Connection erzeugt und\ndie TCP/IP Verbindung hergestellt werden.\n\nEine |ref_install_guide| für die |bindings_name| API Bindings ist Teil deren\nallgemeine Beschreibung.\n<<<intro\n'}
brick_test_intro = {'en': '.. |test_intro| replace::\n To test a {0} you need to have :ref:`Brick Daemon <brickd>` and\n :ref:`Brick Viewer <brickv>` installed. Brick Daemon acts as a proxy between\n the USB interface of the Bricks and the API bindings. Brick Viewer connects\n to Brick Daemon. It helps to figure out basic information about the connected\n Bricks and Bricklets and allows to test them.\n', 'de': '.. |test_intro| replace::\n Um einen {0} testen zu können, müssen zuerst :ref:`Brick Daemon\n <brickd>` und :ref:`Brick Viewer <brickv>` installiert werden. Brick Daemon\n arbeitet als Proxy zwischen der USB Schnittstelle der Bricks und den API\n Bindings. Brick Viewer kann sich mit Brick Daemon verbinden, gibt\n Informationen über die angeschlossenen Bricks und Bricklets aus und ermöglicht\n es diese zu testen.\n'}
brick_test_tab = {'en': '.. |test_tab| replace::\n Now connect the Brick to the PC over USB, you should see a new tab named\n "{0}" in the Brick Viewer after a moment. Select this tab.\n', 'de': '.. |test_tab| replace::\n Wenn der Brick per USB an den PC angeschlossen wird sollte einen Moment später\n im Brick Viewer ein neuer Tab namens "{0}" auftauchen. Wähle diesen Tab\n aus.\n'}
brick_test_pi_ref = {'en': '.. |test_pi_ref| replace::\n After this test you can go on with writing your own application.\n See the :ref:`Programming Interface <{1}_programming_interface>`\n section for the API of the {0} and examples in different programming\n languages.\n', 'de': '.. |test_pi_ref| replace::\n Nun kann ein eigenes Programm geschrieben werden. Der Abschnitt\n :ref:`Programmierschnittstelle <{1}_programming_interface>` listet die\n API des {0} und Beispiele in verschiedenen Programmiersprachen auf.\n'}
bricklet_case_steps = {'en': '\n>>>bricklet_case_steps\nThe assembly is easiest if you follow the following steps:\n\n* Screw spacers to the Bricklet,\n* screw bottom plate to bottom spacers,\n* build up side plates,\n* plug side plates into bottom plate and\n* screw top plate to top spacers.\n\nBelow you can see an exploded assembly drawing of the {0} case:\n<<<bricklet_case_steps\n', 'de': '\n>>>bricklet_case_steps\nDer Aufbau ist am einfachsten wenn die folgenden Schritte befolgt werden:\n\n* Schraube Abstandshalter an das Bricklet,\n* schraube Unterteil an untere Abstandshalter,\n* baue Seitenteile auf,\n* stecke zusammengebaute Seitenteile in Unterteil und\n* schraube Oberteil auf obere Abstandshalter.\n\nIm Folgenden befindet sich eine Explosionszeichnung des {0} Gehäuses:\n<<<bricklet_case_steps\n'}
bricklet_case_hint = {'en': '.. |bricklet_case_hint| replace::\n Hint: There is a protective film on both sides of the plates,\n you have to remove it before assembly.\n', 'de': '.. |bricklet_case_hint| replace::\n Hinweis: Auf beiden Seiten der Platten ist eine Schutzfolie, \n diese muss vor dem Zusammenbau entfernt werden.\n'}
bricklet_test_intro = {'en': '.. |test_intro| replace::\n To test a {0} you need to have :ref:`Brick Daemon <brickd>` and\n :ref:`Brick Viewer <brickv>` installed. Brick Daemon acts as a proxy between\n the USB interface of the Bricks and the API bindings. Brick Viewer connects\n to Brick Daemon. It helps to figure out basic information about the connected\n Bricks and Bricklets and allows to test them.\n', 'de': '.. |test_intro| replace::\n Um ein {0} testen zu können, müssen zuerst :ref:`Brick Daemon\n <brickd>` und :ref:`Brick Viewer <brickv>` installiert werden. Brick Daemon\n arbeitet als Proxy zwischen der USB Schnittstelle der Bricks und den API\n Bindings. Brick Viewer kann sich mit Brick Daemon verbinden, gibt\n Informationen über die angeschlossenen Bricks und Bricklets aus und ermöglicht\n es diese zu testen.\n'}
bricklet_test_connect = {'en': '.. |test_connect| replace::\n Connect the {0} to a :ref:`Brick <primer_bricks>`\n with a Bricklet Cable\n', 'de': '.. |test_connect| replace::\n Als nächstes muss das {0} mittels eines Bricklet Kabels mit\n einem :ref:`Brick <primer_bricks>` verbunden werden\n'}
bricklet_test_tab = {'en': '.. |test_tab| replace::\n If you connect the Brick to the PC over USB, you should see a new tab named\n "{0}" in the Brick Viewer after a moment. Select this tab.\n', 'de': '.. |test_tab| replace::\n Wenn der Brick per USB an den PC angeschlossen wird sollte einen Moment später\n im Brick Viewer ein neuer Tab namens "{0}" auftauchen.\n Wähle diesen Tab aus.\n'}
bricklet_test_pi_ref = {'en': '.. |test_pi_ref| replace::\n After this test you can go on with writing your own application.\n See the :ref:`Programming Interface <{1}_programming_interface>`\n section for the API of the {0} and examples in different programming\n languages.\n', 'de': '.. |test_pi_ref| replace::\n Nun kann ein eigenes Programm geschrieben werden. Der Abschnitt\n :ref:`Programmierschnittstelle <{1}_programming_interface>` listet\n die API des {0} und Beispiele in verschiedenen\n Programmiersprachen auf.\n'}

def make_ipcon_substitutions():
    if False:
        for i in range(10):
            print('nop')
    substitutions = ''
    substitutions += ipcon_common[lang]
    return substitutions

def make_brick_substitutions(brick_info):
    if False:
        while True:
            i = 10
    substitutions = ''
    substitutions += brick_test_intro[lang].format(brick_info.long_display_name) + '\n'
    substitutions += brick_test_tab[lang].format(brick_info.long_display_name) + '\n'
    substitutions += brick_test_pi_ref[lang].format(brick_info.long_display_name, brick_info.ref_name)
    return substitutions

def make_bricklet_substitutions(bricklet_info):
    if False:
        return 10
    substitutions = ''
    substitutions += '>>>substitutions\n'
    substitutions += bricklet_test_intro[lang].format(bricklet_info.long_display_name) + '\n'
    substitutions += bricklet_test_connect[lang].format(bricklet_info.long_display_name) + '\n'
    substitutions += bricklet_test_tab[lang].format(bricklet_info.long_display_name) + '\n'
    substitutions += bricklet_test_pi_ref[lang].format(bricklet_info.long_display_name, bricklet_info.ref_name) + '\n'
    substitutions += bricklet_case_hint[lang] + '\n'
    substitutions += '<<<substitutions\n'
    substitutions += bricklet_case_steps[lang].format(bricklet_info.long_display_name) + '\n'
    return substitutions

def write_if_changed(path, content):
    if False:
        while True:
            i = 10
    if os.path.exists(path):
        with open(path, 'r') as f:
            existing = f.read()
        if existing == content:
            return
    with open(path, 'w') as f:
        f.write(content)

def generate(path):
    if False:
        print('Hello World!')
    global lang
    if path.endswith('/en'):
        lang = 'en'
    elif path.endswith('/de'):
        lang = 'de'
    else:
        debug('Wrong working directory')
        sys.exit(1)
    write_if_changed(os.path.join(path, 'source', 'Software', 'IPConnection_Common.substitutions'), make_ipcon_substitutions())
    for brick_info in brick_infos:
        debug('Generating {0}.substitutions (Hardware)'.format(brick_info.hardware_doc_name))
        write_if_changed(os.path.join(path, 'source', 'Hardware', 'Bricks', brick_info.hardware_doc_name + '.substitutions'), make_brick_substitutions(brick_info))
    for bricklet_info in bricklet_infos:
        debug('Generating {0}.substitutions (Hardware)'.format(bricklet_info.hardware_doc_name))
        write_if_changed(os.path.join(path, 'source', 'Hardware', 'Bricklets', bricklet_info.hardware_doc_name + '.substitutions'), make_bricklet_substitutions(bricklet_info))
if __name__ == '__main__':
    generate(os.getcwd())