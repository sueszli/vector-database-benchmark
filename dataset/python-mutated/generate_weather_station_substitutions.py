DEBUG = False
if __name__ == '__main__' and (not DEBUG):
    print('Suppressing output of generate_weather_station_substitutions.py')

def debug(*args, **kwargs):
    if False:
        while True:
            i = 10
    if DEBUG:
        print(*args, **kwargs)
import os
import sys
import generate_tables
bindings_infos = generate_tables.bindings_infos
lang = 'en'
examples = {'c': 'C', 'csharp': 'C#', 'delphi': 'Delphi', 'java': 'Java', 'php': 'PHP', 'python': 'Python', 'ruby': 'Ruby', 'vbnet': 'Visual Basic .NET'}
binding_name = {'en': ':ref:`{0} <api_bindings_{1}>`', 'de': ':ref:`{0} <api_bindings_{1}>`'}
binding_names = {'en': '\n.. |bindings| replace:: {0}\n', 'de': '\n.. |bindings| replace:: {0}\n'}
common_intro = {'en': '\n>>>intro\nFor this project we are assuming, that you have a {0} development environment\nset up and that you have a rudimentary understanding of the {0} language.\n\nIf you are totally new to {0} itself you should start `here <{2}>`__. If you are\nnew to the Tinkerforge API, you should start :ref:`here <api_bindings_{1}>`.\n<<<intro\n', 'de': '\n>>>intro\nFür diese Projekt setzen wir voraus, dass eine {0} Entwicklungsumgebung\neingerichtet ist und ein grundsätzliches Verständnis der {0} Programmiersprache\nvorhanden ist.\n\nFalls dies nicht der Fall ist sollte `hier <{2}>`__ begonnen werden. Informationen\nüber die Tinkerforge API sind dann :ref:`hier <api_bindings_{1}>` zu finden.\n<<<intro\n'}
write_to_lcd_example_line = {'en': ':ref:`{0} <starter_kit_weather_station_{1}_to_lcd>`', 'de': ':ref:`{0} <starter_kit_weather_station_{1}_to_lcd>`'}
write_to_lcd_examples = {'en': '\n.. |write_to_lcd_examples| replace:: {0}\n', 'de': '\n.. |write_to_lcd_examples| replace:: {0}\n'}
write_to_lcd_examples_toctree_line = {'en': '   Using {0} <{1}ToLCD>', 'de': '   Mit {0} <{1}ToLCD>'}
write_to_lcd_examples_toctree = {'en': '.. toctree::\n   :hidden:\n\n{0}\n', 'de': '.. toctree::\n   :hidden:\n\n{0}\n'}
write_to_lcd_example_downloads = {'en': '\n.. |write_to_lcd_examples_download| replace:: {0}\n', 'de': '\n.. |write_to_lcd_examples_download| replace:: {0}\n'}
write_to_lcd_example_download_line = {'en': '`{0} <https://github.com/Tinkerforge/weather-station/tree/master/write_to_lcd/{1}>`__', 'de': '`{0} <https://github.com/Tinkerforge/weather-station/tree/master/write_to_lcd/{1}>`__'}
write_to_lcd_example_downloads = {'en': '\n.. |write_to_lcd_examples_download| replace:: {0}\n', 'de': '\n.. |write_to_lcd_examples_download| replace:: {0}\n'}
write_to_lcd_goals = {'en': "\n>>>goals\nWe are setting the following goals for this project:\n\n* Temperature, ambient light, humidity and air pressure should be shown on\n  the LCD 20x4 Bricklet,\n* the measured values should be updated automatically when they change and\n* the measured values should be formated to be easily readable.\n\nSince this project will likely run 24/7, we will also make sure\nthat the application is as robust towards external influences as possible.\nThe application should still work when\n\n* Bricklets are exchanged (i.e. we don't rely on UIDs),\n* Brick Daemon isn't running or is restarted,\n* WIFI Extension is out of range or\n* Weather Station is restarted (power loss or accidental USB removal).\n\nIn the following we will show step-by-step how this can be achieved.\n<<<goals\n", 'de': '\n>>>goals\nWir setzen uns folgende Ziele für dieses Projekt:\n\n* Temperatur, Helligkeit, Luftfeuchte und Luftdruck sollen auf dem LCD 20x4\n  Bricklet angezeigt werden,\n* die gemessenen Werte sollen automatisch aktualisiert werden sobald sie sich\n  verändern und\n* die gemessenen Werte sollen in einem verständlichen Format angezeigt werden.\n\nDa dieses Projekt wahrscheinlich 24/7 laufen wird, wollen wir sicherstellen,\ndass das Programm möglichst robust gegen externe Einflüsse ist. Das Programm\nsollte weiterhin funktionieren falls\n\n* Bricklets ausgetauscht werden (z.B. verwenden wir keine fixen UIDs),\n* Brick Daemon läuft nicht oder wird neu gestartet,\n* WIFI Extension ist außer Reichweite oder\n* Wetterstation wurde neu gestartet (Stromausfall oder USB getrennt).\n\nIm Folgenden werden wir Schritt für Schritt zeigen wie diese Ziele erreicht\nwerden können.\n<<<goals\n'}
write_to_lcd_steps = {'en': '\n.. |step1_start_off| replace::\n To start off, we need to define where our program should connect to:\n\n.. |step1_ip_address| replace::\n If the WIFI Extension is used or if the Brick Daemon is\n running on a different PC, you have to exchange "localhost" with the\n IP address or hostname of the WIFI Extension or PC.\n\n.. |step1_register_callbacks| replace::\n When the program is started, we need to register the |ref_CALLBACK_ENUMERATE|\n |callback| and the |ref_CALLBACK_CONNECTED| |callback| and trigger a first\n enumerate:\n\n.. |step1_enumerate_callback| replace::\n The enumerate callback is triggered if a Brick gets connected over USB or if\n the |ref_enumerate| function is called. This allows to discover the Bricks and\n Bricklets in a stack without knowing their types or UIDs beforehand.\n\n.. |step1_connected_callback| replace::\n The connected callback is triggered if the connection to the WIFI Extension or\n to the Brick Daemon got established. In this callback we need to trigger the\n enumerate again, if the reason is an auto reconnect:\n\n.. |step1_auto_reconnect_callback| replace::\n An auto reconnect means, that the connection to the WIFI Extension or to the\n Brick Daemon was lost and could subsequently be established again. In this\n case the Bricklets may have lost their configurations and we have to\n reconfigure them. Since the configuration is done during the\n enumeration process (see below), we have to trigger another enumeration.\n\n.. |step1_put_together| replace::\n Step 1 put together:\n\n.. |step2_intro| replace::\n During the enumeration we want to configure all of the weather measuring\n Bricklets. Doing this during the enumeration ensures that Bricklets get\n reconfigured if the stack was disconnected or there was a power loss.\n\n.. |step2_enumerate| replace::\n The configurations should be performed on first startup\n (|ENUMERATION_TYPE_CONNECTED|) as well as whenever the enumeration is\n triggered externally by us (|ENUMERATION_TYPE_AVAILABLE|):\n\n.. |step2_lcd_config| replace::\n The LCD 20x4 configuration is simple, we want the current text cleared and\n we want the backlight on:\n\n.. |step2_other_config1| replace::\n We configure the Ambient Light, Humidity and Barometer Bricklet to\n return their respective measurements continuously with a period of\n 1000ms (1s):\n\n.. |step2_other_config2| replace::\n This means that the Bricklets will call the |cb_illuminance|, |cb_humidity|\n and |cb_air_pressure| callback functions whenever the value has changed, but\n with a maximum period of 1000ms.\n\n.. |step2_put_together| replace::\n Step 2 put together:\n\n.. |step3_intro| replace::\n We want a neat arrangement of the measurements on the display, such as\n\n.. |step3_format| replace::\n The decimal marks and the units should be below each other. To achieve this\n we use two characters for the unit, two decimal places and crop the name\n to use the maximum characters that are left. That\'s why "Illuminanc" is missing\n its final "e".\n\n.. |step3_printf| replace::\n The code above converts a floating point value to a string according to the given\n `format specification <https://en.wikipedia.org/wiki/Printf_format_string>`__.\n The result will be at least 6 characters long with 2 decimal places, filled up\n with spaces from the left if it would be shorter than 6 characters otherwise.\n\n.. |step3_temperature| replace::\n We are still missing the temperature. The Barometer Bricklet can\n measure temperature, but it doesn\'t have a callback for it. As a\n simple workaround we can retrieve the temperature in the |cb_air_pressure|\n callback function:\n\n.. |step3_put_together| replace::\n Step 3 put together:\n\n.. |step3_complete| replace::\n That\'s it. If we would copy these three steps together in one file and\n execute it, we would have a working Weather Station!\n\n.. |step3_suggestions1| replace::\n There are some obvious ways to make the output better.\n The name could be cropped according to the exact space that is available\n (depending on the number of digits of the measured value). Also, reading the\n temperature in the |cb_air_pressure| callback function is suboptimal. If the\n air pressure doesn\'t change, we won\'t update the temperature.\n\n.. |step3_suggestions2_common| replace::\n It would be better to read the temperature in a different thread in an endless\n loop with a one second sleep after each read. But we want to keep this code as\n simple as possible.\n\n.. |step3_suggestions2_no_thread| replace::\n It would be better to read the temperature every second triggered by an\n additional timer. But we want to keep this code as simple as possible.\n\n.. |step3_robust1| replace::\n However, we do not meet all of our goals yet. The program is not yet\n robust enough. What happens if it can\'t connect on startup? What happens if\n the enumerate after an auto reconnect doesn\'t work?\n\n.. |step3_robust2| replace::\n What we need is error handling!\n\n.. |step4_intro1| replace::\n On startup, we need to try to connect until the connection works:\n\n.. |step4_intro2| replace::\n and we need to try enumerating until the message goes through:\n\n.. |step4_sleep_in_c| replace::\n There is no portable sleep function in C. On Windows ``windows.h`` declares\n a ``Sleep`` function that takes the duration in milliseconds. On POSIX\n systems such as Linux and macOS there is a ``sleep`` function declared\n in ``unistd.h`` that takes the duration in seconds.\n\n.. |step4_connect_afterwards| replace::\n With these changes it is now possible to first start the program and\n connect the Weather Station afterwards.\n\n.. |step4_lcd_initialized1| replace::\n We also need to make sure, that we only write to the LCD if it is\n already initialized:\n\n.. |step4_lcd_initialized2| replace::\n and that we have to deal with errors during the initialization:\n\n.. |step4_logging1| replace::\n Additionally we added some logging. With the logging we can later find out\n what exactly caused a problem, if the Weather Station failed for some\n time period.\n\n.. |step4_logging2| replace::\n For example, if we connect to the Weather Station via Wi-Fi and we have\n regular auto reconnects, it likely means that the Wi-Fi connection is not\n very stable.\n\n.. |step5_intro| replace::\n That\'s it! We are already done with our Weather Station and all of the\n goals should be met.\n\n.. |step5_put_together| replace::\n Now all of the above put together\n', 'de': '\n.. |step1_start_off| replace::\n Als Erstes legen wir fest wohin unser Programm sich verbinden soll:\n\n.. |step1_ip_address| replace::\n Falls eine WIFI Extension verwendet wird, oder der Brick Daemon auf einem\n anderen PC läuft, dann muss "localhost" durch die IP Adresse oder den Hostnamen\n der WIFI Extension oder des anderen PCs ersetzt werden.\n\n.. |step1_register_callbacks| replace::\n Nach dem Start des Programms müssen der |ref_CALLBACK_ENUMERATE| |callback|\n und der |ref_CALLBACK_CONNECTED| |callback| registriert und ein erstes\n Enumerate ausgelöst werden:\n\n.. |step1_enumerate_callback| replace::\n Der Enumerate Callback wird ausgelöst wenn ein Brick per USB angeschlossen wird\n oder wenn die |ref_enumerate| Funktion aufgerufen wird. Dies ermöglicht es die\n Bricks und Bricklets im Stapel zu erkennen ohne im Voraus ihre UIDs kennen zu\n müssen.\n\n.. |step1_connected_callback| replace::\n Der Connected Callback wird ausgelöst wenn die Verbindung zur WIFI Extension\n oder zum Brick Daemon hergestellt wurde. In diesem Callback muss wiederum ein\n Enumerate angestoßen werden, wenn es sich um ein Auto-Reconnect handelt:\n\n.. |step1_auto_reconnect_callback| replace::\n Ein Auto-Reconnect bedeutet, dass die Verbindung zur WIFI Extension oder zum\n Brick Daemon verloren gegangen ist und automatisch wiederhergestellt werden\n konnte. In diesem Fall kann es sein, dass die Bricklets ihre Konfiguration\n verloren haben und wir sie neu konfigurieren müssen. Da die Konfiguration\n beim Enumerate (siehe unten) durchgeführt wird, lösen wir einfach noch ein\n Enumerate aus.\n\n.. |step1_put_together| replace::\n Schritt 1 zusammengefügt:\n\n.. |step2_intro| replace::\n Während des Enumerierungsprozesse sollen alle messenden Bricklets konfiguriert\n werden. Dadurch ist sichergestellt, dass sie neu konfiguriert werden nach\n einem Verbindungsabbruch oder einer Unterbrechung der Stromversorgung.\n\n.. |step2_enumerate| replace::\n Die Konfiguration soll beim ersten Start (|ENUMERATION_TYPE_CONNECTED|)\n durchgeführt werden und auch bei jedem extern ausgelösten Enumerate\n (|ENUMERATION_TYPE_AVAILABLE|):\n\n.. |step2_lcd_config| replace::\n Die Konfiguration des LCD 20x4 ist einfach, wir löschen den aktuellen Inhalt\n des Displays und schalten das Backlight ein:\n\n.. |step2_other_config1| replace::\n Das Ambient Light, Humidity und Barometer Bricklet werden so eingestellt, dass\n sie uns ihre jeweiligen Messwerte höchsten mit einer Periode von 1000ms (1s)\n mitteilen:\n\n.. |step2_other_config2| replace::\n Dies bedeutet, dass die Bricklets die |cb_illuminance|, |cb_humidity|\n und |cb_air_pressure| Callback-Funktionen immer dann aufrufen wenn sich der\n Messwert verändert hat, aber höchsten alle 1000ms.\n\n.. |step2_put_together| replace::\n Schritt 2 zusammengefügt:\n\n.. |step3_intro| replace::\n Wir wollen eine hübsche Darstellung der Messwerte auf dem Display. Zum Beispiel\n\n.. |step3_format| replace::\n Die Dezimaltrennzeichen und die Einheiten sollen in jeweils einer Spalte\n übereinander stehen. Daher verwenden wird zwei Zeichen für jede Einheit, zwei\n Nachkommastellen und kürzen die Namen so, dass sie in den restlichen Platz der\n jeweiligen Zeile passen. Das ist auch der Grund, warum dem "Illuminanc" das\n letzte "e" fehlt.\n\n.. |step3_printf| replace::\n Der obige Ausdruck wandelt eine Fließkommazahl in eine Zeichenkette um,\n gemäß der gegebenen `Formatspezifikation\n <https://en.wikipedia.org/wiki/Printf_format_string>`__. Das Ergebnis ist dann\n mindestens 6 Zeichen lang mit 2 Nachkommastellen. Fall es weniger als 6 Zeichen\n sind wird von Links mit Leerzeichen aufgefüllt.\n\n.. |step3_temperature| replace::\n Es fehlt noch die Temperatur. Das Barometer Bricklet kann auch die Temperatur\n messen, aber es hat dafür keinen Callback. Als einfacher Workaround können wir\n die Temperatur in der |cb_air_pressure| Callback-Funktion abfragen:\n\n.. |step3_put_together| replace::\n Schritt 3 zusammengefügt:\n\n.. |step3_complete| replace::\n Das ist es. Wenn wir diese drei Schritte zusammen in eine Datei kopieren und\n ausführen, dann hätten wir jetzt eine funktionierenden Wetterstation.\n\n.. |step3_suggestions1| replace::\n Es gibt einige offensichtliche Möglichkeiten die Ausgabe der Messdaten noch zu\n verbessern. Die Namen könnten dynamisch exakt gekürzt werden, abhängig vom\n aktuell freien Raum der jeweiligen Zeile. Auch könnten die Namen können noch\n ins  Deutsche übersetzt werden. Ein anderes Problem ist die Abfrage der\n Temperatur in der |cb_air_pressure| Callback-Funktion. Wenn sich der Luftdruck\n nicht ändert dann wird auch die Anzeige der Temperatur nicht aktualisiert, auch\n wenn sich diese eigentlich geändert hat.\n\n.. |step3_suggestions2_common| replace::\n Es wäre besser die Temperatur jede Sekunde in einem eigenen Thread anzufragen.\n Aber wir wollen das Programm für den Anfang einfach halten.\n\n.. |step3_suggestions2_no_thread| replace::\n Es wäre besser die Temperatur jede Sekunde über einen eigenen Timmer anzufragen.\n Aber wir wollen das Programm für den Anfang einfach halten.\n\n.. |step3_robust1| replace::\n Wie dem auch sei, wir haben noch nicht alle Ziele erreicht. Das Programm ist\n noch nicht robust genug. Was passiert wenn die Verbindung beim Start des\n Programms nicht hergestellt werden kann, oder wenn das Enumerate nach einem\n Auto-Reconnect nicht funktioniert?\n\n.. |step3_robust2| replace::\n Wir brauchen noch Fehlerbehandlung!\n\n.. |step4_intro1| replace::\n Beim Start des Programms versuchen wir solange die Verbindung herzustellen,\n bis es klappt:\n\n.. |step4_intro2| replace::\n und es wird solange versucht ein Enumerate zu starten bis auch dis geklappt\n hat:\n\n.. |step4_sleep_in_c| replace::\n Es gibt keine portable Sleep Funktion in C. Auf Windows deklariert `windows.h``\n eine ``Sleep`` Funktion die die Wartedauer in Millisekunden übergeben bekommt.\n Auf POSIX Systemen wie Linux und macOS gibt es eine ``sleep`` Funktion\n deklariert in ``unistd.h`` die die Wartedauer in Sekunden übergeben bekommt.\n\n.. |step4_connect_afterwards| replace::\n Mit diesen Änderungen kann das Programm schon gestartet werden bevor die\n Wetterstation angeschlossen ist.\n\n.. |step4_lcd_initialized1| replace::\n Es muss auch sichergestellt werden, dass wir nur auf das LCD schreiben nachdem\n es initialisiert wurde:\n\n.. |step4_lcd_initialized2| replace::\n und es müssen mögliche Fehler während des Enumerierungsprozesses behandelt\n werden:\n\n.. |step4_logging1| replace::\n Zusätzlich wollen wir noch ein paar Logausgaben einfügen. Diese ermöglichen es\n später herauszufinden was ein Problem ausgelöst hat, wenn die Wetterstation\n nach einer Weile möglicherweise nicht mehr funktioniert wie erwartet.\n\n.. |step4_logging2| replace::\n Zum Beispiel, wenn die Wetterstation über WLAN angebunden ist und häufig\n Auto-Reconnects auftreten, dann ist wahrscheinlich die WLAN Verbindung nicht\n sehr stabil.\n\n.. |step5_intro| replace::\n Jetzt sind alle für diese Projekt gesteckten Ziele erreicht.\n\n.. |step5_put_together| replace::\n Das gesamte Programm für die Wetterstation\n'}

def make_substitutions():
    if False:
        i = 10
        return i + 15
    substitutions = ''
    formated_binding_names = []
    for bindings_info in bindings_infos:
        if bindings_info.is_programming_language and bindings_info.is_released:
            formated_binding_names.append(binding_name[lang].format(bindings_info.display_name[lang], bindings_info.url_part))
    substitutions += binding_names[lang].format(', '.join(formated_binding_names)) + '\n'
    example_lines = []
    for bindings_info in bindings_infos:
        if bindings_info.url_part in examples and bindings_info.is_programming_language and bindings_info.is_released:
            example_lines.append(write_to_lcd_example_line[lang].format(examples[bindings_info.url_part], bindings_info.url_part))
    substitutions += write_to_lcd_examples[lang].format(', '.join(example_lines))
    example_download_lines = []
    for bindings_info in bindings_infos:
        if bindings_info.url_part in examples and bindings_info.is_programming_language and bindings_info.is_released:
            example_download_lines.append(write_to_lcd_example_download_line[lang].format(examples[bindings_info.url_part], bindings_info.url_part))
    substitutions += write_to_lcd_example_downloads[lang].format(', '.join(example_download_lines))
    return substitutions

def make_common_substitutions(bindings_info):
    if False:
        return 10
    substitutions = ''
    if bindings_info.url_part in examples:
        substitutions += common_intro[lang].format(examples[bindings_info.url_part], bindings_info.url_part, bindings_info.tutorial[lang])
    return substitutions

def make_write_to_lcd_substitutions():
    if False:
        i = 10
        return i + 15
    substitutions = ''
    substitutions += write_to_lcd_goals[lang] + '\n'
    substitutions += '>>>substitutions\n'
    substitutions += write_to_lcd_steps[lang] + '\n'
    substitutions += '<<<substitutions\n'
    return substitutions

def make_write_to_lcd_toctree():
    if False:
        print('Hello World!')
    toctree_lines = []
    for bindings_info in bindings_infos:
        if bindings_info.url_part in examples:
            toctree_lines.append(write_to_lcd_examples_toctree_line[lang].format(bindings_info.display_name[lang], bindings_info.software_doc_suffix))
    return write_to_lcd_examples_toctree[lang].format('\n'.join(toctree_lines))

def write_if_changed(path, content):
    if False:
        print('Hello World!')
    if os.path.exists(path):
        with open(path, 'r') as f:
            existing = f.read()
        if existing == content:
            return
    with open(path, 'w') as f:
        f.write(content)

def generate(path):
    if False:
        while True:
            i = 10
    global lang
    if path.endswith('/en'):
        lang = 'en'
    elif path.endswith('/de'):
        lang = 'de'
    else:
        debug('Wrong working directory')
        sys.exit(1)
    generate_tables.lang = lang
    debug('Generating WeatherStation.substitutions')
    write_if_changed(os.path.join(path, 'source', 'Kits', 'WeatherStation', 'WeatherStation.substitutions'), make_substitutions())
    for bindings_info in bindings_infos:
        if bindings_info.url_part in examples:
            debug('Generating {0}Common.substitutions (WeatherStation)'.format(bindings_info.software_doc_suffix))
            write_if_changed(os.path.join(path, 'source', 'Kits', 'WeatherStation', bindings_info.software_doc_suffix + 'Common.substitutions'), make_common_substitutions(bindings_info))
    debug('Generating WriteToLCD.substitutions (WeatherStation)')
    write_if_changed(os.path.join(path, 'source', 'Kits', 'WeatherStation', 'WriteToLCD.substitutions'), make_write_to_lcd_substitutions())
    debug('Generating WriteToLCD.toctree (WeatherStation)')
    write_if_changed(os.path.join(path, 'source', 'Kits', 'WeatherStation', 'WriteToLCD.toctree'), make_write_to_lcd_toctree())
if __name__ == '__main__':
    generate(os.getcwd())