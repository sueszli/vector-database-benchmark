import locale
import os
import secrets
import ascii_magic
import rich
from rich.emoji import Emoji

def motorcycle() -> None:
    if False:
        print('Hello World!')
    print('\n                                             `\n                                         `.+yys/.`\n                                       ``/NMMMNNs`\n                                    `./shNMMMMMMNs``    `..`\n                                  `-smNMMNNMMMMMMN/.``......`\n                                `.yNMMMMNmmmmNNMMm/.`....`\n                              `:sdNMMMMMMNNNNddddds-`.`` `--. `\n                           `.+dNNNNMMMMMMMMMNNNNmddohmh//hddy/.```..`\n                          `-hNMMMMMMMMMMMMNNdmNNMNNdNNd:sdyoo+/++:..`\n                        ../mMMMMMMMMMMMMMMNNmmmmNMNmNNmdmd/hNNNd+:`\n                        `:mMNNMMMMMMMMMMMMNMNNmmmNNNNNdNNd/NMMMMm::\n                       `:mMNNNMMMMMMMMMMMMMMMNNNNdNMNNmmNd:smMMmh//\n                     ``/mMMMMMMMMMMMMMMMMMMMMMMNmdmNNMMNNNy/osoo/-`\n                    `-sNMMMMMMMMMMMMMMMMMMMMMMMMNNmmMMMMNh-....`\n                   `/dNMMMMMMMMMMMMMMMMMMMMMMMMMMMNNMMMNy.`\n                ``.omNNMMMMMMMMMMMMNMMMMMMMNmmmmNNMMMMN+`\n                `:hmNNMMMMMMMMMMMNo/ohNNNNho+os+-+hNys/`\n                -mNNNNNNMMMMMMMMm+``-yNdd+/mMMMms.-:`\n                .+dmNNNNMMMMMMNd:``:dNN+y`oMMMMMm-.`\n                `+dmmmNNNmmmmy+.   `-+m/s/+MMMMm/--\n               `+mmmhNy/-...```     ``-.-sosyys+/-`\n            ``.smmmsoo``               .oh+-:/:.\n          `.:odmdh/````             `.+d+``````\n     ```/sydNdhy+.`              ``-sds.\n    `:hdmhs::-````               `oNs.`\n```.sdmh/``                    `-ym+`\n ``ssy+`                     `-yms.`\n   ``                      `:hNy-``\n   `                     `-yMN/```\n                       `-yNhy-\n                     `/yNd/`\n                   `:dNMs``\n                 `.+mNy/.`\n              `.+hNMMs``\n             `:dMMMMh.`')

def hold_on_tight() -> None:
    if False:
        while True:
            i = 10
    out = os.popen('stty size', 'r').read().split()
    if len(out) == 2:
        (rows, columns) = out
    else:
        'not running in a proper command line (probably a unit test)'
        return
    if int(columns) >= 91:
        print("\n _   _       _     _                 _   _       _     _     _   _                       _\n| | | |     | |   | |               | | (_)     | |   | |   | | | |                     | |\n| |_| | ___ | | __| |   ___  _ __   | |_ _  __ _| |__ | |_  | |_| | __ _ _ __ _ __ _   _| |\n|  _  |/ _ \\| |/ _` |  / _ \\| '_ \\  | __| |/ _` | '_ \\| __| |  _  |/ _` | '__| '__| | | | |\n| | | | (_) | | (_| | | (_) | | | | | |_| | (_| | | | | |_  | | | | (_| | |  | |  | |_| |_|\n\\_| |_/\\___/|_|\\__,_|  \\___/|_| |_|  \\__|_|\\__, |_| |_|\\__| \\_| |_/\\__,_|_|  |_|   \\__, (_)\n                                            __/ |                                   __/ |\n                                           |___/                                   |___/\n            ")
    else:
        print("\n _   _       _     _                 _   _                       _\n| | | |     | |   | |               | | | |                     | |\n| |_| | ___ | | __| |   ___  _ __   | |_| | __ _ _ __ _ __ _   _| |\n|  _  |/ _ \\| |/ _` |  / _ \\| '_ \\  |  _  |/ _` | '__| '__| | | | |\n| | | | (_) | | (_| | | (_) | | | | | | | | (_| | |  | |  | |_| |_|\n\\_| |_/\\___/|_|\\__,_|  \\___/|_| |_| \\_| |_/\\__,_|_|  |_|   \\__, (_)\n                                                            __/ |\n                                                           |___/\n        ")

def hagrid1() -> None:
    if False:
        i = 10
        return i + 15
    from .lib import asset_path
    try:
        ascii_magic.to_terminal(ascii_magic.from_image_file(img_path=str(asset_path()) + '/img/hagrid.png', columns=83))
    except Exception:
        pass

def hagrid2() -> None:
    if False:
        return 10
    from .lib import asset_path
    try:
        ascii_magic.to_terminal(ascii_magic.from_image_file(img_path=str(asset_path()) + '/img/hagrid2.png', columns=83))
    except Exception:
        pass

def quickstart_art() -> None:
    if False:
        while True:
            i = 10
    text = '\n888    888        d8888  .d8888b.          d8b      888\n888    888       d88888 d88P  Y88b         Y8P      888\n888    888      d88P888 888    888                  888\n8888888888     d88P 888 888        888d888 888  .d88888\n888    888    d88P  888 888  88888 888P"   888 d88" 888\n888    888   d88P   888 888    888 888     888 888  888\n888    888  d8888888888 Y88b  d88P 888     888 Y88b 888\n888    888 d88P     888  "Y8888P88 888     888  "Y88888\n\n\n .d88888b.           d8b          888               888                     888\nd88P" "Y88b          Y8P          888               888                     888\n888     888                       888               888                     888\n888     888 888  888 888  .d8888b 888  888 .d8888b  888888  8888b.  888d888 888888\n888     888 888  888 888 d88P"    888 .88P 88K      888        "88b 888P"   888\n888 Y8b 888 888  888 888 888      888888K  "Y8888b. 888    .d888888 888     888\nY88b.Y8b88P Y88b 888 888 Y88b.    888 "88b      X88 Y88b.  888  888 888     Y88b.\n "Y888888"   "Y88888 888  "Y8888P 888  888  88888P\'  "Y888 "Y888888 888      "Y888\n       Y8b\n'
    console = rich.get_console()
    console.print(text, style='bold', justify='left', new_line_start=True)

def hagrid() -> None:
    if False:
        print('Hello World!')
    'Print a random hagrid image with the caption "hold on tight harry".'
    options = [motorcycle, hagrid1, hagrid2]
    i = secrets.randbelow(3)
    options[i]()
    hold_on_tight()

class RichEmoji(Emoji):

    def to_str(self) -> str:
        if False:
            while True:
                i = 10
        return self._char.encode('utf-8').decode(locale.getpreferredencoding())