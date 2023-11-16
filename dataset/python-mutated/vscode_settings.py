from pathlib import Path
try:
    import json5 as json
    HAS_JSON5 = True
except ImportError:
    import json
    HAS_JSON5 = False
ROOT_FOLDER = Path(__file__).absolute().parent.parent
VSCODE_FOLDER = ROOT_FOLDER / '.vscode'
RECOMMENDED_SETTINGS = VSCODE_FOLDER / 'settings_recommended.json'
SETTINGS = VSCODE_FOLDER / 'settings.json'

def deep_update(d: dict, u: dict) -> dict:
    if False:
        for i in range(10):
            print('nop')
    for (k, v) in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        elif isinstance(v, list):
            d[k] = d.get(k, []) + v
        else:
            d[k] = v
    return d

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    recommended_settings = json.loads(RECOMMENDED_SETTINGS.read_text())
    try:
        current_settings_text = SETTINGS.read_text()
    except FileNotFoundError:
        current_settings_text = '{}'
    try:
        current_settings = json.loads(current_settings_text)
    except ValueError as ex:
        if HAS_JSON5:
            raise SystemExit('Failed to parse .vscode/settings.json.') from ex
        raise SystemExit('Failed to parse .vscode/settings.json. Maybe it contains comments or trailing commas. Try `pip install json5` to install an extended JSON parser.') from ex
    settings = deep_update(current_settings, recommended_settings)
    SETTINGS.write_text(json.dumps(settings, indent=4) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()