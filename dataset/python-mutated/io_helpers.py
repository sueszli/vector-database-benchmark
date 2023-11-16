from pathlib import Path

def create_file(file, content):
    if False:
        while True:
            i = 10
    path = Path(file)
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding='utf-8')

def delete_file(file):
    if False:
        while True:
            i = 10
    Path(file).unlink()