import re
from pathlib import Path
PYPI_BANNER = '<img src="https://raw.githubusercontent.com/litestar-org/branding/473f54621e55cde9acbb6fcab7fc03036173eb3d/assets/Branding%20-%20PNG%20-%20Transparent/Logo%20-%20Banner%20-%20Inline%20-%20Light.png" alt="Litestar Logo - Light" width="100%" height="auto" />'

def generate_pypi_readme() -> None:
    if False:
        i = 10
        return i + 15
    source = Path('README.md').read_text()
    output = re.sub('<!-- github-banner-start -->[\\w\\W]*<!-- github-banner-end -->', PYPI_BANNER, source)
    output = re.sub('<!-- contributors-start -->[\\w\\W]*<!-- contributors-end -->', '', output)
    output = re.sub('<!-- ALL-CONTRIBUTORS-BADGE:START[\\w\\W]*<!-- ALL-CONTRIBUTORS-BADGE:END -->', '', output)
    output = output.strip() + '\n'
    Path('docs/PYPI_README.md').write_text(output)
if __name__ == '__main__':
    generate_pypi_readme()