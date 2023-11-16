import os
from typing import Any, Dict, Generator, Iterable, Type
from isort.profiles import profiles
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../docs/configuration/profiles.md'))
HEADER = 'Built-in Profile for isort\n========\n\nThe following profiles are built into isort to allow easy interoperability with\ncommon projects and code styles.\n\nTo use any of the listed profiles, use `isort --profile PROFILE_NAME` from the command line, or `profile=PROFILE_NAME` in your configuration file.\n\n'

def format_profile(profile_name: str, profile: Dict[str, Any]) -> str:
    if False:
        print('Hello World!')
    options = '\n'.join((f' - **{name}**: `{repr(value)}`' for (name, value) in profile.items()))
    return f"\n#{profile_name}\n\n{profile.get('description', '')}\n{options}\n"

def document_text() -> str:
    if False:
        for i in range(10):
            print('nop')
    return f"{HEADER}{''.join((format_profile(profile_name, profile) for (profile_name, profile) in profiles.items()))}"

def write_document():
    if False:
        print('Hello World!')
    with open(OUTPUT_FILE, 'w') as output_file:
        output_file.write(document_text())
if __name__ == '__main__':
    write_document()