from yaml import safe_load
from sys import stdout, stderr

def combine(items):
    if False:
        for i in range(10):
            print('nop')
    if len(items) <= 2:
        return ' and '.join(items)
    else:
        return ', '.join(items[:-1]) + ', and ' + items[-1]
if __name__ == '__main__':
    stdout.write('---\nsuppress-bibliography: true\n---\n\n```{r console_start, include=FALSE}\nconsole_start()\n```\n\n```{console setup_history, include=FALSE}\n export CHAPTER="tools"\n export HISTFILE=/history/history_${CHAPTER}\n rm -f $HISTFILE\n```\n\n\n<!--A[appendix]\n[[appendix-tools]]\nA-->\n# List of Command-Line Tools {-}\n\nThis is an overview of all the command-line tools discussed in this book.\nThis includes binary executables, interpreted scripts, and Z Shell builtins and keywords.\nFor each command-line tool, the following information, when available and appropriate, is provided:\n\n- The actual command to type at the command line\n- A description\n- The version used in the book\n- The year that version was released\n- The primary author(s)\n- A website to find more information\n- How to obtain help\n- An example usage\n\nAll command-line tools listed here are included in the Docker image.\nSee [Chapter 2](#chapter-2-getting-started) for instructions on how to set it up.\nPlease note that citing open source software is not trivial, and that some information may be missing or incorrect.\n\n```{console, include=FALSE}\nunalias csvlook\nunalias parallel\n```\n\n\n')
    with open('../tools.yml') as file:
        tools = safe_load(file)
    for (i, (name, tool)) in enumerate(sorted(tools.items(), key=lambda x: x[0].lower()), start=1):
        stderr.write(f'{i}: {name}\n')
        stdout.write(f'## {name} {{-}}\n\n')
        stdout.write(f"{tool['description']}.\n`{name}`\n")
        if tool.get('builtin', False):
            stdout.write(f'is a Z shell builtin.\n')
        if 'version' in tool:
            stdout.write(f"(version {tool['version']})\n")
        if 'author' in tool:
            stdout.write(f"by {combine(tool['author'])} ({tool['year']}).\n")
        if 'note' in tool:
            stdout.write(f"{tool['note']}.\n")
        if 'url' in tool:
            stdout.write(f"More information: {tool['url']}.\n")
        stdout.write(f'\n```{{console {name}}}\n')
        stdout.write(f'type {name}\n')
        if 'help' in tool:
            if tool['help'] == 'man':
                stdout.write(f'man {name}')
            elif tool['help'] == '--help':
                stdout.write(f'{name} --help')
            else:
                stdout.write(f"{tool['help']}")
            stdout.write('#!enter=FALSE\nC-C#!literal=FALSE\n')
        if 'example' in tool:
            stdout.write(f"{tool['example'].strip()}\n")
        stdout.write('```\n')
        stdout.write('\n\n')