def indent(text, prefix):
    if False:
        i = 10
        return i + 15

    def prefixed_lines():
        if False:
            i = 10
            return i + 15
        for line in text.splitlines(True):
            yield (prefix + line if line.strip() else line)
    return ''.join(prefixed_lines())