from __future__ import annotations
import os

def _manual_substitution(line: str, replacements: dict[str, str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    for (value, repl) in replacements.items():
        line = line.replace(f'|{value}|', repl)
    return line

def build_postprocess(app, exception):
    if False:
        return 10
    'Sphinx "build-finished" event handler.'
    from sphinx.builders import html as builders
    if exception or not isinstance(app.builder, builders.StandaloneHTMLBuilder):
        return
    global_substitutions = app.config.global_substitutions
    for path in app.config.html_extra_with_substitutions:
        with open(path) as file:
            with open(os.path.join(app.outdir, os.path.basename(path)), 'w') as output_file:
                for line in file:
                    output_file.write(_manual_substitution(line, global_substitutions))
    for path in app.config.manual_substitutions_in_generated_html:
        with open(os.path.join(app.outdir, os.path.dirname(path), os.path.basename(path))) as input_file:
            content = input_file.readlines()
        with open(os.path.join(app.outdir, os.path.dirname(path), os.path.basename(path)), 'w') as output_file:
            for line in content:
                output_file.write(_manual_substitution(line, global_substitutions))

def setup(app):
    if False:
        return 10
    'Setup plugin'
    app.connect('build-finished', build_postprocess)
    app.add_config_value('html_extra_with_substitutions', [], 'html', [str])
    app.add_config_value('manual_substitutions_in_generated_html', [], 'html', [str])
    app.add_config_value('global_substitutions', {}, 'html', [dict])
    return {'parallel_write_safe': True, 'parallel_read_safe': True}