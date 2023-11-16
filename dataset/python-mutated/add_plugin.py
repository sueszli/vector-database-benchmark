"""Generate boilerplate for a new Flake8 plugin.

Example usage:

    python scripts/add_plugin.py         flake8-pie         --url https://pypi.org/project/flake8-pie/         --prefix PIE
"""
from __future__ import annotations
import argparse
from _utils import ROOT_DIR, dir_name, get_indent, pascal_case

def main(*, plugin: str, url: str, prefix_code: str) -> None:
    if False:
        i = 10
        return i + 15
    'Generate boilerplate for a new plugin.'
    (ROOT_DIR / 'crates/ruff_linter/resources/test/fixtures' / dir_name(plugin)).mkdir(exist_ok=True)
    plugin_dir = ROOT_DIR / 'crates/ruff_linter/src/rules' / dir_name(plugin)
    plugin_dir.mkdir(exist_ok=True)
    with (plugin_dir / 'mod.rs').open('w+') as fp:
        fp.write(f'//! Rules from [{plugin}]({url}).\n')
        fp.write('pub(crate) mod rules;\n')
        fp.write('\n')
        fp.write('#[cfg(test)]\nmod tests {\n    use std::convert::AsRef;\n    use std::path::Path;\n\n    use anyhow::Result;\n    use test_case::test_case;\n\n    use crate::registry::Rule;\n    use crate::test::test_path;\n    use crate::{assert_messages, settings};\n\n    fn rules(rule_code: Rule, path: &Path) -> Result<()> {\n        let snapshot = format!("{}_{}", rule_code.as_ref(), path.to_string_lossy());\n        let diagnostics = test_path(\n            Path::new("%s").join(path).as_path(),\n            &settings::Settings::for_rule(rule_code),\n        )?;\n        assert_messages!(snapshot, diagnostics);\n        Ok(())\n    }\n}\n' % dir_name(plugin))
    rules_dir = plugin_dir / 'rules'
    rules_dir.mkdir(exist_ok=True)
    (rules_dir / 'mod.rs').touch()
    (plugin_dir / 'snapshots').mkdir(exist_ok=True)
    rules_mod_path = ROOT_DIR / 'crates/ruff_linter/src/rules/mod.rs'
    lines = rules_mod_path.read_text().strip().splitlines()
    lines.append(f'pub mod {dir_name(plugin)};')
    lines.sort()
    rules_mod_path.write_text('\n'.join(lines) + '\n')
    content = (ROOT_DIR / 'crates/ruff_linter/src/registry.rs').read_text()
    with (ROOT_DIR / 'crates/ruff_linter/src/registry.rs').open('w') as fp:
        for line in content.splitlines():
            indent = get_indent(line)
            if line.strip() == '// ruff':
                fp.write(f'{indent}// {plugin}')
                fp.write('\n')
            elif line.strip() == '/// Ruff-specific rules':
                fp.write(f'{indent}/// [{plugin}]({url})\n')
                fp.write(f'{indent}#[prefix = "{prefix_code}"]\n')
                fp.write(f'{indent}{pascal_case(plugin)},')
                fp.write('\n')
            fp.write(line)
            fp.write('\n')
    text = ''
    with (ROOT_DIR / 'crates/ruff_linter/src/codes.rs').open('r') as fp:
        while (line := next(fp)).strip() != '// ruff':
            text += line
        text += ' ' * 8 + f'// {plugin}\n\n'
        text += line
        text += fp.read()
    with (ROOT_DIR / 'crates/ruff_linter/src/codes.rs').open('w') as fp:
        fp.write(text)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate boilerplate for a new Flake8 plugin.', epilog='Example usage: python scripts/add_plugin.py flake8-pie --url https://pypi.org/project/flake8-pie/')
    parser.add_argument('plugin', type=str, help='The name of the plugin to generate.')
    parser.add_argument('--url', required=True, type=str, help='The URL of the latest release in PyPI.')
    parser.add_argument('--prefix', required=False, default='TODO', type=str, help='Prefix code for the plugin. Leave empty to manually fill.')
    args = parser.parse_args()
    main(plugin=args.plugin, url=args.url, prefix_code=args.prefix)