"""
Generate an index table and rendered pages for the common APIs.

The top-level index file should look like
```
## Initialization
Function | Description
-------- | -----------
[rerun.init()](initialization/#rerun.init) | Initialize the Rerun SDK …
[rerun.connect()](initialization/#rerun.connect) | Connect to a remote Rerun Viewer on the …
[rerun.spawn()](initialization/#rerun.spawn) | Spawn a Rerun Viewer …
…

The Summary should look like:
```
* [index](index.md)
* [Initialization](initialization.md)
* [Logging Primitives](primitives.md)
* [Logging Images](images.md)
* [Annotations](annotation.md)
* [Extension Components](extension_components.md)
* [Plotting](plotting.md)
* [Transforms](transforms.md)
* [Helpers](helpers.md)
```
"""
from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import griffe
import mkdocs_gen_files

def all_archetypes() -> list[str]:
    if False:
        while True:
            i = 10
    file_path = Path(__file__).parent.parent.parent.joinpath('rerun_py/rerun_sdk/rerun/archetypes/__init__.py')
    quoted_strings = []
    pattern = '"([^"]*)"'
    with open(file_path) as file:
        for line in file:
            matches = re.findall(pattern, line)
            quoted_strings.extend(matches)
    assert len(quoted_strings) > 0, f'Found no archetypes in {file_path}'
    return quoted_strings

@dataclass
class Section:
    title: str
    func_list: list[str] | None = None
    class_list: list[str] | None = None
    gen_page: bool = True
    mod_path: str = 'rerun'
    show_tables: bool = True
    default_filters: bool = True
    show_submodules: bool = False
SECTION_TABLE: Final[list[Section]] = [Section(title='Initialization functions', func_list=['init', 'connect', 'disconnect', 'save', 'serve', 'spawn', 'memory_recording']), Section(title='Logging functions', func_list=['log', 'set_time_sequence', 'set_time_seconds', 'set_time_nanos']), Section(title='Archetypes', mod_path='rerun.archetypes', show_tables=False), Section(title='Components', mod_path='rerun.components', show_tables=False), Section(title='Datatypes', mod_path='rerun.datatypes', show_tables=False), Section(title='Custom Data', class_list=['AnyValues']), Section(title='Clearing Entities', class_list=['archetypes.Clear'], gen_page=False), Section(title='Annotations', class_list=['archetypes.AnnotationContext', 'datatypes.AnnotationInfo', 'datatypes.ClassDescription'], gen_page=False), Section(title='Images', class_list=['archetypes.DepthImage', 'archetypes.Image', 'ImageEncoded', 'archetypes.SegmentationImage'], gen_page=False), Section(title='Image Helpers', class_list=['ImageEncoded'], show_tables=False), Section(title='Plotting', class_list=['archetypes.BarChart', 'archetypes.TimeSeriesScalar'], gen_page=False), Section(title='Spatial Archetypes', class_list=['archetypes.Arrows3D', 'archetypes.Asset3D', 'archetypes.Boxes2D', 'archetypes.Boxes3D', 'archetypes.LineStrips2D', 'archetypes.LineStrips3D', 'archetypes.Mesh3D', 'archetypes.Points2D', 'archetypes.Points3D'], gen_page=False), Section(title='Tensors', class_list=['archetypes.Tensor'], gen_page=False), Section(title='Text', class_list=['LoggingHandler', 'archetypes.TextDocument', 'archetypes.TextLog'], gen_page=False), Section(title='Transforms and Coordinate Systems', class_list=['archetypes.DisconnectedSpace', 'archetypes.Pinhole', 'archetypes.Transform3D', 'archetypes.ViewCoordinates', 'datatypes.Quaternion', 'datatypes.RotationAxisAngle', 'datatypes.Scale3D', 'datatypes.TranslationAndMat3x3', 'datatypes.TranslationRotationScale3D'], gen_page=False), Section(title='Enums', mod_path='rerun', class_list=['Box2DFormat', 'ImageFormat', 'MeshFormat'], show_tables=False), Section(title='Interfaces', mod_path='rerun', class_list=['AsComponents', 'ComponentBatchLike'], default_filters=False), Section(title='Script Helpers', func_list=['script_add_args', 'script_setup', 'script_teardown']), Section(title='Other classes and functions', show_tables=False, func_list=['get_data_recording', 'get_global_data_recording', 'get_recording_id', 'get_thread_local_data_recording', 'is_enabled', 'log_components', 'new_recording', 'set_global_data_recording', 'set_thread_local_data_recording', 'start_web_viewer_server'], class_list=['RecordingStream', 'LoggingHandler', 'MemoryRecording']), Section(title='Demo utilities', show_tables=False, mod_path='rerun_demo', show_submodules=True), Section(title='Experimental', func_list=['add_space_view', 'new_blueprint', 'set_auto_space_views', 'set_panels'], show_tables=False, mod_path='rerun.experimental')]

def is_mentioned(thing: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    for section in SECTION_TABLE:
        if section.class_list is not None:
            if f'archetypes.{thing}' in section.class_list:
                return True
    return False
root = Path(__file__).parent.parent.joinpath('rerun_sdk').resolve()
common_dir = Path('common')
for archetype in all_archetypes():
    assert is_mentioned(archetype), f"Archetype '{archetype}' is not mentioned in the index of {__file__}"
search_paths = [path for path in sys.path if path]
search_paths.insert(0, root.as_posix())
rerun_pkg = griffe.load('rerun', search_paths=search_paths)
nav = mkdocs_gen_files.Nav()
nav['index'] = 'index.md'
index_path = common_dir.joinpath('index.md')

def make_slug(s: str) -> str:
    if False:
        return 10
    s = s.lower().strip()
    s = re.sub('[\\s]+', '_', s)
    return s
with mkdocs_gen_files.open(index_path, 'w') as index_file:
    index_file.write("\n## Getting Started\n* [Quick start](https://www.rerun.io/docs/getting-started/python)\n* [Tutorial](https://www.rerun.io/docs/getting-started/logging-python)\n* [Examples on GitHub](https://github.com/rerun-io/rerun/tree/latest/examples/python)\n* [Troubleshooting](https://www.rerun.io/docs/getting-started/troubleshooting)\n\nThere are many different ways of sending data to the Rerun Viewer depending on what you're trying\nto achieve and whether the viewer is running in the same process as your code, in another process,\nor even as a separate web application.\n\nCheckout [SDK Operating Modes](https://www.rerun.io/docs/reference/sdk-operating-modes) for an\noverview of what's possible and how.\n\n## APIs\n")
    for section in SECTION_TABLE:
        if section.gen_page:
            md_name = make_slug(section.title)
            md_file = md_name + '.md'
            nav[section.title] = md_file
            write_path = common_dir.joinpath(md_file)
            with mkdocs_gen_files.open(write_path, 'w') as fd:
                fd.write(f'::: {section.mod_path}\n')
                fd.write('    options:\n')
                fd.write('      show_root_heading: True\n')
                fd.write('      heading_level: 3\n')
                fd.write('      members_order: alphabetical\n')
                if section.func_list or section.class_list:
                    fd.write('      members:\n')
                    for func_name in section.func_list or []:
                        fd.write(f'        - {func_name}\n')
                    for class_name in section.class_list or []:
                        fd.write(f'        - {class_name}\n')
                if not section.default_filters:
                    fd.write('      filters: []\n')
                if section.show_submodules:
                    fd.write('      show_submodules: True\n')
        if section.show_tables:
            index_file.write(f'### {section.title}\n')
            if section.func_list:
                index_file.write('Function | Description\n')
                index_file.write('-------- | -----------\n')
                for func_name in section.func_list:
                    func = rerun_pkg[func_name]
                    index_file.write(f'[`rerun.{func_name}()`][rerun.{func_name}] | {func.docstring.lines[0]}\n')
            if section.class_list:
                index_file.write('\n')
                index_file.write('Class | Description\n')
                index_file.write('-------- | -----------\n')
                for class_name in section.class_list:
                    cls = rerun_pkg[class_name]
                    show_class = class_name
                    for maybe_strip in ['archetypes.', 'components.', 'datatypes.']:
                        if class_name.startswith(maybe_strip):
                            stripped = class_name.replace(maybe_strip, '')
                            if stripped in rerun_pkg.classes:
                                show_class = stripped
                    index_file.write(f'[`rerun.{show_class}`][rerun.{class_name}] | {cls.docstring.lines[0]}\n')
        index_file.write('\n')
    index_file.write("\n# Troubleshooting\nYou can set `RUST_LOG=debug` before running your Python script\nand/or `rerun` process to get some verbose logging output.\n\nIf you run into any issues don't hesitate to [open a ticket](https://github.com/rerun-io/rerun/issues/new/choose)\nor [join our Discord](https://discord.gg/Gcm8BbTaAj).\n")
with mkdocs_gen_files.open(common_dir.joinpath('SUMMARY.txt'), 'w') as nav_file:
    nav_file.writelines(nav.build_literate_nav())