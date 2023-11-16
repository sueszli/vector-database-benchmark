"""
Creates debug output from data in a conversion run.
"""
from __future__ import annotations
import typing
from openage.convert.entity_object.conversion.aoc.genie_tech import AgeUpgrade, UnitLineUpgrade, BuildingLineUpgrade, UnitUnlock, BuildingUnlock
from openage.convert.entity_object.conversion.aoc.genie_unit import GenieUnitLineGroup, GenieBuildingLineGroup, GenieStackBuildingGroup, GenieUnitTransformGroup, GenieMonkGroup
from openage.convert.entity_object.export.formats.media_cache import MediaCacheFile
from openage.convert.service.conversion.internal_name_lookups import get_entity_lookups, get_tech_lookups, get_civ_lookups, get_terrain_lookups
from openage.convert.value_object.read.media.datfile.empiresdat import EmpiresDatWrapper
from openage.convert.value_object.read.read_members import IncludeMembers, MultisubtypeMember
from openage.util.fslike.filecollection import FileCollectionPath
from openage.util.fslike.path import Path
from openage.util.hash import hash_file
if typing.TYPE_CHECKING:
    from argparse import Namespace
    from openage.convert.entity_object.conversion.modpack import Modpack
    from openage.convert.entity_object.conversion.stringresource import StringResource
    from openage.convert.entity_object.conversion.aoc.genie_object_container import GenieObjectContainer
    from openage.convert.value_object.init.game_version import GameVersion
    from openage.util.fslike.directory import Directory

def debug_cli_args(debugdir: Directory, loglevel: int, args: Namespace) -> None:
    if False:
        print('Hello World!')
    '\n    Create debug output for the converter CLI args.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param args: CLI arguments.\n    :type args: Namespace\n    '
    if loglevel < 1:
        return
    logfile = debugdir['args']
    logtext = ''
    arg_dict = {}
    for (name, arg) in vars(args).items():
        arg_dict.update({name: arg})
    arg_dict = dict(sorted(arg_dict.items(), key=lambda item: item[0]))
    for (name, arg) in arg_dict.items():
        logtext += f'{name}: {arg}\n'
    with logfile.open('w') as log:
        log.write(logtext)

def debug_game_version(debugdir: Directory, loglevel: int, args: Namespace) -> None:
    if False:
        return 10
    '\n    Create debug output for the detected game version.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param args: CLI arguments.\n    :type args: Namespace\n    '
    if loglevel < 2:
        return
    logfile = debugdir.joinpath('init/')['game_version']
    logtext = ''
    logtext += f'game edition:\n    - {args.game_version.edition}\n'
    if len(args.game_version.expansions) > 0:
        logtext += 'game expansions:\n'
        for expansion in args.game_version.expansions:
            logtext += f'    - {expansion}\n'
    else:
        logtext += 'game expansions: none detected'
    with logfile.open('w') as log:
        log.write(logtext)

def debug_mounts(debugdir: Directory, loglevel: int, args: Namespace) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Create debug output for the mounted files and folders.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param args: CLI arguments.\n    :type args: Namespace\n    '
    if loglevel < 2:
        return
    logfile = debugdir.joinpath('init/')['mounts']
    logtext = ''
    mounts = args.srcdir.fsobj.obj.fsobj.mounts
    mount_dict = {}
    for mount in mounts:
        if mount[0] in mount_dict.keys():
            mount_dict[mount[0]].append(mount[1])
        else:
            mount_dict[mount[0]] = [mount[1]]
    mount_dict = dict(sorted(mount_dict.items(), key=lambda item: item[0]))
    for (mountpoint, resources) in mount_dict.items():
        if len(mountpoint) == 0:
            logtext += 'mountpoint: ${srcdir}/\n'
        else:
            logtext += f'mountpoint: ${{srcdir}}/{mountpoint[0].decode()}/\n'
        for resource in resources:
            resource_type = None
            abs_path = ''
            file_count = 0
            if type(resource) is Path:
                resource_type = 'dir'
                abs_path = resource.fsobj.path.decode()
            elif type(resource) is FileCollectionPath:
                resource_type = 'file collection'
                abs_path = resource.fsobj.fileobj.name.decode()
                file_count = len(resource.fsobj.rootentries[0])
            logtext += f'    resource type: {resource_type}\n'
            logtext += f'    source path: {abs_path}\n'
            if resource_type == 'file collection':
                logtext += f'    file count: {file_count}\n'
            logtext += '    ----\n'
    with logfile.open('w') as log:
        log.write(logtext)

def debug_gamedata_format(debugdir: Directory, loglevel: int, game_version: GameVersion) -> None:
    if False:
        print('Hello World!')
    '\n    Create debug output for the converted .dat format.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param game_version: Game version the .dat file comes with.\n    :type game_version: GameVersion\n    '
    if loglevel < 2:
        return
    logfile = debugdir.joinpath('read/')['data_format']
    logtext = ''
    discovered_structs = {EmpiresDatWrapper}
    handled_structs = set()
    while discovered_structs:
        struct = discovered_structs.pop()
        if struct in handled_structs:
            continue
        members = struct.get_data_format_members(game_version)
        logtext += f'total member count: {len(members)}\n'
        max_name_width = 1
        max_vmemb_width = 1
        for member in members:
            if len(str(member[1])) > max_name_width:
                max_name_width = len(str(member[1]))
            if len(str(member[2])) > max_vmemb_width:
                max_vmemb_width = len(str(member[2]))
            if isinstance(member[3], IncludeMembers):
                discovered_structs.add(member[3].cls)
            elif isinstance(member[3], MultisubtypeMember):
                discovered_structs.update(member[3].class_lookup.values())
        for member in members:
            logtext += f'{str(member[0].value):8}  {str(member[1]):{max_name_width}}  {str(member[2]):{max_vmemb_width}}  {str(member[3])}\n'
        handled_structs.add(struct)
        logtext += '\n'
    with logfile.open('w') as log:
        log.write(logtext)

def debug_string_resources(debugdir: Directory, loglevel: int, string_resources: StringResource) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create debug output for found string resources.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param string_resources: Language and string information.\n    :type string_resources: StringResource\n    '
    if loglevel < 2:
        return
    logfile = debugdir.joinpath('read/')['string_resources']
    logtext = ''
    logtext += 'found languages: '
    logtext += ', '.join(string_resources.get_tables().keys())
    logtext += '\n\n'
    for (language, strings) in string_resources.get_tables().items():
        logtext += f'{language}: {len(strings)} IDs\n'
    with logfile.open('w') as log:
        log.write(logtext)

def debug_registered_graphics(debugdir: Directory, loglevel: int, existing_graphics: list[str]) -> None:
    if False:
        return 10
    '\n    Create debug output for found graphics files.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param existing_graphics: List of graphic ids of graphic files.\n    :type existing_graphics: list\n    '
    if loglevel < 2:
        return
    logfile = debugdir.joinpath('read/')['existing_graphics']
    logtext = ''
    logtext += f'file count: {len(existing_graphics)}\n\n'
    sorted_graphics = list(sorted(existing_graphics))
    logtext += '\n'.join(sorted_graphics)
    with logfile.open('w') as log:
        log.write(logtext)

def debug_converter_objects(debugdir: Directory, loglevel: int, dataset: GenieObjectContainer) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Create debug output for ConverterObject instances from the\n    conversion preprocessor.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param dataset: Dataset containing converter objects from pre-processing.\n    :type dataset: GenieObjectContainer\n    '
    if loglevel < 2:
        return
    logfile = debugdir.joinpath('conversion/')['preprocessor_objects']
    logtext = ''
    logtext += f'unit objects count: {len(dataset.genie_units)}\ntech objects count: {len(dataset.genie_techs)}\nciv objects count: {len(dataset.genie_civs)}\neffect bundles count: {len(dataset.genie_effect_bundles)}\nage connections count: {len(dataset.age_connections)}\nbuilding connections count: {len(dataset.building_connections)}\nunit connections count: {len(dataset.unit_connections)}\ntech connections count: {len(dataset.tech_connections)}\ngraphics objects count: {len(dataset.genie_graphics)}\nsound objects count: {len(dataset.genie_sounds)}\nterrain objects count: {len(dataset.genie_terrains)}\n'
    with logfile.open('w') as log:
        log.write(logtext)

def debug_converter_object_groups(debugdir: Directory, loglevel: int, dataset: GenieObjectContainer) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create debug output for ConverterObjectGroup instances from the\n    conversion preprocessor.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param dataset: Dataset containing converter object groups from processing.\n    :type dataset: GenieObjectContainer\n    '
    if loglevel < 3:
        return
    enitity_groups = {}
    enitity_groups.update(dataset.unit_lines)
    enitity_groups.update(dataset.building_lines)
    enitity_groups.update(dataset.ambient_groups)
    enitity_groups.update(dataset.variant_groups)
    entity_name_lookup_dict = get_entity_lookups(dataset.game_version)
    tech_name_lookup_dict = get_tech_lookups(dataset.game_version)
    civ_name_lookup_dict = get_civ_lookups(dataset.game_version)
    terrain_name_lookup_dict = get_terrain_lookups(dataset.game_version)
    nnn = ('NameNotFound', 'NameNotFound')
    for (key, line) in enitity_groups.items():
        logfile = debugdir.joinpath('conversion/entity_groups/')[str(key)]
        logtext = ''
        logtext += f'repr: {line}\n'
        logtext += f'nyan name: {entity_name_lookup_dict.get(line.get_head_unit_id(), nnn)[0]}\n'
        logtext += f'is_creatable: {line.is_creatable()}\n'
        logtext += f'is_harvestable: {line.is_harvestable()}\n'
        logtext += f'is_garrison: {line.is_garrison()}\n'
        logtext += f'is_gatherer: {line.is_gatherer()}\n'
        logtext += f'is_passable: {line.is_passable()}\n'
        logtext += f'is_projectile_shooter: {line.is_projectile_shooter()}\n'
        logtext += f'is_ranged: {line.is_ranged()}\n'
        logtext += f'is_melee: {line.is_melee()}\n'
        logtext += f'is_repairable: {line.is_repairable()}\n'
        logtext += f'is_unique: {line.is_unique()}\n'
        logtext += f'class id: {line.get_class_id()}\n'
        logtext += f'garrison mode: {line.get_garrison_mode()}\n'
        logtext += f'head unit: {line.get_head_unit()}\n'
        logtext += f'train location id: {line.get_train_location_id()}\n'
        logtext += 'line:\n'
        for unit in line.line:
            logtext += f'    - {unit}\n'
        if len(line.creates) > 0:
            logtext += 'creates:\n'
            for unit in line.creates:
                logtext += f'    - {unit} ({entity_name_lookup_dict.get(unit.get_head_unit_id(), nnn)[0]})\n'
        else:
            logtext += 'creates: nothing\n'
        if len(line.researches) > 0:
            logtext += 'researches:\n'
            for tech in line.researches:
                logtext += f'    - {tech} ({tech_name_lookup_dict.get(tech.get_id(), nnn)[0]})\n'
        else:
            logtext += 'researches: nothing\n'
        if len(line.garrison_entities) > 0:
            logtext += 'garrisons units:\n'
            for unit in line.garrison_entities:
                logtext += f'    - {unit} ({entity_name_lookup_dict.get(unit.get_head_unit_id(), nnn)[0]})\n'
        else:
            logtext += 'garrisons units: nothing\n'
        if len(line.garrison_locations) > 0:
            logtext += 'garrisons in:\n'
            for unit in line.garrison_locations:
                logtext += f'    - {unit} ({entity_name_lookup_dict.get(unit.get_head_unit_id(), nnn)[0]})\n'
        else:
            logtext += 'garrisons in: nothing\n'
        if isinstance(line, GenieUnitLineGroup):
            logtext += '\n'
            logtext += f'civ id: {line.get_civ_id()} ({civ_name_lookup_dict.get(line.get_civ_id(), nnn)[0]})\n'
            logtext += f'enabling research id: {line.get_enabling_research_id()} ({tech_name_lookup_dict.get(line.get_enabling_research_id(), nnn)[0]})\n'
        if isinstance(line, GenieBuildingLineGroup):
            logtext += '\n'
            logtext += f'has_foundation: {line.has_foundation()}\n'
            logtext += f'is_dropsite: {line.is_dropsite()}\n'
            logtext += f'is_trade_post {line.is_trade_post()}\n'
            logtext += f'enabling research id: {line.get_enabling_research_id()} ({tech_name_lookup_dict.get(line.get_enabling_research_id(), nnn)[0]})\n'
            logtext += f'dropoff gatherer ids: {line.get_gatherer_ids()}\n'
        if isinstance(line, GenieStackBuildingGroup):
            logtext += '\n'
            logtext += f'is_gate: {line.is_gate()}\n'
            logtext += f'stack unit: {line.get_stack_unit()}\n'
        if isinstance(line, GenieUnitTransformGroup):
            logtext += '\n'
            logtext += f'transform unit: {line.get_transform_unit()}\n'
        if isinstance(line, GenieMonkGroup):
            logtext += '\n'
            logtext += f'switch unit: {line.get_switch_unit()}\n'
        with logfile.open('w') as log:
            log.write(logtext)
    for (key, civ) in dataset.civ_groups.items():
        logfile = debugdir.joinpath('conversion/civ_groups/')[str(key)]
        logtext = ''
        logtext += f'repr: {civ}\n'
        logtext += f'nyan name: {civ_name_lookup_dict.get(civ.get_id(), nnn)[0]}\n'
        logtext += f'team bonus: {civ.team_bonus}\n'
        logtext += f'tech tree: {civ.tech_tree}\n'
        logtext += 'civ bonus ids:\n'
        for bonus in civ.civ_boni:
            logtext += f'    - {bonus}\n'
        logtext += 'unique unit ids:\n'
        for unit in civ.unique_entities:
            logtext += f'    - {unit} ({entity_name_lookup_dict.get(unit, nnn)[0]})\n'
        logtext += 'unique tech ids:\n'
        for tech in civ.unique_techs:
            logtext += f'    - {tech} ({tech_name_lookup_dict.get(tech, nnn)[0]})\n'
        with logfile.open('w') as log:
            log.write(logtext)
    for (key, tech) in dataset.tech_groups.items():
        logfile = debugdir.joinpath('conversion/tech_groups/')[str(key)]
        logtext = ''
        logtext += f'repr: {tech}\n'
        logtext += f'nyan name: {tech_name_lookup_dict.get(tech.get_id(), nnn)[0]}\n'
        logtext += f'is_researchable: {tech.is_researchable()}\n'
        logtext += f'is_unique: {tech.is_unique()}\n'
        logtext += f'research location id: {tech.get_research_location_id()} ({entity_name_lookup_dict.get(tech.get_research_location_id(), nnn)[0]})\n'
        logtext += f'required tech count: {tech.get_required_tech_count()}\n'
        logtext += 'required techs:\n'
        for req_tech in tech.get_required_techs():
            logtext += f'    - {req_tech} ({tech_name_lookup_dict.get(req_tech, nnn)[0]})\n'
        if isinstance(tech, AgeUpgrade):
            logtext += '\n'
            logtext += f'researched age id: {tech.age_id}\n'
        if isinstance(tech, UnitLineUpgrade):
            logtext += '\n'
            logtext += f'upgraded line id: {tech.get_line_id()}\n'
            logtext += f'upgraded line: {tech.get_upgraded_line()} ({entity_name_lookup_dict.get(tech.get_line_id(), nnn)[0]})\n'
            logtext += f'upgrade target id: {tech.get_upgrade_target_id()}\n'
        if isinstance(tech, BuildingLineUpgrade):
            logtext += '\n'
            logtext += f'upgraded line id: {tech.get_line_id()}\n'
            logtext += f'upgraded line: {tech.get_upgraded_line()} ({entity_name_lookup_dict.get(tech.get_line_id(), nnn)[0]})\n'
            logtext += f'upgrade target id: {tech.get_upgrade_target_id()}\n'
        if isinstance(tech, UnitUnlock):
            logtext += '\n'
            logtext += f'unlocked line: {tech.get_unlocked_line()} ({entity_name_lookup_dict.get(tech.get_unlocked_line().get_head_unit_id(), nnn)[0]})\n'
        if isinstance(tech, BuildingUnlock):
            logtext += '\n'
            logtext += f'unlocked line: {tech.get_unlocked_line()} ({entity_name_lookup_dict.get(tech.get_unlocked_line().get_head_unit_id(), nnn)[0]})\n'
        with logfile.open('w') as log:
            log.write(logtext)
    for (key, terrain) in dataset.terrain_groups.items():
        logfile = debugdir.joinpath('conversion/terrain_groups/')[str(key)]
        logtext = ''
        logtext += f'repr: {terrain}\n'
        logtext += f'nyan name: {terrain_name_lookup_dict.get(terrain.get_id(), nnn)[1]}\n'
        logtext += f'has_subterrain: {terrain.has_subterrain()}\n'
        with logfile.open('w') as log:
            log.write(logtext)

def debug_modpack(debugdir: Directory, loglevel: int, modpack: Modpack) -> None:
    if False:
        while True:
            i = 10
    '\n    Create debug output for a modpack.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param modpack: Modpack container.\n    :type modpack: Modpack\n    '
    if loglevel < 1:
        return
    logdir = debugdir.joinpath(f'export/{modpack.name}')
    with logdir[modpack.info.filename].open('wb') as outfile:
        outfile.write(modpack.info.dump().encode('utf-8'))
    with logdir[modpack.manifest.filename].open('wb') as outfile:
        outfile.write(modpack.manifest.dump().encode('utf-8'))
    if loglevel < 2:
        return
    logfile = debugdir.joinpath(f'export/{modpack.name}')['summary']
    logtext = ''
    logtext += f'name: {modpack.name}\n'
    file_count = len(modpack.get_data_files()) + len(modpack.get_media_files()) + len(modpack.get_metadata_files())
    logtext += f'file count: {file_count}\n'
    logtext += f'    data: {len(modpack.get_data_files())}\n'
    logtext += f'    media: {len(modpack.get_media_files())}\n'
    media_dict = {}
    for (media_type, files) in modpack.get_media_files().items():
        media_dict[media_type.value] = len(files)
    media_dict = dict(sorted(media_dict.items(), key=lambda item: item[0]))
    for (media_type, file_count) in media_dict.items():
        logtext += f'        {media_type}: {file_count}\n'
    logtext += f'    metadata: {len(modpack.get_metadata_files())}\n'
    with logfile.open('w') as log:
        log.write(logtext)

def debug_media_cache(debugdir: Directory, loglevel: int, sourcedir: Directory, cachedata: dict, game_version: GameVersion) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create media cache data for graphics files. This allows using deterministic\n    packer and compression settings for graphics file conversion.\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param sourcedir: Sourcedir where the graphics files are mounted.\n    :type sourcedir: Directory\n    :param cachedata: Dict with cache data.\n    :type cachedata: dict\n    :param game_version: Game version.\n    :type game_version: GameVersion\n    '
    if loglevel < 6:
        return
    cache_file = MediaCacheFile('export/', 'media_cache.toml', game_version)
    cache_file.set_hash_func('sha3_256')
    cache_data = dict(sorted(cachedata.items(), key=lambda item: item[0].source_filename))
    for (request, cache) in cache_data.items():
        filepath = sourcedir[request.get_type().value, request.source_filename]
        cache_file.add_cache_data(request.get_type(), request.source_filename, hash_file(filepath), cache[1], cache[0])
    logfile = debugdir.joinpath('export/')['media_cache.toml']
    logtext = cache_file.dump()
    with logfile.open('w') as log:
        log.write(logtext)

def debug_execution_time(debugdir: Directory, loglevel: int, stages_time: dict[str, float]) -> None:
    if False:
        return 10
    '\n    Create debug output for execution time for each stage\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param stages_time: Dict with execution time for each stage.\n    :type stages_time: dict\n    '
    if loglevel < 1:
        return
    logfile = debugdir['execution_time']
    logtext = ''.join((f'{k}: {v}\n' for (k, v) in stages_time.items()))
    with logfile.open('w') as log:
        log.write(logtext)

def debug_not_found_sounds(debugdir: Directory, loglevel: int, sound: Path) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create debug output for sounds not found\n\n    :param debugdir: Output directory for the debug info.\n    :type debugdir: Directory\n    :param loglevel: Determines how detailed the output is.\n    :type loglevel: int\n    :param sound: Sound object with path and name values.\n    :type sound: Path\n    '
    if loglevel < 6:
        return
    logfile = debugdir.joinpath('export/not_found_sounds')[sound.stem]
    path = [part.decode() for part in sound.parts]
    logtext = f"name: {sound.name}\npath: {'/'.join(path)}"
    with logfile.open('w') as log:
        log.write(logtext)