"""Generate the C++ glue code needed to map EXPRESS to C++"""
import sys, os, re
if sys.version_info < (3, 0):
    print('must use python 3.0 or greater')
    sys.exit(-2)
use_ifc_template = False
input_step_template_h = 'StepReaderGen.h.template'
input_step_template_cpp = 'StepReaderGen.cpp.template'
input_ifc_template_h = 'IFCReaderGen.h.template'
input_ifc_template_cpp = 'IFCReaderGen.cpp.template'
cpp_keywords = 'class'
output_file_h = ''
output_file_cpp = ''
if use_ifc_template:
    input_template_h = input_ifc_template_h
    input_template_cpp = input_ifc_template_cpp
    output_file_h = os.path.join('..', '..', 'code', 'IFCReaderGen.h')
    output_file_cpp = os.path.join('..', '..', 'code', 'IFCReaderGen.cpp')
else:
    input_template_h = input_step_template_h
    input_template_cpp = input_step_template_cpp
    output_file_h = os.path.join('..', '..', 'code/Importer/StepFile', 'StepReaderGen.h')
    output_file_cpp = os.path.join('..', '..', 'code/Importer/StepFile', 'StepReaderGen.cpp')
template_entity_predef = '\tstruct {entity};\n'
template_entity_predef_ni = '\ttypedef NotImplemented {entity}; // (not currently used by Assimp)\n'
template_entity = '\n\n    // C++ wrapper for {entity}\n    struct {entity} : {parent} ObjectHelper<{entity},{argcnt}> {{ {entity}() : Object("{entity}") {{}}\n{fields}\n    }};'
template_entity_ni = ''
template_type = '\n    // C++ wrapper type for {type}\n    typedef {real_type} {type};'
template_stub_decl = '\tDECL_CONV_STUB({type});\n'
template_schema = '\t\tSchemaEntry("{normalized_name}",&STEP::ObjectHelper<{type},{argcnt}>::Construct )\n'
template_schema_type = '\t\tSchemaEntry("{normalized_name}",nullptr )\n'
template_converter = '\n// -----------------------------------------------------------------------------------------------------------\ntemplate <> size_t GenericFill<{type}>(const DB& db, const LIST& params, {type}* in)\n{{\n{contents}\n}}'
template_converter_prologue_a = '\tsize_t base = GenericFill(db,params,static_cast<{parent}*>(in));\n'
template_converter_prologue_b = '\tsize_t base = 0;\n'
template_converter_check_argcnt = '\tif (params.GetSize() < {max_arg}) {{ throw STEP::TypeError("expected {max_arg} arguments to {name}"); }}'
template_converter_code_per_field = "    do {{ // convert the '{fieldname}' argument\n        std::shared_ptr<const DataType> arg = params[base++];{handle_unset}{convert}\n    }} while(0);\n"
template_allow_optional = '\n        if (dynamic_cast<const UNSET*>(&*arg)) break;'
template_allow_derived = '\n        if (dynamic_cast<const ISDERIVED*>(&*arg)) {{ in->ObjectHelper<Assimp::IFC::{type},{argcnt}>::aux_is_derived[{argnum}]=true; break; }}'
template_convert_single = '\n        try {{ GenericConvert( in->{name}, arg, db ); break; }} \n        catch (const TypeError& t) {{ throw TypeError(t.what() + std::string(" - expected argument {argnum} to {classname} to be a `{full_type}`")); }}'
template_converter_omitted = '// this data structure is not used yet, so there is no code generated to fill its members\n'
template_converter_epilogue = '\treturn base;'
import ExpressReader

def get_list_bounds(collection_spec):
    if False:
        for i in range(10):
            print('nop')
    (start, end) = [int(n) if n != '?' else 0 for n in re.findall('(\\d+|\\?)', collection_spec)]
    return (start, end)

def get_cpp_type(field, schema):
    if False:
        for i in range(10):
            print('nop')
    isobjref = field.type in schema.entities
    base = field.type
    if isobjref:
        base = 'Lazy< ' + (base if base in schema.whitelist else 'NotImplemented') + ' >'
    if field.collection:
        (start, end) = get_list_bounds(field.collection)
        base = 'ListOf< {0}, {1}, {2} >'.format(base, start, end)
    if not isobjref:
        base += '::Out'
    if field.optional:
        base = 'Maybe< ' + base + ' >'
    return base

def generate_fields(entity, schema):
    if False:
        print('Hello World!')
    fields = []
    for e in entity.members:
        fields.append('\t\t{type} {name};'.format(type=get_cpp_type(e, schema), name=e.name))
    return '\n'.join(fields)

def handle_unset_args(field, entity, schema, argnum):
    if False:
        i = 10
        return i + 15
    n = ''
    if any((entity.name == e.parent for e in schema.entities.values())):
        n += template_allow_derived.format(type=entity.name, argcnt=len(entity.members), argnum=argnum)
    if not field.optional:
        return n + ''
    return n + template_allow_optional.format()

def get_single_conversion(field, schema, argnum=0, classname='?'):
    if False:
        for i in range(10):
            print('nop')
    name = field.name
    return template_convert_single.format(name=name, argnum=argnum, classname=classname, full_type=field.fullspec)

def count_args_up(entity, schema):
    if False:
        while True:
            i = 10
    return len(entity.members) + (count_args_up(schema.entities[entity.parent], schema) if entity.parent else 0)

def resolve_base_type(base, schema):
    if False:
        i = 10
        return i + 15
    if base in ('INTEGER', 'REAL', 'STRING', 'ENUMERATION', 'BOOLEAN', 'NUMBER', 'SELECT', 'LOGICAL'):
        return base
    if base in schema.types:
        return resolve_base_type(schema.types[base].equals, schema)
    print(base)
    return None

def gen_type_struct(typen, schema):
    if False:
        return 10
    base = resolve_base_type(typen.equals, schema)
    if not base:
        return ''
    if typen.aggregate:
        (start, end) = get_list_bounds(typen.aggregate)
        base = 'ListOf< {0}, {1}, {2} >'.format(base, start, end)
    return template_type.format(type=typen.name, real_type=base)

def gen_converter(entity, schema):
    if False:
        return 10
    max_arg = count_args_up(entity, schema)
    arg_idx = arg_idx_ofs = max_arg - len(entity.members)
    code = template_converter_prologue_a.format(parent=entity.parent) if entity.parent else template_converter_prologue_b
    if entity.name in schema.blacklist_partial:
        return code + template_converter_omitted + template_converter_epilogue
    if max_arg > 0:
        code += template_converter_check_argcnt.format(max_arg=max_arg, name=entity.name)
    for field in entity.members:
        code += template_converter_code_per_field.format(fieldname=field.name, handle_unset=handle_unset_args(field, entity, schema, arg_idx - arg_idx_ofs), convert=get_single_conversion(field, schema, arg_idx, entity.name))
        arg_idx += 1
    return code + template_converter_epilogue

def get_base_classes(e, schema):
    if False:
        i = 10
        return i + 15

    def addit(e, out):
        if False:
            i = 10
            return i + 15
        if e.parent:
            out.append(e.parent)
            addit(schema.entities[e.parent], out)
    res = []
    addit(e, res)
    return list(reversed(res))

def get_derived(e, schema):
    if False:
        i = 10
        return i + 15

    def get_deriv(e, out):
        if False:
            print('Hello World!')
        s = [ee for ee in schema.entities.values() if ee.parent == e.name]
        for sel in s:
            out.append(sel.name)
            get_deriv(sel, out)
    res = []
    get_deriv(e, res)
    return res

def get_hierarchy(e, schema):
    if False:
        print('Hello World!')
    return get_derived(e, schema) + [e.name] + get_base_classes(e, schema)

def sort_entity_list(schema):
    if False:
        while True:
            i = 10
    deps = []
    entities = schema.entities
    for e in entities.values():
        deps += get_base_classes(e, schema) + [e.name]
    checked = []
    for e in deps:
        if e not in checked:
            checked.append(e)
    return [entities[e] for e in checked]

def work(filename):
    if False:
        i = 10
        return i + 15
    schema = ExpressReader.read(filename, silent=True)
    (entities, stub_decls, schema_table, converters, typedefs, predefs) = ('', '', [], '', '', '')
    entitylist = 'ifc_entitylist.txt'
    if not use_ifc_template:
        entitylist = 'step_entitylist.txt'
    whitelist = []
    with open(entitylist, 'rt') as inp:
        whitelist = [n.strip() for n in inp.read().split('\n') if n[:1] != '#' and n.strip()]
    schema.whitelist = set()
    schema.blacklist_partial = set()
    for ename in whitelist:
        try:
            e = schema.entities[ename]
        except KeyError:
            continue
        for base in [e.name] + get_base_classes(e, schema):
            schema.whitelist.add(base)
        for base in get_derived(e, schema):
            schema.blacklist_partial.add(base)
    schema.blacklist_partial -= schema.whitelist
    schema.whitelist |= schema.blacklist_partial
    cpp_types = cpp_keywords.split(',')
    for ntype in schema.types.values():
        typedefs += gen_type_struct(ntype, schema)
        schema_table.append(template_schema_type.format(normalized_name=ntype.name.lower()))
    sorted_entities = sort_entity_list(schema)
    for entity in sorted_entities:
        parent = entity.parent + ',' if entity.parent else ''
        if entity.name in cpp_types:
            entity.name = entity.name + '_t'
            print('renaming ' + entity.name)
        if entity.name in schema.whitelist:
            converters += template_converter.format(type=entity.name, contents=gen_converter(entity, schema))
            schema_table.append(template_schema.format(type=entity.name, normalized_name=entity.name.lower(), argcnt=len(entity.members)))
            entities += template_entity.format(entity=entity.name, argcnt=len(entity.members), parent=parent, fields=generate_fields(entity, schema))
            predefs += template_entity_predef.format(entity=entity.name)
            stub_decls += template_stub_decl.format(type=entity.name)
        else:
            entities += template_entity_ni.format(entity=entity.name)
            predefs += template_entity_predef_ni.format(entity=entity.name)
            schema_table.append(template_schema.format(type='NotImplemented', normalized_name=entity.name.lower(), argcnt=0))
    schema_table = ','.join(schema_table)
    with open(input_template_h, 'rt') as inp:
        with open(output_file_h, 'wt') as outp:
            outp.write(inp.read().replace('{predefs}', predefs).replace('{types}', typedefs).replace('{entities}', entities).replace('{converter-decl}', stub_decls))
    with open(input_template_cpp, 'rt') as inp:
        with open(output_file_cpp, 'wt') as outp:
            outp.write(inp.read().replace('{schema-static-table}', schema_table).replace('{converter-impl}', converters))
    return 0
if __name__ == '__main__':
    sys.exit(work(sys.argv[1] if len(sys.argv) > 1 else 'schema.exp'))