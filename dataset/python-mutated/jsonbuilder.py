import json
import os.path
from robot.running import ArgInfo, TypeInfo
from robot.errors import DataError
from .datatypes import EnumMember, TypedDictItem, TypeDoc
from .model import LibraryDoc, KeywordDoc

class JsonDocBuilder:

    def build(self, path):
        if False:
            return 10
        spec = self._parse_spec_json(path)
        return self.build_from_dict(spec)

    def build_from_dict(self, spec):
        if False:
            return 10
        libdoc = LibraryDoc(name=spec['name'], doc=spec['doc'], version=spec['version'], type=spec['type'], scope=spec['scope'], doc_format=spec['docFormat'], source=spec['source'], lineno=int(spec.get('lineno', -1)))
        libdoc.inits = [self._create_keyword(kw) for kw in spec['inits']]
        libdoc.keywords = [self._create_keyword(kw) for kw in spec['keywords']]
        if 'typedocs' in spec:
            libdoc.type_docs = self._parse_type_docs(spec['typedocs'])
        elif 'dataTypes' in spec:
            libdoc.type_docs = self._parse_data_types(spec['dataTypes'])
        return libdoc

    def _parse_spec_json(self, path):
        if False:
            while True:
                i = 10
        if not os.path.isfile(path):
            raise DataError(f"Spec file '{path}' does not exist.")
        with open(path) as json_source:
            libdoc_dict = json.load(json_source)
        return libdoc_dict

    def _create_keyword(self, data):
        if False:
            while True:
                i = 10
        kw = KeywordDoc(name=data.get('name'), doc=data['doc'], short_doc=data['shortdoc'], tags=data['tags'], private=data.get('private', False), deprecated=data.get('deprecated', False), source=data['source'], lineno=int(data.get('lineno', -1)))
        self._create_arguments(data['args'], kw)
        self._add_return_type(data.get('returnType'), kw)
        return kw

    def _create_arguments(self, arguments, kw: KeywordDoc):
        if False:
            print('Hello World!')
        spec = kw.args
        setters = {ArgInfo.POSITIONAL_ONLY: spec.positional_only.append, ArgInfo.POSITIONAL_ONLY_MARKER: lambda value: None, ArgInfo.POSITIONAL_OR_NAMED: spec.positional_or_named.append, ArgInfo.VAR_POSITIONAL: lambda value: setattr(spec, 'var_positional', value), ArgInfo.NAMED_ONLY_MARKER: lambda value: None, ArgInfo.NAMED_ONLY: spec.named_only.append, ArgInfo.VAR_NAMED: lambda value: setattr(spec, 'var_named', value)}
        for arg in arguments:
            name = arg['name']
            setters[arg['kind']](name)
            default = arg.get('defaultValue')
            if default is not None:
                spec.defaults[name] = default
            if 'type' in arg:
                type_docs = {}
                type_info = self._parse_type_info(arg['type'], type_docs)
            else:
                type_docs = arg.get('typedocs', {})
                type_info = self._parse_legacy_type_info(arg['types'])
            if type_info:
                if not spec.types:
                    spec.types = {}
                spec.types[name] = type_info
            kw.type_docs[name] = type_docs

    def _parse_type_info(self, data, type_docs):
        if False:
            return 10
        if not data:
            return None
        if data.get('typedoc'):
            type_docs[data['name']] = data['typedoc']
        nested = [self._parse_type_info(typ, type_docs) for typ in data.get('nested', ())]
        return TypeInfo(data['name'], nested=nested)

    def _parse_legacy_type_info(self, types):
        if False:
            while True:
                i = 10
        return TypeInfo.from_sequence(types) if types else None

    def _add_return_type(self, data, kw: KeywordDoc):
        if False:
            while True:
                i = 10
        if data:
            type_docs = {}
            kw.args.return_type = self._parse_type_info(data, type_docs)
            kw.type_docs['return'] = type_docs

    def _parse_type_docs(self, type_docs):
        if False:
            for i in range(10):
                print('nop')
        for data in type_docs:
            doc = TypeDoc(data['type'], data['name'], data['doc'], data['accepts'], data['usages'])
            if doc.type == TypeDoc.ENUM:
                doc.members = [EnumMember(d['name'], d['value']) for d in data['members']]
            if doc.type == TypeDoc.TYPED_DICT:
                doc.items = [TypedDictItem(d['key'], d['type'], d['required']) for d in data['items']]
            yield doc

    def _parse_data_types(self, data_types):
        if False:
            i = 10
            return i + 15
        for obj in data_types['enums']:
            yield self._create_enum_doc(obj)
        for obj in data_types['typedDicts']:
            yield self._create_typed_dict_doc(obj)

    def _create_enum_doc(self, data):
        if False:
            return 10
        return TypeDoc(TypeDoc.ENUM, data['name'], data['doc'], members=[EnumMember(member['name'], member['value']) for member in data['members']])

    def _create_typed_dict_doc(self, data):
        if False:
            print('Hello World!')
        return TypeDoc(TypeDoc.TYPED_DICT, data['name'], data['doc'], items=[TypedDictItem(item['key'], item['type'], item['required']) for item in data['items']])