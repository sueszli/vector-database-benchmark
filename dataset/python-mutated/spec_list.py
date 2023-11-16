import itertools
from typing import List
import spack.variant
from spack.error import SpackError
from spack.spec import Spec

class SpecList:

    def __init__(self, name='specs', yaml_list=None, reference=None):
        if False:
            while True:
                i = 10
        yaml_list = yaml_list or []
        reference = reference or {}
        self.name = name
        self._reference = reference
        if not all((isinstance(s, str) or isinstance(s, (list, dict)) for s in yaml_list)):
            raise ValueError('yaml_list can contain only valid YAML types!  Found:\n  %s' % [type(s) for s in yaml_list])
        self.yaml_list = yaml_list[:]
        self._expanded_list = None
        self._constraints = None
        self._specs = None

    @property
    def is_matrix(self):
        if False:
            while True:
                i = 10
        for item in self.specs_as_yaml_list:
            if isinstance(item, dict):
                return True
        return False

    @property
    def specs_as_yaml_list(self):
        if False:
            i = 10
            return i + 15
        if self._expanded_list is None:
            self._expanded_list = self._expand_references(self.yaml_list)
        return self._expanded_list

    @property
    def specs_as_constraints(self):
        if False:
            while True:
                i = 10
        if self._constraints is None:
            constraints = []
            for item in self.specs_as_yaml_list:
                if isinstance(item, dict):
                    constraints.extend(_expand_matrix_constraints(item))
                else:
                    constraints.append([Spec(item)])
            self._constraints = constraints
        return self._constraints

    @property
    def specs(self) -> List[Spec]:
        if False:
            while True:
                i = 10
        if self._specs is None:
            specs = []
            for constraint_list in self.specs_as_constraints:
                spec = constraint_list[0].copy()
                for const in constraint_list[1:]:
                    spec.constrain(const)
                specs.append(spec)
            self._specs = specs
        return self._specs

    def add(self, spec):
        if False:
            i = 10
            return i + 15
        self.yaml_list.append(str(spec))
        if self._expanded_list is not None:
            self._expanded_list.append(str(spec))
        self._constraints = None
        self._specs = None

    def remove(self, spec):
        if False:
            while True:
                i = 10
        remove = [s for s in self.yaml_list if (isinstance(s, str) and (not s.startswith('$'))) and Spec(s) == Spec(spec)]
        if not remove:
            msg = f'Cannot remove {spec} from SpecList {self.name}.\n'
            msg += f'Either {spec} is not in {self.name} or {spec} is '
            msg += 'expanded from a matrix and cannot be removed directly.'
            raise SpecListError(msg)
        for item in remove:
            self.yaml_list.remove(item)
        self._expanded_list = None
        self._constraints = None
        self._specs = None

    def extend(self, other, copy_reference=True):
        if False:
            for i in range(10):
                print('nop')
        self.yaml_list.extend(other.yaml_list)
        self._expanded_list = None
        self._constraints = None
        self._specs = None
        if copy_reference:
            self._reference = other._reference

    def update_reference(self, reference):
        if False:
            i = 10
            return i + 15
        self._reference = reference
        self._expanded_list = None
        self._constraints = None
        self._specs = None

    def _parse_reference(self, name):
        if False:
            for i in range(10):
                print('nop')
        sigil = ''
        name = name[1:]
        if name.startswith('^') or name.startswith('%'):
            sigil = name[0]
            name = name[1:]
        if name not in self._reference:
            msg = f"SpecList '{self.name}' refers to named list '{name}'"
            msg += ' which does not appear in its reference dict.'
            raise UndefinedReferenceError(msg)
        return (name, sigil)

    def _expand_references(self, yaml):
        if False:
            while True:
                i = 10
        if isinstance(yaml, list):
            ret = []
            for item in yaml:
                if isinstance(item, str) and item.startswith('$'):
                    (name, sigil) = self._parse_reference(item)
                    referent = [_sigilify(item, sigil) for item in self._reference[name].specs_as_yaml_list]
                    ret.extend(referent)
                else:
                    ret.append(self._expand_references(item))
            return ret
        elif isinstance(yaml, dict):
            return dict(((name, self._expand_references(val)) for (name, val) in yaml.items()))
        else:
            return yaml

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.specs)

    def __getitem__(self, key):
        if False:
            return 10
        return self.specs[key]

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.specs)

def _expand_matrix_constraints(matrix_config):
    if False:
        for i in range(10):
            print('nop')
    expanded_rows = []
    for row in matrix_config['matrix']:
        new_row = []
        for r in row:
            if isinstance(r, dict):
                new_row.extend([[' '.join([str(c) for c in expanded_constraint_list])] for expanded_constraint_list in _expand_matrix_constraints(r)])
            else:
                new_row.append([r])
        expanded_rows.append(new_row)
    excludes = matrix_config.get('exclude', [])
    sigil = matrix_config.get('sigil', '')
    results = []
    for combo in itertools.product(*expanded_rows):
        flat_combo = [constraint for constraint_list in combo for constraint in constraint_list]
        flat_combo = [Spec(x).lookup_hash() for x in flat_combo]
        test_spec = flat_combo[0].copy()
        for constraint in flat_combo[1:]:
            test_spec.constrain(constraint)
        try:
            spack.variant.substitute_abstract_variants(test_spec)
        except spack.variant.UnknownVariantError:
            pass
        if any((test_spec.satisfies(x) for x in excludes)):
            continue
        if sigil:
            flat_combo[0] = Spec(sigil + str(flat_combo[0]))
        results.append(flat_combo)
    return results

def _sigilify(item, sigil):
    if False:
        while True:
            i = 10
    if isinstance(item, dict):
        if sigil:
            item['sigil'] = sigil
        return item
    else:
        return sigil + item

class SpecListError(SpackError):
    """Error class for all errors related to SpecList objects."""

class UndefinedReferenceError(SpecListError):
    """Error class for undefined references in Spack stacks."""

class InvalidSpecConstraintError(SpecListError):
    """Error class for invalid spec constraints at concretize time."""