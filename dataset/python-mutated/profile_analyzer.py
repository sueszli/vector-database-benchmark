import collections
import copy
import functools
from typing import Callable, List, Optional, Union
import numpy as np

class NonExistNum:
    """An object that behaves like a number but means a field does not exist; It is
    always greater than any real number.
    """

    def __truediv__(self, _):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __add__(self, rhs):
        if False:
            for i in range(10):
                print('nop')
        return rhs

    def __radd__(self, lhs):
        if False:
            return 10
        return lhs

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __gt__(self, rhs):
        if False:
            print('Hello World!')
        if isinstance(rhs) is NonExistNum:
            return id(self) > id(rhs)
        return True

    def __ge__(self, rhs):
        if False:
            i = 10
            return i + 15
        return self > rhs or self == rhs

    def __lt__(self, rhs):
        if False:
            return 10
        if isinstance(rhs) is NonExistNum:
            return id(self) < id(rhs)
        return False

    def __le__(self, rhs):
        if False:
            for i in range(10):
                print('nop')
        return self < rhs or self == rhs

    def __eq__(self, rhs):
        if False:
            return 10
        return self is rhs

    def __format__(self, spec):
        if False:
            i = 10
            return i + 15
        return 'N/A'

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'N/A'

class OprProfRst:
    """Opr profiling result dumped from megengine profiler.

    Args:
        entry: profiling json exec_graph items. Opr profiling initialization, 
            which sets up name, type and id of opr_info.
    """
    opr_info = None
    'A dict containing operator info:  name, id and type.'
    time_dict = None
    '\n    A mapping from ``"host"`` or ``"device"`` to list of profiling\n    results.'
    footprint = None
    '\n    A mapping from ``"memory"`` or ``"computation"`` to the actual number\n    of corresponding operations.'

    def __init__(self, entry: dict):
        if False:
            return 10
        assert isinstance(entry, dict)
        self.opr_info = collections.OrderedDict()
        for key in ['name', 'type', 'id']:
            self.opr_info[key] = entry[key]
        self.time_dict = collections.defaultdict(list)
        self.footprint = collections.defaultdict(NonExistNum)

    def update_device_prof_info(self, dev_time: dict):
        if False:
            while True:
                i = 10
        'Updates device profiling info.\n\n        Args:\n            dev_time: device time for single opr,\n                is an attribute of profiling result.\n        '
        assert isinstance(dev_time, dict)
        self.time_dict['device'].append(copy.deepcopy(dev_time))

    def update_host_prof_info(self, host_time: dict):
        if False:
            while True:
                i = 10
        'Updates host profiling info.\n\n        Args:\n            host_time: host time for single opr,\n                is an attribute of profiling result.\n        '
        assert isinstance(host_time, dict)
        self.time_dict['host'].append(copy.deepcopy(host_time))

    def update_footprint(self, footprint: dict):
        if False:
            while True:
                i = 10
        'Updates opr footprint.\n\n        Args:\n            footprint: footprint for single opr,\n                is an attribute of profiling result.\n        '
        assert isinstance(footprint, dict)
        self.footprint.update(footprint)

class Record:
    """A record of analyzing result

    Args:
        time: opr running time, evaluated by applying users providing
            function to OprProfRst.
        info: opr information, could be original opr information or
            aggregate infomation if aggregating enabled.
        footprint: contains footprint information, for now, we have
            ``"computation"``, ``"memory"``, ``"in_shapes"``, ``"out_shapes"``.
    """
    __slot__ = ['time', 'info', 'computation', 'memory', 'in_shapes', 'in_layouts', 'out_shapes', 'flops', 'bandwidth', 'opr_id']

    def __init__(self, time: float, info: dict, footprint: dict):
        if False:
            while True:
                i = 10
        assert isinstance(footprint, dict)
        self.time = time
        self.info = collections.OrderedDict(copy.deepcopy(info))
        self.computation = footprint['computation'] or NonExistNum()
        self.memory = footprint['memory']
        self.in_shapes = footprint['in_shapes']
        self.in_layouts = footprint.get('in_layouts')
        self.out_shapes = footprint['out_shapes']
        self.flops = self.computation / self.time
        self.bandwidth = self.memory / self.time
        self.opr_id = info.get('id')
        if isinstance(self.opr_id, str) and self.opr_id != 'N/A':
            self.opr_id = int(self.opr_id)

    def get_column_by_name(self, name: str=None):
        if False:
            while True:
                i = 10
        'Extracts column value by its column name.\n\n        Args:\n            name: column name, None for time.\n        '
        if name is None:
            name = 'time'
        return getattr(self, name)

class ProfileAnalyzer:
    """Initializes ProfileAnalyzer.

    Args:
        obj: dict dumped from json str.
        opr_filter: function that filter oprs.
    """

    def __init__(self, obj: dict, opr_filter: Callable=lambda opr, inp, out: True):
        if False:
            i = 10
            return i + 15
        self._opr_set = dict()
        assert isinstance(obj, dict), type(obj)
        varz = obj['graph_exec']['var']
        for (opr_id, entry) in obj['graph_exec']['operator'].items():
            inp = [varz[i] for i in entry['input']]
            out = [varz[i] for i in entry['output']]
            if opr_filter(entry, inp, out):
                self._opr_set[opr_id] = OprProfRst(entry)
        for (opr_id, entry) in obj['profiler']['device'].items():
            if opr_id not in self._opr_set:
                continue
            opr = self._opr_set[opr_id]
            for (_, time) in entry.items():
                opr.update_device_prof_info(time)
        for (opr_id, entry) in obj['profiler']['host'].items():
            if opr_id not in self._opr_set:
                continue
            opr = self._opr_set[opr_id]
            for (_, time) in entry.items():
                opr.update_host_prof_info(time)
        for (opr_id, entry) in obj['profiler'].get('opr_footprint', {}).items():
            if opr_id not in self._opr_set:
                continue
            opr = self._opr_set[opr_id]
            opr.update_footprint(entry)

    def _aggregate(self, records: List[Record], aop: Union[str, Callable], atype: Optional[str]) -> List[Record]:
        if False:
            while True:
                i = 10
        'Aggregate operation.\n\n        Args:\n            records: selected records.\n            aop: aggregate operation, if aop is str, we would replace it\n                with associated numpy function wth aop name".\n            atype: the type aggregated by, None for aggregating all into single\n                record.\n        '
        if aop is None:
            assert atype is None, 'must specify aggregate op'
            return records
        if isinstance(aop, str):
            aop = getattr(np, aop)
        type2stat = collections.defaultdict(lambda : [[], [], []])
        for item in records:
            if atype == 'type':
                d = type2stat[item.info['type']]
            else:
                d = type2stat['all']
            d[0].append(item.time)
            d[1].append(item.computation)
            d[2].append(item.memory)
        rst = []
        for opr_type in type2stat.keys():
            (time, computation, memory) = type2stat[opr_type]
            nr_oprs = len(time)
            time_rst = aop(time)
            comp_rst = aop(computation)
            mem_rst = aop(memory)
            item = Record(time_rst, {'type': opr_type, 'count': nr_oprs, 'id': 'N/A'}, {'computation': comp_rst, 'memory': mem_rst, 'in_shapes': None, 'out_shapes': None})
            rst.append(item)
        return rst

    def _sort(self, records: List[Record], sort_by: str) -> List[Record]:
        if False:
            for i in range(10):
                print('nop')
        'Sort operation.\n\n        Args:\n            records: the records after aggregate operation.\n            sort_by: keyword for sorting the list.\n        '
        if sort_by is None:
            return records
        if sort_by.startswith('+'):
            sort_by = sort_by[1:]
            key = lambda record: record.get_column_by_name(sort_by)
        else:
            key = lambda record: -record.get_column_by_name(sort_by)
        records.sort(key=key)
        return records

    def select(self, time_func: Callable, opr_filter: Callable=lambda opr: True, aggregate: Callable=None, aggregate_by: str=None, sort_by: str=None, top_k: int=0) -> List[Record]:
        if False:
            return 10
        'Select operation.\n\n        Args:\n            time_func: time_func provided by user, would apply to every\n                OprProfRst.\n            opr_filter: filter satisfied operatiors.\n            aggregate: function that apply to list of records which are\n                aggregated by atype.\n            aggregate_by: the type aggregated by.\n            sort_by: keyword for sorting all records.\n            top_k: specify the maximum number of records.\n\n        Returns:\n            the records that go through select, aggregate, sort.\n        '
        records = []
        for opr in self._opr_set.values():
            if opr_filter(opr):
                time = time_func(opr)
                if time is None:
                    continue
                item = Record(time, opr.opr_info, opr.footprint)
                records.append(item)
        records = self._aggregate(records, aggregate, aggregate_by)
        if not records:
            return records
        return self._sort(records, sort_by)[0:len(records) if top_k == 0 else top_k]

class TimeFuncHelper:
    """Time Function Helper for users."""

    @staticmethod
    def _eval_time(prof_type, end_key, func, opr_prof):
        if False:
            i = 10
            return i + 15
        "Eval time.\n\n        Args:\n             prof_type: host' or 'device'.\n            end_key: kern' or 'end'.\n            func: apply to list of all ``thread`` of ``gpu`` time.\n            opr_prof: operator profiling result.\n\n        Returns:\n            time.\n        "
        if prof_type not in opr_prof.time_dict:
            return None
        time = [time[end_key] - time['start'] for time in opr_prof.time_dict[prof_type]]
        return func(time)

    @staticmethod
    def eval_time_func(prof_type: str, end_key: str, func: Callable) -> float:
        if False:
            i = 10
            return i + 15
        "Eval oprerator profile time.\n\n        Args:\n            prof_type: host' or 'device'.\n            end_key: kern' or 'end'.\n            func: apply to list of all ``thread`` of ``gpu`` time.\n\n        Returns:\n            eval time results.\n        "
        return functools.partial(TimeFuncHelper._eval_time, prof_type, end_key, func)

    @staticmethod
    def _min_start(prof_type, end_key, func, opr_prof):
        if False:
            print('Hello World!')
        "Eval minimum start time.\n\n        Args:\n            prof_type(str): 'host' or 'device'.\n            end_key(str): 'kern' or 'end'.\n            func(function): apply to list of all ``thread`` of ``gpu`` time.\n            opr_prof(OprProfRst): operator profiling result.\n        \n        Returns:\n            time.\n        "
        if prof_type not in opr_prof.time_dict:
            return None
        time = [time['start'] for time in opr_prof.time_dict[prof_type]]
        return np.min(time)

    @staticmethod
    def min_start_func(prof_type: str, end_key: str, func: Callable) -> float:
        if False:
            return 10
        "Eval oprerator profile min start time.\n\n        Args:\n            prof_type(str): 'host' or 'device'.\n            end_key(str): 'kern' or 'end'.\n            func(function): apply to list of all ``thread`` of ``gpu`` time.\n\n        Returns:\n            eval time results.\n        "
        return functools.partial(TimeFuncHelper._min_start, prof_type, end_key, func)

    @staticmethod
    def _max_end(prof_type, end_key, func, opr_prof):
        if False:
            while True:
                i = 10
        "Eval maximum end time\n\n        Args:\n            prof_type(str): 'host' or 'device'.\n            end_key(str): 'kern' or 'end'.\n            func(function): apply to list of all ``thread`` of ``gpu`` time.\n            opr_prof(OprProfRst): operator profiling result.\n        \n        Returns:\n            time.\n        "
        if prof_type not in opr_prof.time_dict:
            return None
        time = [time['end'] for time in opr_prof.time_dict[prof_type]]
        return np.max(time)

    @staticmethod
    def max_end_func(prof_type: str, end_key: str, func: Callable) -> float:
        if False:
            print('Hello World!')
        "Eval oprerator profile max end time.\n\n        Args:\n            prof_type(str): 'host' or 'device'.\n            end_key(str): 'kern' or 'end'.\n            func(function): apply to list of all ``thread`` of ``gpu`` time.\n\n        Returns:\n            eval time results.\n        "
        return functools.partial(TimeFuncHelper._max_end, prof_type, end_key, func)