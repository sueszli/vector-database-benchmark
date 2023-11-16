from __future__ import annotations
from airflow.utils.operator_resources import Resources

class TestResources:

    def test_resource_eq(self):
        if False:
            for i in range(10):
                print('nop')
        r = Resources(cpus=0.1, ram=2048)
        assert r not in [{}, [], None]
        assert r == r
        r2 = Resources(cpus=0.1, ram=2048)
        assert r == r2
        assert r2 == r
        r3 = Resources(cpus=0.2, ram=2048)
        assert r != r3

    def test_to_dict(self):
        if False:
            return 10
        r = Resources(cpus=0.1, ram=2048, disk=1024, gpus=1)
        assert r.to_dict() == {'cpus': {'name': 'CPU', 'qty': 0.1, 'units_str': 'core(s)'}, 'ram': {'name': 'RAM', 'qty': 2048, 'units_str': 'MB'}, 'disk': {'name': 'Disk', 'qty': 1024, 'units_str': 'MB'}, 'gpus': {'name': 'GPU', 'qty': 1, 'units_str': 'gpu(s)'}}