import sys
import pytest
import ray

@pytest.mark.skipif(sys.platform == 'win32', reason='Failing on Windows, affected by protocolbuffers/protobuf#10075.')
def test_protobuf_compatibility(shutdown_only):
    if False:
        i = 10
        return i + 15
    protobuf_4_21_0 = {'pip': ['protobuf==4.21.0']}
    protobuf_3_12_2 = {'pip': ['protobuf==3.12.2']}
    ray.init()

    @ray.remote
    def load_ray():
        if False:
            while True:
                i = 10
        import google.protobuf
        import ray
        return google.protobuf.__version__
    assert '4.21.0' == ray.get(load_ray.options(runtime_env=protobuf_4_21_0).remote())
    assert '3.12.2' == ray.get(load_ray.options(runtime_env=protobuf_3_12_2).remote())
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-sv', __file__]))