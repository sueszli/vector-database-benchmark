from PyInstaller.depend.imphook import AdditionalFilesCache

def test_binaries_and_datas():
    if False:
        while True:
            i = 10
    datas = [('source', 'dest'), ('src', 'dst')]
    binaries = [('abc', 'def'), ('ghi', 'jkl')]
    modules = ['testmodule1', 'testmodule2']
    cache = AdditionalFilesCache()
    for modname in modules:
        cache.add(modname, binaries, datas)
        assert cache.datas(modname) == datas
        cache.add(modname, binaries, datas)
        assert cache.binaries(modname) != binaries
        assert cache.binaries(modname) == binaries * 2