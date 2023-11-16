import BoostBuild

def test_generator_added_after_already_building_a_target_of_its_target_type():
    if False:
        while True:
            i = 10
    '\n      Regression test for a Boost Build bug causing it to not use a generator\n    if it got added after already building a target of its target type.\n\n    '
    t = BoostBuild.Tester()
    t.write('dummy.cpp', 'void f() {}\n')
    t.write('jamroot.jam', 'import common ;\nimport generators ;\nimport type ;\ntype.register MY_OBJ : my_obj ;\ngenerators.register-standard common.copy : CPP : MY_OBJ ;\n\n# Building this dummy target must not cause a later defined CPP target type\n# generator not to be recognized as viable.\nmy-obj dummy : dummy.cpp ;\nalias the-other-obj : Other//other-obj ;\n')
    t.write('Other/source.extension', 'A dummy source file.')
    t.write('Other/mygen.jam', 'import common ;\nimport generators ;\nimport type ;\ntype.register MY_TYPE : extension ;\ngenerators.register-standard $(__name__).generate-a-cpp-file : MY_TYPE : CPP ;\nrule generate-a-cpp-file { ECHO Generating a CPP file... ; }\nCREATE-FILE = [ common.file-creation-command ] ;\nactions generate-a-cpp-file { $(CREATE-FILE) "$(<)" }\n')
    t.write('Other/mygen.py', 'import b2.build.generators as generators\nimport b2.build.type as type\n\nfrom b2.manager import get_manager\n\nimport os\n\ntype.register(\'MY_TYPE\', [\'extension\'])\ngenerators.register_standard(\'mygen.generate-a-cpp-file\', [\'MY_TYPE\'], [\'CPP\'])\nif os.name == \'nt\':\n    action = \'echo void g() {} > "$(<)"\'\nelse:\n    action = \'echo "void g() {}" > "$(<)"\'\ndef f(*args):\n    print "Generating a CPP file..."\n\nget_manager().engine().register_action("mygen.generate-a-cpp-file", action,\n    function=f)\n')
    t.write('Other/jamfile.jam', 'import mygen ;\nmy-obj other-obj : source.extension ;\n')
    t.run_build_system()
    t.expect_output_lines('Generating a CPP file...')
    t.expect_addition('bin/dummy.my_obj')
    t.expect_addition('Other/bin/other-obj.cpp')
    t.expect_addition('Other/bin/other-obj.my_obj')
    t.expect_nothing_more()
    t.cleanup()

def test_using_a_derived_source_type_created_after_generator_already_used():
    if False:
        for i in range(10):
            print('nop')
    "\n      Regression test for a Boost Build bug causing it to not use a generator\n    with a source type derived from one of the generator's sources but created\n    only after already using the generateor.\n\n    "
    t = BoostBuild.Tester()
    t.write('dummy.xxx', 'Hello. My name is Peter Pan.\n')
    t.write('jamroot.jam', 'import common ;\nimport generators ;\nimport type ;\ntype.register XXX : xxx ;\ntype.register YYY : yyy ;\ngenerators.register-standard common.copy : XXX : YYY ;\n\n# Building this dummy target must not cause a later defined XXX2 target type not\n# to be recognized as a viable source type for building YYY targets.\nyyy dummy : dummy.xxx ;\nalias the-test-output : Other//other ;\n')
    t.write('Other/source.xxx2', 'Hello. My name is Tinkerbell.\n')
    t.write('Other/jamfile.jam', 'import type ;\ntype.register XXX2 : xxx2 : XXX ;\n# We are careful not to do anything between defining our new XXX2 target type\n# and using the XXX --> YYY generator that could potentially cover the Boost\n# Build bug by clearing its internal viable source target type state.\nyyy other : source.xxx2 ;\n')
    t.run_build_system()
    t.expect_addition('bin/dummy.yyy')
    t.expect_addition('Other/bin/other.yyy')
    t.expect_nothing_more()
    t.cleanup()
test_generator_added_after_already_building_a_target_of_its_target_type()
test_using_a_derived_source_type_created_after_generator_already_used()