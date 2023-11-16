import BoostBuild
configuring_default_toolset_message = 'warning: Configuring default toolset "%s".'

def test_conditions_on_default_toolset():
    if False:
        i = 10
        return i + 15
    'Test that toolset and toolset subfeature conditioned properties get\n    applied correctly when the toolset is selected by default. Implicitly tests\n    that we can use the set-default-toolset rule to set the default toolset to\n    be used by Boost Build.\n    '
    t = BoostBuild.Tester('--user-config= --ignore-site-config', pass_toolset=False, use_test_config=False)
    toolset_name = 'myCustomTestToolset'
    toolset_version = 'v'
    toolset_version_unused = 'v_unused'
    message_loaded = "Toolset '%s' loaded." % toolset_name
    message_initialized = "Toolset '%s' initialized." % toolset_name
    t.write(toolset_name + '.jam', '\nimport feature ;\nECHO "%(message_loaded)s" ;\nfeature.extend toolset : %(toolset_name)s ;\nfeature.subfeature toolset %(toolset_name)s : version : %(toolset_version)s %(toolset_version_unused)s ;\nrule init ( version ) { ECHO "%(message_initialized)s" ; }\n' % {'message_loaded': message_loaded, 'message_initialized': message_initialized, 'toolset_name': toolset_name, 'toolset_version': toolset_version, 'toolset_version_unused': toolset_version_unused})
    t.write('jamroot.jam', '\nimport build-system ;\nimport errors ;\nimport feature ;\nimport notfile ;\n\nbuild-system.set-default-toolset %(toolset_name)s : %(toolset_version)s ;\n\nfeature.feature description : : free incidental ;\n\n# We use a rule instead of an action to avoid problems with action output not\n# getting piped to stdout by the testing system.\nrule buildRule ( names : targets ? : properties * )\n{\n    local descriptions = [ feature.get-values description : $(properties) ] ;\n    ECHO "descriptions:" /$(descriptions)/ ;\n    local toolset = [ feature.get-values toolset : $(properties) ] ;\n    ECHO "toolset:" /$(toolset)/ ;\n    local toolset-version = [ feature.get-values "toolset-$(toolset):version" : $(properties) ] ;\n    ECHO "toolset-version:" /$(toolset-version)/ ;\n}\n\nnotfile testTarget\n    : @buildRule\n    :\n    :\n    <description>stand-alone\n    <toolset>%(toolset_name)s:<description>toolset\n    <toolset>%(toolset_name)s-%(toolset_version)s:<description>toolset-version\n    <toolset>%(toolset_name)s-%(toolset_version_unused)s:<description>toolset-version-unused ;\n' % {'toolset_name': toolset_name, 'toolset_version': toolset_version, 'toolset_version_unused': toolset_version_unused})
    t.run_build_system()
    t.expect_output_lines(configuring_default_toolset_message % toolset_name)
    t.expect_output_lines(message_loaded)
    t.expect_output_lines(message_initialized)
    t.expect_output_lines('descriptions: /stand-alone/ /toolset/ /toolset-version/')
    t.expect_output_lines('toolset: /%s/' % toolset_name)
    t.expect_output_lines('toolset-version: /%s/' % toolset_version)
    t.cleanup()

def test_default_toolset_on_os(os, expected_toolset):
    if False:
        i = 10
        return i + 15
    'Test that the given toolset is used as the default toolset on the given\n    os. Uses hardcoded knowledge of how Boost Build decides on which host OS it\n    is currently running. Note that we must not do much after tricking Boost\n    Build into believing it has a specific host OS as this might mess up other\n    important internal Boost Build state.\n    '
    t = BoostBuild.Tester('--user-config= --ignore-site-config', pass_toolset=False, use_test_config=False)
    t.write('jamroot.jam', 'modules.poke os : .name : %s ;' % os)
    t.run_build_system(stderr=None)
    t.expect_output_lines(configuring_default_toolset_message % expected_toolset)
    t.cleanup()

def test_default_toolset_requirements():
    if False:
        print('Hello World!')
    "Test that default toolset's requirements get applied correctly.\n    "
    t = BoostBuild.Tester('--user-config= --ignore-site-config', pass_toolset=False, use_test_config=False, ignore_toolset_requirements=False)
    toolset_name = 'customTestToolsetWithRequirements'
    t.write(toolset_name + '.jam', '\nimport feature ;\nimport toolset ;\nfeature.extend toolset : %(toolset_name)s ;\ntoolset.add-requirements <description>toolset-requirement ;\nrule init ( ) { }\n' % {'toolset_name': toolset_name})
    t.write('jamroot.jam', '\nimport build-system ;\nimport errors ;\nimport feature ;\nimport notfile ;\n\nbuild-system.set-default-toolset %(toolset_name)s ;\n\nfeature.feature description : : free incidental ;\n\n# We use a rule instead of an action to avoid problems with action output not\n# getting piped to stdout by the testing system.\nrule buildRule ( names : targets ? : properties * )\n{\n    local descriptions = [ feature.get-values description : $(properties) ] ;\n    ECHO "descriptions:" /$(descriptions)/ ;\n    local toolset = [ feature.get-values toolset : $(properties) ] ;\n    ECHO "toolset:" /$(toolset)/ ;\n}\n\nnotfile testTarget\n    : @buildRule\n    :\n    :\n    <description>target-requirement\n    <description>toolset-requirement:<description>conditioned-requirement\n    <description>unrelated-condition:<description>unrelated-description ;\n' % {'toolset_name': toolset_name})
    t.run_build_system()
    t.expect_output_lines(configuring_default_toolset_message % toolset_name)
    t.expect_output_lines('descriptions: /conditioned-requirement/ /target-requirement/ /toolset-requirement/')
    t.expect_output_lines('toolset: /%s/' % toolset_name)
    t.cleanup()
test_default_toolset_on_os('NT', 'msvc')
test_default_toolset_on_os('LINUX', 'gcc')
test_default_toolset_on_os('CYGWIN', 'gcc')
test_default_toolset_on_os('SomeOtherOS', 'gcc')
test_default_toolset_requirements()
test_conditions_on_default_toolset()