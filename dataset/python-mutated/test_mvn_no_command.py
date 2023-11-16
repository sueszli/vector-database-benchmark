import pytest
from thefuck.rules.mvn_no_command import match, get_new_command
from thefuck.types import Command

@pytest.mark.parametrize('command', [Command('mvn', '[ERROR] No goals have been specified for this build. You must specify a valid lifecycle phase or a goal in the format <plugin-prefix>:<goal> or <plugin-group-id>:<plugin-artifact-id>[:<plugin-version>]:<goal>. Available lifecycle phases are: validate, initialize, generate-sources, process-sources, generate-resources, process-resources, compile, process-classes, generate-test-sources, process-test-sources, generate-test-resources, process-test-resources, test-compile, process-test-classes, test, prepare-package, package, pre-integration-test, integration-test, post-integration-test, verify, install, deploy, pre-clean, clean, post-clean, pre-site, site, post-site, site-deploy. -> [Help 1]')])
def test_match(command):
    if False:
        i = 10
        return i + 15
    assert match(command)

@pytest.mark.parametrize('command', [Command('mvn clean', '\n[INFO] Scanning for projects...[INFO]                                                                         \n[INFO] ------------------------------------------------------------------------\n[INFO] Building test 0.2\n[INFO] ------------------------------------------------------------------------\n[INFO] \n[INFO] --- maven-clean-plugin:2.5:clean (default-clean) @ test ---\n[INFO] Deleting /home/mlk/code/test/target\n[INFO] ------------------------------------------------------------------------\n[INFO] BUILD SUCCESS\n[INFO] ------------------------------------------------------------------------\n[INFO] Total time: 0.477s\n[INFO] Finished at: Wed Aug 26 13:05:47 BST 2015\n[INFO] Final Memory: 6M/240M\n[INFO] ------------------------------------------------------------------------\n'), Command('mvn --help', ''), Command('mvn -v', '')])
def test_not_match(command):
    if False:
        i = 10
        return i + 15
    assert not match(command)

@pytest.mark.parametrize('command, new_command', [(Command('mvn', '[ERROR] No goals have been specified for this build. You must specify a valid lifecycle phase or a goal in the format <plugin-prefix>:<goal> or <plugin-group-id>:<plugin-artifact-id>[:<plugin-version>]:<goal>. Available lifecycle phases are: validate, initialize, generate-sources, process-sources, generate-resources, process-resources, compile, process-classes, generate-test-sources, process-test-sources, generate-test-resources, process-test-resources, test-compile, process-test-classes, test, prepare-package, package, pre-integration-test, integration-test, post-integration-test, verify, install, deploy, pre-clean, clean, post-clean, pre-site, site, post-site, site-deploy. -> [Help 1]'), ['mvn clean package', 'mvn clean install']), (Command('mvn -N', '[ERROR] No goals have been specified for this build. You must specify a valid lifecycle phase or a goal in the format <plugin-prefix>:<goal> or <plugin-group-id>:<plugin-artifact-id>[:<plugin-version>]:<goal>. Available lifecycle phases are: validate, initialize, generate-sources, process-sources, generate-resources, process-resources, compile, process-classes, generate-test-sources, process-test-sources, generate-test-resources, process-test-resources, test-compile, process-test-classes, test, prepare-package, package, pre-integration-test, integration-test, post-integration-test, verify, install, deploy, pre-clean, clean, post-clean, pre-site, site, post-site, site-deploy. -> [Help 1]'), ['mvn -N clean package', 'mvn -N clean install'])])
def test_get_new_command(command, new_command):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(command) == new_command