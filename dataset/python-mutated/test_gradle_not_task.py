import pytest
from io import BytesIO
from thefuck.rules.gradle_no_task import match, get_new_command
from thefuck.types import Command
gradle_tasks = b"\n:tasks\n\n------------------------------------------------------------\nAll tasks runnable from root project\n------------------------------------------------------------\n\nAndroid tasks\n-------------\nandroidDependencies - Displays the Android dependencies of the project.\nsigningReport - Displays the signing info for each variant.\nsourceSets - Prints out all the source sets defined in this project.\n\nBuild tasks\n-----------\nassemble - Assembles all variants of all applications and secondary packages.\nassembleAndroidTest - Assembles all the Test applications.\nassembleDebug - Assembles all Debug builds.\nassembleRelease - Assembles all Release builds.\nbuild - Assembles and tests this project.\nbuildDependents - Assembles and tests this project and all projects that depend on it.\nbuildNeeded - Assembles and tests this project and all projects it depends on.\ncompileDebugAndroidTestSources\ncompileDebugSources\ncompileDebugUnitTestSources\ncompileReleaseSources\ncompileReleaseUnitTestSources\nextractDebugAnnotations - Extracts Android annotations for the debug variant into the archive file\nextractReleaseAnnotations - Extracts Android annotations for the release variant into the archive file\nmockableAndroidJar - Creates a version of android.jar that's suitable for unit tests.\n\nBuild Setup tasks\n-----------------\ninit - Initializes a new Gradle build. [incubating]\nwrapper - Generates Gradle wrapper files. [incubating]\n\nHelp tasks\n----------\ncomponents - Displays the components produced by root project 'org.rerenderer_example.snake'. [incubating]\ndependencies - Displays all dependencies declared in root project 'org.rerenderer_example.snake'.\ndependencyInsight - Displays the insight into a specific dependency in root project 'org.rerenderer_example.snake'.\nhelp - Displays a help message.\nmodel - Displays the configuration model of root project 'org.rerenderer_example.snake'. [incubating]\nprojects - Displays the sub-projects of root project 'org.rerenderer_example.snake'.\nproperties - Displays the properties of root project 'org.rerenderer_example.snake'.\ntasks - Displays the tasks runnable from root project 'org.rerenderer_example.snake' (some of the displayed tasks may belong to subprojects).\n\nInstall tasks\n-------------\ninstallDebug - Installs the Debug build.\ninstallDebugAndroidTest - Installs the android (on device) tests for the Debug build.\ninstallRelease - Installs the Release build.\nuninstallAll - Uninstall all applications.\nuninstallDebug - Uninstalls the Debug build.\nuninstallDebugAndroidTest - Uninstalls the android (on device) tests for the Debug build.\nuninstallRelease - Uninstalls the Release build.\n\nReact tasks\n-----------\nbundleDebugJsAndAssets - bundle JS and assets for Debug.\nbundleReleaseJsAndAssets - bundle JS and assets for Release.\n\nVerification tasks\n------------------\ncheck - Runs all checks.\nclean - Deletes the build directory.\nconnectedAndroidTest - Installs and runs instrumentation tests for all flavors on connected devices.\nconnectedCheck - Runs all device checks on currently connected devices.\nconnectedDebugAndroidTest - Installs and runs the tests for debug on connected devices.\ndeviceAndroidTest - Installs and runs instrumentation tests using all Device Providers.\ndeviceCheck - Runs all device checks using Device Providers and Test Servers.\nlint - Runs lint on all variants.\nlintDebug - Runs lint on the Debug build.\nlintRelease - Runs lint on the Release build.\ntest - Run unit tests for all variants.\ntestDebugUnitTest - Run unit tests for the debug build.\ntestReleaseUnitTest - Run unit tests for the release build.\n\nOther tasks\n-----------\nassembleDefault\ncopyDownloadableDepsToLibs\njarDebugClasses\njarReleaseClasses\n\nTo see all tasks and more detail, run gradlew tasks --all\n\nTo see more detail about a task, run gradlew help --task <task>\n\nBUILD SUCCESSFUL\n\nTotal time: 1.936 secs\n"
output_not_found = "\n\nFAILURE: Build failed with an exception.\n\n* What went wrong:\nTask '{}' not found in root project 'org.rerenderer_example.snake'.\n\n* Try:\nRun gradlew tasks to get a list of available tasks. Run with --stacktrace option to get the stack trace. Run with --info or --debug option to get more log output.\n".format
output_ambiguous = "\n\nFAILURE: Build failed with an exception.\n\n* What went wrong:\nTask '{}' is ambiguous in root project 'org.rerenderer_example.snake'. Candidates are: 'assembleRelease', 'assembleReleaseUnitTest'.\n\n* Try:\nRun gradlew tasks to get a list of available tasks. Run with --stacktrace option to get the stack trace. Run with --info or --debug option to get more log output.\n".format

@pytest.fixture(autouse=True)
def tasks(mocker):
    if False:
        print('Hello World!')
    patch = mocker.patch('thefuck.rules.gradle_no_task.Popen')
    patch.return_value.stdout = BytesIO(gradle_tasks)
    return patch

@pytest.mark.parametrize('command', [Command('./gradlew assembler', output_ambiguous('assembler')), Command('./gradlew instar', output_not_found('instar')), Command('gradle assembler', output_ambiguous('assembler')), Command('gradle instar', output_not_found('instar'))])
def test_match(command):
    if False:
        for i in range(10):
            print('nop')
    assert match(command)

@pytest.mark.parametrize('command', [Command('./gradlew assemble', ''), Command('gradle assemble', ''), Command('npm assembler', output_ambiguous('assembler')), Command('npm instar', output_not_found('instar'))])
def test_not_match(command):
    if False:
        i = 10
        return i + 15
    assert not match(command)

@pytest.mark.parametrize('command, result', [(Command('./gradlew assembler', output_ambiguous('assembler')), './gradlew assemble'), (Command('./gradlew instardebug', output_not_found('instardebug')), './gradlew installDebug'), (Command('gradle assembler', output_ambiguous('assembler')), 'gradle assemble'), (Command('gradle instardebug', output_not_found('instardebug')), 'gradle installDebug')])
def test_get_new_command(command, result):
    if False:
        i = 10
        return i + 15
    assert get_new_command(command)[0] == result