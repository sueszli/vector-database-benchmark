from io import BytesIO
import pytest
from thefuck.types import Command
from thefuck.rules.grunt_task_not_found import match, get_new_command
output = '\nWarning: Task "{}" not found. Use --force to continue.\n\nAborted due to warnings.\n\n\nExecution Time (2016-08-13 21:01:40 UTC+3)\nloading tasks  11ms  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 92%\nTotal 12ms\n\n'.format
grunt_help_stdout = b'\nGrunt: The JavaScript Task Runner (v0.4.5)\n\nUsage\n grunt [options] [task [task ...]]\n\nOptions\n    --help, -h  Display this help text.\n        --base  Specify an alternate base path. By default, all file paths are\n                relative to the Gruntfile. (grunt.file.setBase) *\n    --no-color  Disable colored output.\n   --gruntfile  Specify an alternate Gruntfile. By default, grunt looks in the\n                current or parent directories for the nearest Gruntfile.js or\n                Gruntfile.coffee file.\n   --debug, -d  Enable debugging mode for tasks that support it.\n       --stack  Print a stack trace when exiting with a warning or fatal error.\n   --force, -f  A way to force your way past warnings. Want a suggestion? Don\'t\n                use this option, fix your code.\n       --tasks  Additional directory paths to scan for task and "extra" files.\n                (grunt.loadTasks) *\n         --npm  Npm-installed grunt plugins to scan for task and "extra" files.\n                (grunt.loadNpmTasks) *\n    --no-write  Disable writing files (dry run).\n --verbose, -v  Verbose mode. A lot more information output.\n --version, -V  Print the grunt version. Combine with --verbose for more info.\n  --completion  Output shell auto-completion rules. See the grunt-cli\n                documentation for more information.\n\nOptions marked with * have methods exposed via the grunt API and should instead\nbe specified inside the Gruntfile wherever possible.\n\nAvailable tasks\n  autoprefixer  Prefix CSS files. *\n    concurrent  Run grunt tasks concurrently *\n         clean  Clean files and folders. *\n       compass  Compile Sass to CSS using Compass *\n        concat  Concatenate files. *\n       connect  Start a connect web server. *\n          copy  Copy files. *\n        cssmin  Minify CSS *\n       htmlmin  Minify HTML *\n      imagemin  Minify PNG, JPEG, GIF and SVG images *\n        jshint  Validate files with JSHint. *\n        uglify  Minify files with UglifyJS. *\n         watch  Run predefined tasks whenever watched files change.\n       filerev  File revisioning based on content hashing *\n        cdnify  Replace scripts with refs to the Google CDN *\n         karma  run karma. *\n         newer  Run a task with only those source files that have been modified\n                since the last successful run.\n     any-newer  DEPRECATED TASK.  Use the "newer" task instead\n newer-postrun  Internal task.\n   newer-clean  Remove cached timestamps.\n    ngAnnotate  Add, remove and rebuild AngularJS dependency injection\n                annotations *\n    ngconstant  Dynamic angular constant generator task. *\n        svgmin  Minify SVG *\n        usemin  Replaces references to non-minified scripts / stylesheets *\n useminPrepare  Using HTML markup as the primary source of information *\n       wiredep  Inject Bower components into your source code. *\n         serve  Compile then start a connect web server\n        server  DEPRECATED TASK. Use the "serve" task instead\n          test  Alias for "clean:server", "ngconstant:test", "wiredep",\n                "concurrent:test", "autoprefixer", "connect:test", "karma"\n                tasks.\n         build  Alias for "ngconstant:production", "clean:dist", "wiredep",\n                "useminPrepare", "concurrent:dist", "autoprefixer", "concat",\n                "ngAnnotate", "copy:dist", "cdnify", "cssmin", "uglify",\n                "filerev", "usemin", "htmlmin" tasks.\n       default  Alias for "newer:jshint", "test", "build" tasks.\n\nTasks run in the order specified. Arguments may be passed to tasks that accept\nthem by using colons, like "lint:files". Tasks marked with * are "multi tasks"\nand will iterate over all sub-targets if no argument is specified.\n\nThe list of available tasks may change based on tasks directories or grunt\nplugins specified in the Gruntfile or via command-line options.\n\nFor more information, see http://gruntjs.com/\n'

@pytest.fixture(autouse=True)
def grunt_help(mocker):
    if False:
        return 10
    patch = mocker.patch('thefuck.rules.grunt_task_not_found.Popen')
    patch.return_value.stdout = BytesIO(grunt_help_stdout)
    return patch

@pytest.mark.parametrize('command', [Command('grunt defualt', output('defualt')), Command('grunt buld:css', output('buld:css'))])
def test_match(command):
    if False:
        print('Hello World!')
    assert match(command)

@pytest.mark.parametrize('command', [Command('npm nuild', output('nuild')), Command('grunt rm', '')])
def test_not_match(command):
    if False:
        i = 10
        return i + 15
    assert not match(command)

@pytest.mark.parametrize('command, result', [(Command('grunt defualt', output('defualt')), 'grunt default'), (Command('grunt cmpass:all', output('cmpass:all')), 'grunt compass:all'), (Command('grunt cmpass:all --color', output('cmpass:all')), 'grunt compass:all --color')])
def test_get_new_command(command, result):
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(command) == result