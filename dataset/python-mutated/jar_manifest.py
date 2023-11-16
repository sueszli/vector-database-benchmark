"""jc - JSON Convert Java `MANIFEST.MF` file parser

Usage (cli):

    $ cat MANIFEST.MF | jc --jar-manifest

Usage (module):

    import jc
    result = jc.parse('jar_manifest', jar_manifest_file_output)

Schema:

    [
      {
        "key1":     string,
        "key2":     string
      }
    ]

Examples:

    $ cat MANIFEST.MF | jc --jar-manifest -p
    $ unzip -c log4j-core-2.16.0.jar META-INF/MANIFEST.MF | \\
      jc --jar-manifest -p
    $ unzip -c 'apache-log4j-2.16.0-bin/*.jar' META-INF/MANIFEST.MF | \\
      jc --jar-manifest -p

    $ cat MANIFEST.MF | jc --jar-manifest -p
    [
      {
        "Import_Package": "com.conversantmedia.util.concurrent;resoluti...",
        "Export_Package": "org.apache.logging.log4j.core;uses:="org.ap...",
        "Manifest_Version": "1.0",
        "Bundle_License": "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "Bundle_SymbolicName": "org.apache.logging.log4j.core",
        "Built_By": "matt",
        "Bnd_LastModified": "1639373735804",
        "Implementation_Vendor_Id": "org.apache.logging.log4j",
        "Specification_Title": "Apache Log4j Core",
        "Log4jReleaseManager": "Matt Sicker",
        ...
      }
    ]

    $ unzip -c 'apache-log4j-2.16.0-bin/*.jar' META-INF/MANIFEST.MF | \\
      jc --jar-manifest -p
    [
      ...
      {
        "Archive": "apache-log4j-2.16.0-bin/log4j-spring-boot-2.16.0-so...",
        "Manifest_Version": "1.0",
        "Built_By": "matt",
        "Created_By": "Apache Maven 3.8.4",
        "Build_Jdk": "1.8.0_312"
      },
      {
        "Archive": "apache-log4j-2.16.0-bin/log4j-spring-boot-2.16.0-ja...",
        "Manifest_Version": "1.0",
        "Built_By": "matt",
        "Created_By": "Apache Maven 3.8.4",
        "Build_Jdk": "1.8.0_312"
      },
      {
        "Bundle_SymbolicName": "org.apache.logging.log4j.spring-cloud-c...",
        "Export_Package": "org.apache.logging.log4j.spring.cloud.config...",
        "Archive": "apache-log4j-2.16.0-bin/log4j-spring-cloud-config-c...",
        "Manifest_Version": "1.0",
        "Bundle_License": "https://www.apache.org/licenses/LICENSE-2.0.txt",
        ...
      }
      ...
    ]
"""
import jc.utils
import re

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '0.01'
    description = 'Java MANIFEST.MF file parser'
    author = 'Matt J'
    author_email = 'https://github.com/listuser'
    compatible = ['linux', 'darwin', 'cygwin', 'win32', 'aix', 'freebsd']
    tags = ['file']
__version__ = info.version

def _process(proc_data):
    if False:
        i = 10
        return i + 15
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (List of Dictionaries) raw structured data to process\n\n    Returns:\n\n        List of Dictionaries. Structured data to conform to the schema.\n    '
    return proc_data

def parse(data, raw=False, quiet=False):
    if False:
        return 10
    '\n    Main text parsing function\n\n    Parameters:\n\n        data:        (string)  text data to parse\n        raw:         (boolean) unprocessed output if True\n        quiet:       (boolean) suppress warning messages if True\n\n    Returns:\n\n        List of Dictionaries. Raw or processed structured data.\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    jc.utils.input_type_check(data)
    raw_output = []
    archives = []
    if jc.utils.has_data(data):
        datalines = data.splitlines()
        if datalines[-1].endswith('archives were successfully processed.'):
            datalines.pop(-1)
        this_archive = []
        for row in datalines:
            if row == '':
                archives.append(this_archive)
                this_archive = []
                continue
            this_archive.append(row)
        if this_archive:
            archives.append(this_archive)
        for archive_item in archives:
            manifests = []
            this_manifest = {}
            plines = []
            for (i, line) in enumerate(archive_item):
                last = archive_item[-1]
                if re.match('^\\s+inflating\\s*:\\s*META-INF/MANIFEST.MF', line, re.IGNORECASE):
                    archive_item.pop(i)
                    continue
                if re.match('\\s', line):
                    if not this_manifest:
                        (k, v) = archive_item[i - 1].split(':', maxsplit=1)
                        v = v + line
                        v = re.sub('\\s', '', v)
                        this_manifest = {k: v}
                        plines.append(i - 1)
                        plines.append(i)
                    else:
                        plines.append(i)
                        linecmp = line
                        for (k, v) in this_manifest.items():
                            line = v + line
                            line = re.sub('\\s', '', line)
                        this_manifest.update({k: line})
                        if linecmp is not last:
                            nextline = archive_item[i + 1]
                            if re.match('\\S', nextline):
                                manifests.append(this_manifest)
                                this_manifest = False
                            else:
                                manifests.append(this_manifest)
            if plines:
                for p in reversed(plines):
                    archive_item.pop(p)
            for (i, line) in enumerate(archive_item):
                (k, v) = line.split(':', maxsplit=1)
                v = v.strip()
                manifests.append({k: v})
            if manifests:
                this_manifest = {}
                for d in manifests:
                    for (k, v) in d.items():
                        k = re.sub('\\s', '', k)
                        k = re.sub('-', '_', k)
                        this_manifest.update({k: v})
                raw_output.append(this_manifest)
    return raw_output if raw else _process(raw_output)