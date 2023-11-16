import sys
import os
import re

class Check:

    def __init__(self, file_name):
        if False:
            print('Hello World!')
        self.file_name = file_name
        self.lineno = 0

    def parse_error(self, message, s, f):
        if False:
            while True:
                i = 10
        print('ERROR ' + message + ' ' + self.file_name + ' line ' + str(self.lineno))
        sys.stdout.write('    >>> ' + s)
        s = f.readline()
        while len(s) > 0:
            sys.stdout.write('    >>> ' + s)
            s = f.readline()
        sys.exit(1)

    def process(self):
        if False:
            print('Hello World!')
        f = open(self.file_name, 'r')
        allowed_regex_list = ['^\\* using log directory', '^\\* using R version', '^\\* using R Under development', '^\\* using platform', '^\\* using session charset', '^\\* using option .*', '^\\* checking .* \\.\\.\\. OK', '^\\* checking extension type ... Package', '^\\* this is package', '^\\* checking CRAN incoming feasibility \\.\\.\\.', '^\\*\\* found \\\\donttest examples: .*', '^Maintainer:', '^New maintainer:', '^\\s*The H2O.ai team .*', '^Version contains large components .*', '^Insufficient package version .*', '^\\Days since last update: .*', '^Old maintainer\\(s\\):', '^\\s*Tom Kraljevic .*', '^NOTE: There was 1 note.', '^\\* checking DESCRIPTION meta-information \\.\\.\\.', '^Author field differs from that derived from Authors@R', '^\\s*Author: .*', '^\\s*Authors@R: .*', '^\\n', '^New submission', '^Package was archived on CRAN', '^CRAN repository db overrides:', '^  X-CRAN-Comment: Archived on 2014-09-23 as did not comply with CRAN', '^    policies on use of multiple threads.', '^\\* checking installed package size ... NOTE', '^  installed size is .*Mb', '^  sub-directories of 1Mb or more:', '^    java  .*Mb', '^    R  .*Mb', '^    help  .*Mb', '^NOTE: There were 2 notes.', '^Status: 2 NOTEs', '^Status: 1 NOTE', '^See', '^ .*/h2o-r/h2o\\.Rcheck/00check\\.log.*', '^for details.', '^  Running .*', '^ OK', "^Package has FOSS license, installs .class/.jar but has no 'java' directory.", '^Size of tarball: .* bytes', '^\\* DONE', '^The Date field is over a month old.*', "^Checking URLs requires 'libcurl' support in the R build", '^\\* checking package dependencies ... NOTE', '^Package suggested but not available for checking*', '^\\* checking Rd cross-references ... NOTE', '^Package unavailable to check Rd xrefs*', '^Status: 3 NOTEs', '^Status: 4 NOTEs', '^\\* checking for future file timestamps ... NOTE', '^unable to verify current time']
        s = f.readline()
        while len(s) > 0:
            self.lineno = self.lineno + 1
            allowed = False
            for regex in allowed_regex_list:
                match_groups = re.search(regex, s)
                if match_groups is not None:
                    allowed = True
                    break
            if not allowed:
                self.parse_error('Illegal output found', s, f)
            s = f.readline()

def main(argv):
    if False:
        return 10
    if not os.path.exists('h2o.Rcheck'):
        print('ERROR:  You must run this script inside the generated R package source directory.')
        sys.exit(1)
    c = Check('h2o.Rcheck/00check.log')
    c.process()
    sys.exit(0)
if __name__ == '__main__':
    main(sys.argv)