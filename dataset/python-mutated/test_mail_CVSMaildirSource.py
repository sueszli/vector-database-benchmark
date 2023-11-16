from email import message_from_string
from email.utils import mktime_tz
from email.utils import parsedate_tz
from twisted.trial import unittest
from buildbot.changes.mail import CVSMaildirSource
cvs1_11_msg = 'From: Andy Howell <andy@example.com>\nTo: buildbot@example.com\nSubject: cvs module MyModuleName\nDate: Sat, 07 Aug 2010 11:11:49 +0000\nX-Mailer: Python buildbot-cvs-mail $Revision: 1.3 $\n\nCvsmode: 1.11\nCategory: None\nCVSROOT: :ext:cvshost.example.com:/cvsroot\nFiles: base/module/src/make GNUmakefile,1.362,1.363\nProject: MyModuleName\nUpdate of /cvsroot/base/module/src/make\nIn directory cvshost:/tmp/cvs-serv10922\n\nModified Files:\n        GNUmakefile\nLog Message:\nCommented out some stuff.\n'
cvs1_12_msg = 'Date: Wed, 11 Aug 2010 04:56:44 +0000\nFrom: andy@example.com\nTo: buildbot@example.com\nSubject: cvs update for project RaiCore\nX-Mailer: Python buildbot-cvs-mail $Revision: 1.3 $\n\nCvsmode: 1.12\nCategory: None\nCVSROOT: :ext:cvshost.example.com:/cvsroot\nFiles: file1.cpp 1.77 1.78 file2.cpp 1.75 1.76\nPath: base/module/src\nProject: MyModuleName\nUpdate of /cvsroot/base/module/src\nIn directory example.com:/tmp/cvs-serv26648/InsightMonAgent\n\nModified Files:\n        file1.cpp file2.cpp\nLog Message:\nChanges for changes sake\n'

class TestCVSMaildirSource(unittest.TestCase):

    def test_CVSMaildirSource_create_change_from_cvs1_11msg(self):
        if False:
            i = 10
            return i + 15
        m = message_from_string(cvs1_11_msg)
        src = CVSMaildirSource('/dev/null')
        (src, chdict) = src.parse(m)
        self.assertNotEqual(chdict, None)
        self.assertEqual(chdict['author'], 'andy')
        self.assertEqual(len(chdict['files']), 1)
        self.assertEqual(chdict['files'][0], 'base/module/src/make/GNUmakefile')
        self.assertEqual(chdict['comments'], 'Commented out some stuff.\n')
        self.assertFalse(chdict['isdir'])
        self.assertEqual(chdict['revision'], '2010-08-07 11:11:49')
        dateTuple = parsedate_tz('Sat, 07 Aug 2010 11:11:49 +0000')
        self.assertEqual(chdict['when'], mktime_tz(dateTuple))
        self.assertEqual(chdict['branch'], None)
        self.assertEqual(chdict['repository'], ':ext:cvshost.example.com:/cvsroot')
        self.assertEqual(chdict['project'], 'MyModuleName')
        self.assertEqual(len(chdict['properties']), 0)
        self.assertEqual(src, 'cvs')

    def test_CVSMaildirSource_create_change_from_cvs1_12msg(self):
        if False:
            print('Hello World!')
        m = message_from_string(cvs1_12_msg)
        src = CVSMaildirSource('/dev/null')
        (src, chdict) = src.parse(m)
        self.assertNotEqual(chdict, None)
        self.assertEqual(chdict['author'], 'andy')
        self.assertEqual(len(chdict['files']), 2)
        self.assertEqual(chdict['files'][0], 'base/module/src/file1.cpp')
        self.assertEqual(chdict['files'][1], 'base/module/src/file2.cpp')
        self.assertEqual(chdict['comments'], 'Changes for changes sake\n')
        self.assertFalse(chdict['isdir'])
        self.assertEqual(chdict['revision'], '2010-08-11 04:56:44')
        dateTuple = parsedate_tz('Wed, 11 Aug 2010 04:56:44 +0000')
        self.assertEqual(chdict['when'], mktime_tz(dateTuple))
        self.assertEqual(chdict['branch'], None)
        self.assertEqual(chdict['repository'], ':ext:cvshost.example.com:/cvsroot')
        self.assertEqual(chdict['project'], 'MyModuleName')
        self.assertEqual(len(chdict['properties']), 0)
        self.assertEqual(src, 'cvs')

    def test_CVSMaildirSource_create_change_from_cvs1_12_with_no_path(self):
        if False:
            return 10
        msg = cvs1_12_msg.replace('Path: base/module/src', '')
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        try:
            assert src.parse(m)[1]
        except ValueError:
            pass
        else:
            self.fail('Expect ValueError.')

    def test_CVSMaildirSource_create_change_with_bad_cvsmode(self):
        if False:
            return 10
        msg = cvs1_11_msg.replace('Cvsmode: 1.11', 'Cvsmode: 9.99')
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        try:
            assert src.parse(m)[1]
        except ValueError:
            pass
        else:
            self.fail('Expected ValueError')

    def test_CVSMaildirSource_create_change_with_branch(self):
        if False:
            i = 10
            return i + 15
        msg = cvs1_11_msg.replace('        GNUmakefile', '      Tag: Test_Branch\n      GNUmakefile')
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        chdict = src.parse(m)[1]
        self.assertEqual(chdict['branch'], 'Test_Branch')

    def test_CVSMaildirSource_create_change_with_category(self):
        if False:
            while True:
                i = 10
        msg = cvs1_11_msg.replace('Category: None', 'Category: Test category')
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        chdict = src.parse(m)[1]
        self.assertEqual(chdict['category'], 'Test category')

    def test_CVSMaildirSource_create_change_with_no_comment(self):
        if False:
            i = 10
            return i + 15
        msg = cvs1_11_msg[:cvs1_11_msg.find('Commented out some stuff')]
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        chdict = src.parse(m)[1]
        self.assertEqual(chdict['comments'], None)

    def test_CVSMaildirSource_create_change_with_no_files(self):
        if False:
            i = 10
            return i + 15
        msg = cvs1_11_msg.replace('Files: base/module/src/make GNUmakefile,1.362,1.363', '')
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        chdict = src.parse(m)
        self.assertEqual(chdict, None)

    def test_CVSMaildirSource_create_change_with_no_project(self):
        if False:
            print('Hello World!')
        msg = cvs1_11_msg.replace('Project: MyModuleName', '')
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        chdict = src.parse(m)[1]
        self.assertEqual(chdict['project'], None)

    def test_CVSMaildirSource_create_change_with_no_repository(self):
        if False:
            while True:
                i = 10
        msg = cvs1_11_msg.replace('CVSROOT: :ext:cvshost.example.com:/cvsroot', '')
        m = message_from_string(msg)
        src = CVSMaildirSource('/dev/null')
        chdict = src.parse(m)[1]
        self.assertEqual(chdict['repository'], None)

    def test_CVSMaildirSource_create_change_with_property(self):
        if False:
            return 10
        m = message_from_string(cvs1_11_msg)
        propDict = {'foo': 'bar'}
        src = CVSMaildirSource('/dev/null', properties=propDict)
        chdict = src.parse(m)[1]
        self.assertEqual(chdict['properties']['foo'], 'bar')