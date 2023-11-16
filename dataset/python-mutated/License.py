from . import Framework

class License(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.license = self.g.get_license('mit')

    def testAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.license.key, 'mit')
        self.assertEqual(self.license.name, 'MIT License')
        self.assertEqual(self.license.description, 'A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.')
        self.assertEqual(self.license.spdx_id, 'MIT')
        self.assertEqual(self.license.body, 'MIT License\n\nCopyright (c) [year] [fullname]\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n')
        self.assertEqual(self.license.permissions, ['commercial-use', 'modifications', 'distribution', 'private-use'])
        self.assertEqual(self.license.conditions, ['include-copyright'])
        self.assertEqual(self.license.limitations, ['liability', 'warranty'])
        self.assertEqual(self.license.url, 'https://api.github.com/licenses/mit')
        self.assertEqual(self.license.html_url, 'http://choosealicense.com/licenses/mit/')
        self.assertEqual(self.license.implementation, 'Create a text file (typically named LICENSE or LICENSE.txt) in the root of your source code and copy the text of the license into the file. Replace [year] with the current year and [fullname] with the name (or names) of the copyright holders.')
        self.assertEqual(repr(self.license), 'License(name="MIT License")')