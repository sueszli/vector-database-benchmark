import os
from shutil import rmtree
from tempfile import mkdtemp
import filecmp
import tools.pdf2txt as pdf2txt
from helpers import absolute_sample_path
from tempfilepath import TemporaryFilePath

def run(sample_path, options=None):
    if False:
        i = 10
        return i + 15
    absolute_path = absolute_sample_path(sample_path)
    with TemporaryFilePath() as output_file_name:
        if options:
            s = 'pdf2txt -o{} {} {}'.format(output_file_name, options, absolute_path)
        else:
            s = 'pdf2txt -o{} {}'.format(output_file_name, absolute_path)
        pdf2txt.main(s.split(' ')[1:])

class TestPdf2Txt:

    def test_jo(self):
        if False:
            while True:
                i = 10
        run('jo.pdf')

    def test_simple1(self):
        if False:
            for i in range(10):
                print('nop')
        run('simple1.pdf')

    def test_simple2(self):
        if False:
            print('Hello World!')
        run('simple2.pdf')

    def test_simple3(self):
        if False:
            for i in range(10):
                print('nop')
        run('simple3.pdf')

    def test_sample_one_byte_identity_encode(self):
        if False:
            return 10
        run('sampleOneByteIdentityEncode.pdf')

    def test_nonfree_175(self):
        if False:
            print('Hello World!')
        'Regression test for:\n        https://github.com/pdfminer/pdfminer.six/issues/65\n        '
        run('nonfree/175.pdf')

    def test_nonfree_dmca(self):
        if False:
            for i in range(10):
                print('nop')
        run('nonfree/dmca.pdf')

    def test_nonfree_f1040nr(self):
        if False:
            i = 10
            return i + 15
        run('nonfree/f1040nr.pdf', '-p 1')

    def test_nonfree_i1040nr(self):
        if False:
            i = 10
            return i + 15
        run('nonfree/i1040nr.pdf', '-p 1')

    def test_nonfree_kampo(self):
        if False:
            return 10
        run('nonfree/kampo.pdf')

    def test_nonfree_naacl06_shinyama(self):
        if False:
            return 10
        run('nonfree/naacl06-shinyama.pdf')

    def test_nlp2004slides(self):
        if False:
            while True:
                i = 10
        run('nonfree/nlp2004slides.pdf', '-p 1')

    def test_contrib_2b(self):
        if False:
            print('Hello World!')
        run('contrib/2b.pdf', '-A -t xml')

    def test_contrib_issue_350(self):
        if False:
            print('Hello World!')
        'Regression test for\n        https://github.com/pdfminer/pdfminer.six/issues/350'
        run('contrib/issue-00352-asw-oct96-p41.pdf')

    def test_scancode_patchelf(self):
        if False:
            return 10
        'Regression test for https://github.com/euske/pdfminer/issues/96'
        run('scancode/patchelf.pdf')

    def test_contrib_hash_two_complement(self):
        if False:
            while True:
                i = 10
        'Check that unsigned integer is added correctly to encryption hash.et\n\n        See https://github.com/pdfminer/pdfminer.six/issues/186\n        '
        run('contrib/issue-00352-hash-twos-complement.pdf')

    def test_contrib_excel(self):
        if False:
            i = 10
            return i + 15
        'Regression test for\n        https://github.com/pdfminer/pdfminer.six/issues/369\n        '
        run('contrib/issue-00369-excel.pdf', '-t html')

    def test_encryption_aes128(self):
        if False:
            return 10
        run('encryption/aes-128.pdf', '-P foo')

    def test_encryption_aes128m(self):
        if False:
            print('Hello World!')
        run('encryption/aes-128-m.pdf', '-P foo')

    def test_encryption_aes256(self):
        if False:
            while True:
                i = 10
        run('encryption/aes-256.pdf', '-P foo')

    def test_encryption_aes256m(self):
        if False:
            return 10
        run('encryption/aes-256-m.pdf', '-P foo')

    def test_encryption_aes256_r6_user(self):
        if False:
            i = 10
            return i + 15
        run('encryption/aes-256-r6.pdf', '-P usersecret')

    def test_encryption_aes256_r6_owner(self):
        if False:
            return 10
        run('encryption/aes-256-r6.pdf', '-P ownersecret')

    def test_encryption_base(self):
        if False:
            print('Hello World!')
        run('encryption/base.pdf', '-P foo')

    def test_encryption_rc4_40(self):
        if False:
            print('Hello World!')
        run('encryption/rc4-40.pdf', '-P foo')

    def test_encryption_rc4_128(self):
        if False:
            i = 10
            return i + 15
        run('encryption/rc4-128.pdf', '-P foo')

    def test_html_simple1(self):
        if False:
            print('Hello World!')
        run('simple1.pdf', '-t html')

    def test_hocr_simple1(self):
        if False:
            for i in range(10):
                print('nop')
        run('simple1.pdf', '-t hocr')

class TestDumpImages:

    @staticmethod
    def extract_images(input_file, *args):
        if False:
            return 10
        output_dir = mkdtemp()
        with TemporaryFilePath() as output_file_name:
            commands = ['-o', output_file_name, '--output-dir', output_dir, input_file, *args]
            pdf2txt.main(commands)
        image_files = os.listdir(output_dir)
        rmtree(output_dir)
        return image_files

    def test_nonfree_dmca(self):
        if False:
            return 10
        'Extract images of pdf containing bmp images\n\n        Regression test for:\n        https://github.com/pdfminer/pdfminer.six/issues/131\n        '
        filepath = absolute_sample_path('../samples/nonfree/dmca.pdf')
        image_files = self.extract_images(filepath, '-p', '1')
        assert image_files[0].endswith('bmp')

    def test_nonfree_175(self):
        if False:
            while True:
                i = 10
        'Extract images of pdf containing jpg images'
        self.extract_images(absolute_sample_path('../samples/nonfree/175.pdf'))

    def test_jbig2_image_export(self):
        if False:
            while True:
                i = 10
        'Extract images of pdf containing jbig2 images\n\n        Feature test for: https://github.com/pdfminer/pdfminer.six/pull/46\n        '
        input_file = absolute_sample_path('../samples/contrib/pdf-with-jbig2.pdf')
        output_dir = mkdtemp()
        with TemporaryFilePath() as output_file_name:
            commands = ['-o', output_file_name, '--output-dir', output_dir, input_file]
            pdf2txt.main(commands)
        image_files = os.listdir(output_dir)
        try:
            assert image_files[0].endswith('.jb2')
            assert filecmp.cmp(output_dir + '/' + image_files[0], absolute_sample_path('../samples/contrib/XIPLAYER0.jb2'))
        finally:
            rmtree(output_dir)

    def test_contrib_matplotlib(self):
        if False:
            i = 10
            return i + 15
        'Test a pdf with Type3 font'
        run('contrib/matplotlib.pdf')

    def test_nonfree_cmp_itext_logo(self):
        if False:
            return 10
        'Test a pdf with Type3 font'
        run('nonfree/cmp_itext_logo.pdf')