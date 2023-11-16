"""Test the pypdf._encryption module."""
import secrets
from pathlib import Path
import pytest
import pypdf
from pypdf import PasswordType, PdfReader, PdfWriter
from pypdf._crypt_providers import crypt_provider
from pypdf._crypt_providers._fallback import _DEPENDENCY_ERROR_STR
from pypdf._encryption import AlgV5, CryptAES, CryptRC4
from pypdf.errors import DependencyError, PdfReadError
USE_CRYPTOGRAPHY = crypt_provider[0] == 'cryptography'
USE_PYCRYPTODOME = crypt_provider[0] == 'pycryptodome'
HAS_AES = USE_CRYPTOGRAPHY or USE_PYCRYPTODOME
TESTS_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_ROOT.parent
RESOURCE_ROOT = PROJECT_ROOT / 'resources'
SAMPLE_ROOT = PROJECT_ROOT / 'sample-files'

@pytest.mark.parametrize(('name', 'requires_aes'), [('unencrypted.pdf', False), ('r2-empty-password.pdf', False), ('r3-empty-password.pdf', False), ('r2-user-password.pdf', False), ('r2-owner-password.pdf', False), ('r3-user-password.pdf', False), ('r4-user-password.pdf', False), ('r4-owner-password.pdf', False), ('r4-aes-user-password.pdf', True), ('r5-empty-password.pdf', True), ('r5-user-password.pdf', True), ('r5-owner-password.pdf', True), ('r6-empty-password.pdf', True), ('r6-user-password.pdf', True), ('r6-owner-password.pdf', True)])
def test_encryption(name, requires_aes):
    if False:
        print('Hello World!')
    '\n    Encrypted PDFs are handled correctly.\n\n    This test function ensures that:\n    - If PyCryptodome or cryptography is not available and required, a DependencyError is raised\n    - Encrypted PDFs are identified correctly\n    - Decryption works for encrypted PDFs\n    - Metadata is properly extracted from the decrypted PDF\n    '
    inputfile = RESOURCE_ROOT / 'encryption' / name
    if requires_aes and (not HAS_AES):
        with pytest.raises(DependencyError) as exc:
            ipdf = pypdf.PdfReader(inputfile)
            ipdf.decrypt('asdfzxcv')
            dd = dict(ipdf.metadata)
        assert exc.value.args[0] == _DEPENDENCY_ERROR_STR
        return
    else:
        ipdf = pypdf.PdfReader(inputfile)
        if str(inputfile).endswith('unencrypted.pdf'):
            assert not ipdf.is_encrypted
        else:
            assert ipdf.is_encrypted
            ipdf.decrypt('asdfzxcv')
        assert len(ipdf.pages) == 1
        dd = dict(ipdf.metadata)
    dd = {x[0]: x[1] for x in dd.items() if x[1]}
    assert dd == {'/Author': 'cheng', '/CreationDate': "D:20220414132421+05'24'", '/Creator': 'WPS Writer', '/ModDate': "D:20220414132421+05'24'", '/SourceModified': "D:20220414132421+05'24'", '/Trapped': '/False'}

@pytest.mark.parametrize(('name', 'user_passwd', 'owner_passwd'), [('r6-both-passwords.pdf', 'foo', 'bar')])
@pytest.mark.skipif(not HAS_AES, reason='No AES implementation')
def test_pdf_with_both_passwords(name, user_passwd, owner_passwd):
    if False:
        i = 10
        return i + 15
    '\n    PDFs with both user and owner passwords are handled correctly.\n\n    This test function ensures that:\n    - Encrypted PDFs with both user and owner passwords are identified correctly\n    - Decryption works for both user and owner passwords\n    - The correct password type is returned after decryption\n    - The number of pages is correctly identified after decryption\n    '
    inputfile = RESOURCE_ROOT / 'encryption' / name
    ipdf = pypdf.PdfReader(inputfile)
    assert ipdf.is_encrypted
    assert ipdf.decrypt(user_passwd) == PasswordType.USER_PASSWORD
    assert ipdf.decrypt(owner_passwd) == PasswordType.OWNER_PASSWORD
    assert len(ipdf.pages) == 1

@pytest.mark.parametrize(('pdffile', 'password'), [('crazyones-encrypted-256.pdf', 'password'), ('crazyones-encrypted-256.pdf', b'password')])
@pytest.mark.skipif(not HAS_AES, reason='No AES implementation')
def test_read_page_from_encrypted_file_aes_256(pdffile, password):
    if False:
        for i in range(10):
            print('nop')
    '\n    A page can be read from an encrypted.\n\n    This is a regression test for issue 327:\n    IndexError for get_page() of decrypted file\n    '
    path = RESOURCE_ROOT / pdffile
    pypdf.PdfReader(path, password=password).pages[0]

@pytest.mark.parametrize('names', [['unencrypted.pdf', 'r3-user-password.pdf', 'r4-aes-user-password.pdf', 'r5-user-password.pdf']])
@pytest.mark.skipif(not HAS_AES, reason='No AES implementation')
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_merge_encrypted_pdfs(names):
    if False:
        i = 10
        return i + 15
    'Encrypted PDFs can be merged after decryption.'
    merger = pypdf.PdfMerger()
    files = [RESOURCE_ROOT / 'encryption' / x for x in names]
    pdfs = [pypdf.PdfReader(x) for x in files]
    for pdf in pdfs:
        if pdf.is_encrypted:
            pdf.decrypt('asdfzxcv')
        merger.append(pdf)
    merger.close()

@pytest.mark.skipif(USE_CRYPTOGRAPHY, reason='Limitations of cryptography. see https://github.com/pyca/cryptography/issues/2494')
@pytest.mark.parametrize('cryptcls', [CryptRC4])
def test_encrypt_decrypt_with_cipher_class(cryptcls):
    if False:
        for i in range(10):
            print('nop')
    'Encryption and decryption using a cipher class work as expected.'
    message = b'Hello World'
    key = bytes((0 for _ in range(128)))
    crypt = cryptcls(key)
    assert crypt.decrypt(crypt.encrypt(message)) == message

def test_attempt_decrypt_unencrypted_pdf():
    if False:
        i = 10
        return i + 15
    'Attempting to decrypt an unencrypted PDF raises a PdfReadError.'
    path = RESOURCE_ROOT / 'crazyones.pdf'
    with pytest.raises(PdfReadError) as exc:
        PdfReader(path, password='nonexistant')
    assert exc.value.args[0] == 'Not encrypted file'

@pytest.mark.skipif(not HAS_AES, reason='No AES implementation')
def test_alg_v5_generate_values():
    if False:
        print('Hello World!')
    '\n    Algorithm V5 values are generated without raising exceptions.\n\n    This test function checks if there is an exception during the value generation.\n    It does not verify that the content is correct.\n    '
    key = b'0123456789123451'
    values = AlgV5.generate_values(R=5, user_password=b'foo', owner_password=b'bar', key=key, p=0, metadata_encrypted=True)
    assert values == {'/U': values['/U'], '/UE': values['/UE'], '/O': values['/O'], '/OE': values['/OE'], '/Perms': values['/Perms']}

@pytest.mark.parametrize(('alg', 'requires_aes'), [('RC4-40', False), ('RC4-128', False), ('AES-128', True), ('AES-256-R5', True), ('AES-256', True), ('ABCD', False)])
def test_pdf_encrypt(pdf_file_path, alg, requires_aes):
    if False:
        i = 10
        return i + 15
    user_password = secrets.token_urlsafe(10)
    owner_password = secrets.token_urlsafe(10)
    reader = PdfReader(RESOURCE_ROOT / 'encryption' / 'unencrypted.pdf')
    page = reader.pages[0]
    text0 = page.extract_text()
    writer = PdfWriter()
    writer.add_page(page)
    if alg == 'ABCD':
        with pytest.raises(ValueError) as exc:
            writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm=alg)
        assert exc.value.args[0] == "algorithm 'ABCD' NOT supported"
        return
    if requires_aes and (not HAS_AES):
        with pytest.raises(DependencyError) as exc:
            writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm=alg)
            with open(pdf_file_path, 'wb') as output_stream:
                writer.write(output_stream)
        assert exc.value.args[0] == _DEPENDENCY_ERROR_STR
        return
    writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm=alg)
    with open(pdf_file_path, 'wb') as output_stream:
        writer.write(output_stream)
    reader = PdfReader(pdf_file_path)
    assert reader.is_encrypted
    assert reader.decrypt(owner_password) == PasswordType.OWNER_PASSWORD
    assert reader.decrypt(user_password) == PasswordType.USER_PASSWORD
    page = reader.pages[0]
    text1 = page.extract_text()
    assert text0 == text1

@pytest.mark.parametrize('count', [1, 2, 3, 4, 5, 10])
def test_pdf_encrypt_multiple(pdf_file_path, count):
    if False:
        return 10
    user_password = secrets.token_urlsafe(10)
    owner_password = secrets.token_urlsafe(10)
    reader = PdfReader(RESOURCE_ROOT / 'encryption' / 'unencrypted.pdf')
    page = reader.pages[0]
    text0 = page.extract_text()
    writer = PdfWriter()
    writer.add_page(page)
    if count == 1:
        owner_password = None
    for _i in range(count):
        writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm='RC4-128')
    with open(pdf_file_path, 'wb') as output_stream:
        writer.write(output_stream)
    reader = PdfReader(pdf_file_path)
    assert reader.is_encrypted
    if owner_password is None:
        assert reader.decrypt(user_password) == PasswordType.OWNER_PASSWORD
    else:
        assert reader.decrypt(owner_password) == PasswordType.OWNER_PASSWORD
        assert reader.decrypt(user_password) == PasswordType.USER_PASSWORD
    page = reader.pages[0]
    text1 = page.extract_text()
    assert text0 == text1

@pytest.mark.skipif(not HAS_AES, reason='No AES implementation')
def test_aes_decrypt_corrupted_data():
    if False:
        while True:
            i = 10
    'Just for robustness'
    aes = CryptAES(secrets.token_bytes(16))
    for num in [0, 17, 32]:
        aes.decrypt(secrets.token_bytes(num))

def test_encrypt_stream_dictionary(pdf_file_path):
    if False:
        while True:
            i = 10
    user_password = secrets.token_urlsafe(10)
    reader = PdfReader(SAMPLE_ROOT / '023-cmyk-image/cmyk-image.pdf')
    page = reader.pages[0]
    original_image_obj = reader.get_object(page.images['/I'].indirect_reference)
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.encrypt(user_password=user_password, owner_password=None, algorithm='RC4-128')
    with open(pdf_file_path, 'wb') as output_stream:
        writer.write(output_stream)
    reader = PdfReader(pdf_file_path)
    assert reader.is_encrypted
    assert reader.decrypt(user_password) == PasswordType.OWNER_PASSWORD
    page = reader.pages[0]
    decrypted_image_obj = reader.get_object(page.images['/I'].indirect_reference)
    assert decrypted_image_obj['/ColorSpace'][3] == original_image_obj['/ColorSpace'][3]