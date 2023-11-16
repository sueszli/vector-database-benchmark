"""
Check PDF encryption:
* make a PDF with owber and user passwords
* open and decrypt as owner or user
"""
import fitz

def test_encryption():
    if False:
        while True:
            i = 10
    text = 'some secret information'
    perm = int(fitz.PDF_PERM_ACCESSIBILITY | fitz.PDF_PERM_PRINT | fitz.PDF_PERM_COPY | fitz.PDF_PERM_ANNOTATE)
    owner_pass = 'owner'
    user_pass = 'user'
    encrypt_meth = fitz.PDF_ENCRYPT_AES_256
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    tobytes = doc.tobytes(encryption=encrypt_meth, owner_pw=owner_pass, user_pw=user_pass, permissions=perm)
    doc.close()
    doc = fitz.open('pdf', tobytes)
    assert doc.needs_pass
    assert doc.is_encrypted
    rc = doc.authenticate('owner')
    assert rc == 4
    assert not doc.is_encrypted
    doc.close()
    doc = fitz.open('pdf', tobytes)
    rc = doc.authenticate('user')
    assert rc == 2