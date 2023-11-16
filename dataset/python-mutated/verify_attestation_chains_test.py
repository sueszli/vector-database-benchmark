import tempfile
from cryptography import x509
from cryptography.hazmat import backends
import pytest
import verify_attestation_chains
TEST_MANUFACTURER_ROOT = b'-----BEGIN CERTIFICATE-----\nMIIDujCCAqKgAwIBAgIJAMs+bXbmbsuPMA0GCSqGSIb3DQEBCwUAMHIxCzAJBgNV\nBAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX\naWRnaXRzIFB0eSBMdGQxKzApBgNVBAMMIlRlc3QgTWFudWZhY3R1cmVyIFJvb3Qg\nQ2VydGlmaWNhdGUwHhcNMjAwNDA4MTMzNDI1WhcNMzAwNDA2MTMzNDI1WjByMQsw\nCQYDVQQGEwJBVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJu\nZXQgV2lkZ2l0cyBQdHkgTHRkMSswKQYDVQQDDCJUZXN0IE1hbnVmYWN0dXJlciBS\nb290IENlcnRpZmljYXRlMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA\n242hKtNxBY2TLzyjIzFJtCf44WKha2A3KcfZ2Ul7Q/f6gAlLOK5/GvIsdB8MK/Y0\nJgJBYUfPIZ8h0gLJVLhopopc4oOTRexCNCca97klSQCTZT4+wGqf/uIEF3PoL2Bb\nuLhCQgxu+pPhfweBtEuqVcA33DYNN77J0f5KLKTHpOFEh1S1Q6ee/oRapj6J0hw6\na/7FW1R7329V5Nr9qRIzhnlpy2lBFfCV95yukAf9pUp1jCVkesugrkFY3U8np4Kn\nKhO4x1jHTsTmfq/oCUzj/hiUtD8vlg//0ZL0SMFDCANM+shi/AbyWU1BCESLrKTa\nkvI+atplTmLk8o6lOb+hMwIDAQABo1MwUTAdBgNVHQ4EFgQUlDYZuNfW4l4XDGiA\nyvJjmAiTtR8wHwYDVR0jBBgwFoAUlDYZuNfW4l4XDGiAyvJjmAiTtR8wDwYDVR0T\nAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEADyDx4HpwOpDNnG+Mf4DiSFAk\nujYaWsQGB8hUXKuJLsLLtcPNTvwSa/umu/IhBVYop1GpTadAG5yjYULwPnM4GIt3\nPOffWAptw/a0XtFRxCXtvGlb+KvwQfYN8kq3weHcSvxTkrzgeWxs0LIQrZoaBPoo\nMTJ88/s/p0d9Cf//5ukrW3LjF8OD8hd6HLpEie2qKc0wb6NXAuNgZ9m62kHSjXRS\nBd7Bwm5ZX3cOSz9SSseJKxEvD3lYUIF9w7gOeuifEpq2cfdT/VoiSL4GdN9wQ84R\n0lM4loNrim85zL8YJdGAMlAQ5gbo9/Y8khSmoUOHHoV6P4UybOj+HEhDObhpQw==\n-----END CERTIFICATE-----'
TEST_MANUFACTURER_SUBJECT_BYTES = b'0r1\x0b0\t\x06\x03U\x04\x06\x13\x02AU1\x130\x11\x06\x03U\x04\x08\x0c\nSome-State1!0\x1f\x06\x03U\x04\n\x0c\x18Internet Widgits Pty Ltd1+0)\x06\x03U\x04\x03\x0c"Test Manufacturer Root Certificate'
TEST_MANUFACTURER_CHAIN = b'-----BEGIN CERTIFICATE-----\nMIIDSzCCAjMCCQDectxvPIHUkzANBgkqhkiG9w0BAQsFADBlMQswCQYDVQQGEwJB\nVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJuZXQgV2lkZ2l0\ncyBQdHkgTHRkMR4wHAYDVQQDDBVUZXN0IENhcmQgQ2VydGlmaWNhdGUwHhcNMjAw\nNDA4MTQ1NDAzWhcNMjEwNDA4MTQ1NDAzWjBqMQswCQYDVQQGEwJBVTETMBEGA1UE\nCAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJuZXQgV2lkZ2l0cyBQdHkgTHRk\nMSMwIQYDVQQDDBpUZXN0IFBhcnRpdGlvbiBDZXJ0aWZpY2F0ZTCCASIwDQYJKoZI\nhvcNAQEBBQADggEPADCCAQoCggEBAOKNNkMQXNHO9DnYcD+U8Ll/v9Z4v7oVJ2xV\nYlDMMj2IJzhfZI77miii/ll3UZj3EDGC4y4pdS0L81988WWaQ4yVEZtgV5+WHLGr\nBb08Ex5qKRJ5ag2dI/Sz6+M9+5pI1wQ2TscqpFTIjYmBOB91CK96JJOKKtuPcDC1\n31guMzTBpm3WiYyJBhR4Xj1McOwFLGBoPuZl9N8CzbqofVG1aTZi/C+AedU4YMjM\nlKnB/Qxv94w6tsNPgbkjUGl3ZqmyUEKOG7zlIatnuXs70QEuv+KIouuFOTGiQppE\nQaTNZ7UO0nhEG3e+cXyUzHxzlf8RTJpmhwqEHGnjF6YTsjVipGkCAwEAATANBgkq\nhkiG9w0BAQsFAAOCAQEAD0rrWYZ+++ib6vvYi37j7hbwTuCRi1CSxXsrgiBFM1YK\ncGzUnmqDXvJiBHX/qDIlkPxJy4AcfJ29r2dQ6kq1nwyaCAjqbGiFNR+VjNo4oAih\n/xi1O1tIIZ4j979NAZmozsOXScYV1jVM/chM5QibDvdMp71YN8HwdMfGhssQe4sf\nKmu1IXq8XY/b0f/k3q5URgM+ur0qBJbqumY6fYgAYrfhSv7v4YlH2yqPVMWUOpHt\nxFUKVbiXLux9xCsZ948b+lZLlETudUES5MpjHarlrr/CWxibjk6RJomN3eHHauZs\n2ccag9KEQnjsBP2N2wqJeHU902attwvOoWQokUp9/Q==\n-----END CERTIFICATE-----\n-----BEGIN CERTIFICATE-----\nMIIDUzCCAjsCCQDNU9BSQM85jTANBgkqhkiG9w0BAQsFADByMQswCQYDVQQGEwJB\nVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJuZXQgV2lkZ2l0\ncyBQdHkgTHRkMSswKQYDVQQDDCJUZXN0IE1hbnVmYWN0dXJlciBSb290IENlcnRp\nZmljYXRlMB4XDTIwMDQwODE0NTQwNloXDTIxMDQwODE0NTQwNlowZTELMAkGA1UE\nBhMCQVUxEzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoMGEludGVybmV0IFdp\nZGdpdHMgUHR5IEx0ZDEeMBwGA1UEAwwVVGVzdCBDYXJkIENlcnRpZmljYXRlMIIB\nIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAmVDLbIEpoVx9IzEalubKQuer\n1iOI3c59gQDa9V6+iFBjmeOeE2vI97pojTAvWMWkRRKARklY3BruVKR5698yTLzb\ncPwg3vqPvGNQFztJzaYAwRoerdT289upoEADdjQmA2Y7PF3zH88nLh74B9M5O8/t\njCfOWehl/ctUhTnIRCwEs7ZLMc3HHhG+puymhruD+hLNcgWfX2aizDEW8wpNZ6dd\nJhGyA/OPXgFOhpqOZd/BFM/9w8rSBHfijitdNRK5UQj6TtRBykBj9ZvBzapRbLS/\nrrqnSoXFo6dlZaxpJKXT/bFvx1ImZSizJln2klYcKv30g4lg+U4PaHvxNOFdrQID\nAQABMA0GCSqGSIb3DQEBCwUAA4IBAQCtCrCZFLDYjEcKoMwbHa7XyELcQ79qM+YV\ni2UBYZqvoq9ty0cLa1O1pvF35qbG1Z8B7dZ3rKwiXsBnFk87rMaweWvQWZkZEs3T\nu4hBRdSrDCb2SEIEOusI49fZVhbZbgqnNX1AkFe9lqNVu82pYKpNSkP13cATVghB\nmrDi6mxjdeiTHiek+ND05YEdGi38ja5T3/0PJjyEj35WXT+7CP95MArfNc3rhy6J\nwkovm2tDYaojIJJtgFcBP3yhzsJOyHCkEzMqcOSjoT9pDv5qvrHKDMTZRKjcOgKP\nYlIjHK237cZfoEh1PO2OI14sM2iD3ZpD5idGZI/GHF6GUWZ2AJqM\n-----END CERTIFICATE-----'
TEST_OWNER_ROOT = b'-----BEGIN CERTIFICATE-----\nMIIDrDCCApSgAwIBAgIJAJdbRMhyDhf+MA0GCSqGSIb3DQEBCwUAMGsxCzAJBgNV\nBAYTAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX\naWRnaXRzIFB0eSBMdGQxJDAiBgNVBAMMG1Rlc3QgT3duZXIgUm9vdCBDZXJ0aWZp\nY2F0ZTAeFw0yMDA0MDgxMzM3MDVaFw0zMDA0MDYxMzM3MDVaMGsxCzAJBgNVBAYT\nAkFVMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBXaWRn\naXRzIFB0eSBMdGQxJDAiBgNVBAMMG1Rlc3QgT3duZXIgUm9vdCBDZXJ0aWZpY2F0\nZTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANwhPaA1nzlVrl1sFThl\nGmwNSVczMmIVXOUyklTI2HIW7iHXWqqyusksQZJBvWpYLom0CzEv42wYWLtx5U9S\nKE8P++DLCzspkV+KW11Gyyq5xsVxlrrcQZ6/SXDS8092IIZc/qht9Sgwv/u03tQA\nJdwOgisnKRX2wtQfSi8lT+kNiT8IG6nbc1oRGcRa0cNY9uKaElF/EHxj33quZnQ0\ntvxH7NhjxZ+GSiMbLChp3DZGnq9lcTurBFGPUe31riP5uThTiKA5DZDJadFSs8Q1\nO0W0XPyWysm5y/7pAuz4yHZPdVzj6kssklHOE9+Yk3D8Q39n0utQJ84m+IrMrj8r\nqSsCAwEAAaNTMFEwHQYDVR0OBBYEFCK8CyqtlBC06Hd9x680rN3nRJeZMB8GA1Ud\nIwQYMBaAFCK8CyqtlBC06Hd9x680rN3nRJeZMA8GA1UdEwEB/wQFMAMBAf8wDQYJ\nKoZIhvcNAQELBQADggEBAHXopvAk9qAqsW6tK9YmRy+EXCqxoOuKdXlU75G1GVnX\nObk/1DCbT2/ioXmSeA15p7fFdf5JG4J7ejCfJG8uXrKb0JW0RWRGXmGnO967FejD\nSTV8tp/uAYduPVcR9EqVDfOKU2jf0MoZnP95/dlBxZ+yTMnuusBu8z7ERPsQWve8\nyitpRkFz9nrFytZRJVJxl+bm0Llz2eqINYR3Oia7v0xtS1XaJUGX5mG2gpMlIfpu\n1ByJP8g11f9HW84eeZ9ceKU848uJtj+DTDnx4Ck1x6huMvZkxOAVTNWmT1+osDv7\nvsVkjBnGwXfcAv6jFQiErcdpZ1MVdLxsAFHrAvYH67E=\n-----END CERTIFICATE-----'
TEST_OWNER_CARD_CHAIN = b'-----BEGIN CERTIFICATE-----\nMIIDTDCCAjQCCQDp+9ouYgrR0DANBgkqhkiG9w0BAQsFADBrMQswCQYDVQQGEwJB\nVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJuZXQgV2lkZ2l0\ncyBQdHkgTHRkMSQwIgYDVQQDDBtUZXN0IE93bmVyIFJvb3QgQ2VydGlmaWNhdGUw\nHhcNMjAwNDA4MTYyMTEyWhcNMjEwNDA4MTYyMTEyWjBlMQswCQYDVQQGEwJBVTET\nMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJuZXQgV2lkZ2l0cyBQ\ndHkgTHRkMR4wHAYDVQQDDBVUZXN0IENhcmQgQ2VydGlmaWNhdGUwggEiMA0GCSqG\nSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCZUMtsgSmhXH0jMRqW5spC56vWI4jdzn2B\nANr1Xr6IUGOZ454Ta8j3umiNMC9YxaRFEoBGSVjcGu5UpHnr3zJMvNtw/CDe+o+8\nY1AXO0nNpgDBGh6t1Pbz26mgQAN2NCYDZjs8XfMfzycuHvgH0zk7z+2MJ85Z6GX9\ny1SFOchELASztksxzcceEb6m7KaGu4P6Es1yBZ9fZqLMMRbzCk1np10mEbID849e\nAU6Gmo5l38EUz/3DytIEd+KOK101ErlRCPpO1EHKQGP1m8HNqlFstL+uuqdKhcWj\np2VlrGkkpdP9sW/HUiZlKLMmWfaSVhwq/fSDiWD5Tg9oe/E04V2tAgMBAAEwDQYJ\nKoZIhvcNAQELBQADggEBAM3hz+TfQNaeuBgPyqBedN6QkhSiTdzpNG7Eyfw3Sx8n\nOSuZxcsZgNRo+WNJt4zi9cMaOwgPcuoGCW7Iw2StEtBqgujlExrfUHzu17yoBHxQ\nDTvi7QRHb6W2amsSKcuoFkI1txVmVWQA2HkSQVqIzZZoI3qVu2cQMyVHG7MKPHFU\n5Mzw0H37gfttXYnUDZM84ETpGuf7EXA7ROdgwDvDD8CqOMDBKpKqau9QVh4aBZW4\nkoGMoga+RwjNt4FVCW4F4qn43fteDSmxdUuxqCn6V7CpRlHIc8J2q9nzsne/NCA0\n2W+pXJD8hjvb9YQuXyV1QOaV6dcDLDcKG6NCdtxisxM=\n-----END CERTIFICATE-----'
TEST_OWNER_PARTITION_CHAIN = b'-----BEGIN CERTIFICATE-----\nMIIDUTCCAjkCCQDp+9ouYgrR0TANBgkqhkiG9w0BAQsFADBrMQswCQYDVQQGEwJB\nVTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJuZXQgV2lkZ2l0\ncyBQdHkgTHRkMSQwIgYDVQQDDBtUZXN0IE93bmVyIFJvb3QgQ2VydGlmaWNhdGUw\nHhcNMjAwNDA4MTYyNDQ0WhcNMjEwNDA4MTYyNDQ0WjBqMQswCQYDVQQGEwJBVTET\nMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50ZXJuZXQgV2lkZ2l0cyBQ\ndHkgTHRkMSMwIQYDVQQDDBpUZXN0IFBhcnRpdGlvbiBDZXJ0aWZpY2F0ZTCCASIw\nDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAOKNNkMQXNHO9DnYcD+U8Ll/v9Z4\nv7oVJ2xVYlDMMj2IJzhfZI77miii/ll3UZj3EDGC4y4pdS0L81988WWaQ4yVEZtg\nV5+WHLGrBb08Ex5qKRJ5ag2dI/Sz6+M9+5pI1wQ2TscqpFTIjYmBOB91CK96JJOK\nKtuPcDC131guMzTBpm3WiYyJBhR4Xj1McOwFLGBoPuZl9N8CzbqofVG1aTZi/C+A\nedU4YMjMlKnB/Qxv94w6tsNPgbkjUGl3ZqmyUEKOG7zlIatnuXs70QEuv+KIouuF\nOTGiQppEQaTNZ7UO0nhEG3e+cXyUzHxzlf8RTJpmhwqEHGnjF6YTsjVipGkCAwEA\nATANBgkqhkiG9w0BAQsFAAOCAQEARAcOL0Y1XBYqIT/ySRFmN6f+cLUO3sPGllt8\nBLYhHsnyJjzV4By00GLcLp5cd14sIfkyVPPFA7kCwOGbL52avRo6+bPB2Bdg6MRt\nwbXG3pa6DQXQ2vsZ0I1bXUdCS05YbfvfTrF7LLNCPYbul4Wph5zT360JmBVLKPaX\nsmw8fAPhdDwHix1ee2bopldrPrS0L55t3HLOBEF4XT9TCXFS23yBpujGsLEwHgaJ\nx1un6v2v+bEPr8tpPv33WlSC9Fwlit0Xwf6sp/YuX11t223D7QmGN8rRyqv9Fm5l\nRZh2a6rJjlnErdcLx/+5ojsCjbElfluJvsToc+iCcwut6FKcPg==\n-----END CERTIFICATE-----'
TEST_ATTESTATION = b'\x1f\x8b\x08\x08\xb9\xe37`\x00\x03attestation.dat\x00\x01\x08\x01\xf7\xfecontent\n>\xede\xeb\xd3\x9d\x9e\x8e*\xa2\xf4\x04i\xec\x10lI\xa1\xc5\xd6\x0c\xfd\x1a^T\x1f&>f\xb2\xae}UD\xf1\xbaW\xcf\xec\xc5\x10\x86s\x92A\xa1E\xc3\xf9=8/\xe5\xf4y\xf1\xa4H\xb1"\x08\xe7\x1a\xd1[\xc2\xb1CCO\x82\xe7-\xbd-=u\x15\x9a=\x1b\x98\xec\xb6\x1d\xc0\xd2\xf7\xcb\x99g\xdd\xed\xba\xcb\x9bK\xc7\xd8[\xd9\xf8?K\x0f\xd5\xaaO\xd4R0\xf6>\x18\xb2F\x13 \xedi\xedV\xdc\x1bR-j\x85\xe3\xd5\x92\x9a\x9dU4\xc8\x13\xa10\xbbg\xee\xa3R\x8a\xcf\x88\x91p\xde\x9c\xe1\x82\xcd\x8a>\xa0\x1c\xf2\xb5\xb2\xb2\x91.Z\t\xc8a{\x896\x03+|\x8b\xa0\xb7\x16\xe8E\xca\x0c+\x17)\xd4\xd2s\x96\xfen\xa9\xf7\xa2\x1eW\xd3\xbd\n\x16\x12\xea8\xaa\x85xJ\x13d\xe5\x85\xdd\xe6\xca\x82;qw\xbe\x8fa\xb7\xeb\x06L\xd2\\pb\x0b\xbf\x9bj\xcc\xb0\x92\xf2\x81v\x1c\xa0\xa3?{~\x8e\xc1O\x1a\xc9\x7f\x9cCH\x1d\xef\x85\xe1\xeb\xa5\x08\x01\x00\x00'

def test_verify_certificate():
    if False:
        for i in range(10):
            print('nop')
    signing_cert = x509.load_pem_x509_certificate(TEST_OWNER_ROOT, backends.default_backend())
    issued_cert = x509.load_pem_x509_certificate(TEST_OWNER_PARTITION_CHAIN, backends.default_backend())
    assert verify_attestation_chains.verify_certificate(signing_cert, issued_cert)

def test_verify_certificate_fail():
    if False:
        while True:
            i = 10
    signing_cert = x509.load_pem_x509_certificate(TEST_OWNER_ROOT, backends.default_backend())
    issued_cert = x509.load_pem_x509_certificate(TEST_MANUFACTURER_ROOT, backends.default_backend())
    assert not verify_attestation_chains.verify_certificate(signing_cert, issued_cert)

def get_test_manufacturer_root():
    if False:
        for i in range(10):
            print('nop')
    return x509.load_pem_x509_certificate(TEST_MANUFACTURER_ROOT, backends.default_backend())

def get_test_owner_root():
    if False:
        return 10
    return x509.load_pem_x509_certificate(TEST_OWNER_ROOT, backends.default_backend())

def make_temporary_file(contents):
    if False:
        print('Hello World!')
    'Creates a NamedTemporaryFile with contents and returns its file name.\n\n    Args:\n        contents: The contents to write to the temporary file.\n\n    Returns:\n        The name of the temporary file.\n    '
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(contents)
    temp_file.close()
    return temp_file.name

@pytest.fixture(scope='function')
def test_data():
    if False:
        i = 10
        return i + 15
    mfr_root = make_temporary_file(TEST_MANUFACTURER_ROOT)
    mfr_chain = make_temporary_file(TEST_MANUFACTURER_CHAIN)
    owner_root = make_temporary_file(TEST_OWNER_ROOT)
    owner_card_chain = make_temporary_file(TEST_OWNER_CARD_CHAIN)
    owner_partition_chain = make_temporary_file(TEST_OWNER_PARTITION_CHAIN)
    cert_chains = make_temporary_file(b'\n'.join([TEST_MANUFACTURER_CHAIN, TEST_OWNER_CARD_CHAIN, TEST_OWNER_PARTITION_CHAIN]))
    attestation = make_temporary_file(TEST_ATTESTATION)
    param = {'mfr_root': mfr_root, 'mfr_chain': mfr_chain, 'owner_root': owner_root, 'owner_card_chain': owner_card_chain, 'owner_partition_chain': owner_partition_chain, 'cert_chains': cert_chains, 'attestation': attestation}
    yield param

def test_verify(monkeypatch, test_data):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(verify_attestation_chains, 'MANUFACTURER_CERT_SUBJECT_BYTES', TEST_MANUFACTURER_SUBJECT_BYTES)
    monkeypatch.setattr(verify_attestation_chains, 'get_manufacturer_root_certificate', get_test_manufacturer_root)
    monkeypatch.setattr(verify_attestation_chains, 'get_owner_root_certificate', get_test_owner_root)
    assert verify_attestation_chains.verify(test_data['cert_chains'], test_data['attestation'])

def test_verify_invalid_mfr_root_fails(monkeypatch, test_data):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(verify_attestation_chains, 'MANUFACTURER_CERT_SUBJECT_BYTES', b'invalid')
    monkeypatch.setattr(verify_attestation_chains, 'get_manufacturer_root_certificate', get_test_owner_root)
    monkeypatch.setattr(verify_attestation_chains, 'get_owner_root_certificate', get_test_owner_root)
    assert not verify_attestation_chains.verify(test_data['cert_chains'], test_data['attestation'])