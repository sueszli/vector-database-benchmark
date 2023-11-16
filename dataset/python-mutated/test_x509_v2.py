import base64
import datetime
import pytest
import salt.exceptions
from salt.utils.odict import OrderedDict
try:
    import cryptography
    import cryptography.x509 as cx509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, pkcs7, pkcs12
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
CRYPTOGRAPHY_VERSION = tuple((int(x) for x in cryptography.__version__.split('.')))
pytestmark = [pytest.mark.skipif(HAS_LIBS is False, reason='Needs cryptography library')]

@pytest.fixture(scope='module')
def minion_config_overrides():
    if False:
        while True:
            i = 10
    return {'x509_signing_policies': {'testpolicy': {'CN': 'from_signing_policy', 'basicConstraints': 'critical, CA:FALSE', 'keyUsage': 'critical, cRLSign, keyCertSign', 'authorityKeyIdentifier': 'keyid:always', 'subjectKeyIdentifier': 'hash'}, 'testsubjectstrpolicy': {'subject': 'CN=from_signing_policy'}, 'testsubjectdictpolicy': {'subject': {'CN': 'from_signing_policy'}}, 'testsubjectlistpolicy': {'subject': ['C=US', 'L=Salt Lake City', 'O=Salt Test']}, 'testnosubjectpolicy': {'basicConstraints': 'critical, CA:FALSE'}, 'testdeprecatednamepolicy': {'commonName': 'deprecated'}, 'testdeprecatedextpolicy': {'X509v3 Basic Constraints': 'critical CA:FALSE'}}, 'features': {'x509_v2': True}}

@pytest.fixture
def x509(loaders, modules):
    if False:
        i = 10
        return i + 15
    yield modules.x509

@pytest.fixture
def ca_cert():
    if False:
        while True:
            i = 10
    return '-----BEGIN CERTIFICATE-----\nMIIDODCCAiCgAwIBAgIIbfpgqP0VGPgwDQYJKoZIhvcNAQELBQAwKzELMAkGA1UE\nBhMCVVMxDTALBgNVBAMMBFRlc3QxDTALBgNVBAoMBFNhbHQwHhcNMjIxMTE1MTQw\nNDMzWhcNMzIxMTEyMTQwNDMzWjArMQswCQYDVQQGEwJVUzENMAsGA1UEAwwEVGVz\ndDENMAsGA1UECgwEU2FsdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEB\nAOGTScvrjcEt6vsJcG9RUp6fKaDNDWZnJET0omanK9ZwaoGpJPp8UDYe/8ADeI7N\n10wdyB4oDM9gRDjInBtdQO/PsrmKZF6LzqVFgLMxu2up+PHMi9z6B2P4esIAzMu9\nPYxc9zH4HzLImHqscVD2HCabsjp9X134Af7hVY5NN/W/4qTP7uOM20wSG2TPI6+B\ntA9VyPbEPMPRzXzrqc45rVYe6kb2bT84GE93Vcu/e5JZ/k2AKD8Hoa2cxLPsTLq5\nigl+D+k+dfUtiABiKPvVQiYBsD1fyHDn2m7B6pCgvrGqHjsoAKufgFnXy6PJRg7n\nvQfaxSiusM5s+VS+fjlvgwsCAwEAAaNgMF4wDwYDVR0TBAgwBgEB/wIBATALBgNV\nHQ8EBAMCAQYwHQYDVR0OBBYEFFzy8fRTKSOe7kBakqO0Ki71potnMB8GA1UdIwQY\nMBaAFFzy8fRTKSOe7kBakqO0Ki71potnMA0GCSqGSIb3DQEBCwUAA4IBAQBZS4MP\nfXYPoGZ66seM+0eikScZHirbRe8vHxHkujnTBUjQITKm86WeQgeBCD2pobgBGZtt\n5YFozM4cERqY7/1BdemUxFvPmMFFznt0TM5w+DfGWVK8un6SYwHnmBbnkWgX4Srm\nGsL0HHWxVXkGnFGFk6Sbo3vnN7CpkpQTWFqeQQ5rHOw91pt7KnNZwc6I3ZjrCUHJ\n+UmKKrga16a4Q+8FBpYdphQU609npo/0zuaE6FyiJYlW3tG+mlbbNgzY/+eUaxt2\n9Bp9mtA+Hkox551Mfpq45Oi+ehwMt0xjZCjuFCM78oiUdHCGO+EmcT7ogiYALiOF\nLN1w5sybsYwIw6QN\n-----END CERTIFICATE-----\n'

@pytest.fixture
def ca_key():
    if False:
        return 10
    return '-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA4ZNJy+uNwS3q+wlwb1FSnp8poM0NZmckRPSiZqcr1nBqgakk\n+nxQNh7/wAN4js3XTB3IHigMz2BEOMicG11A78+yuYpkXovOpUWAszG7a6n48cyL\n3PoHY/h6wgDMy709jFz3MfgfMsiYeqxxUPYcJpuyOn1fXfgB/uFVjk039b/ipM/u\n44zbTBIbZM8jr4G0D1XI9sQ8w9HNfOupzjmtVh7qRvZtPzgYT3dVy797kln+TYAo\nPwehrZzEs+xMurmKCX4P6T519S2IAGIo+9VCJgGwPV/IcOfabsHqkKC+saoeOygA\nq5+AWdfLo8lGDue9B9rFKK6wzmz5VL5+OW+DCwIDAQABAoIBAFfImc9hu6iR1gAb\njEXFwAE6r1iEc9KGEPdEvG52X/jzhn8u89UGy7BEIAL5VtE8Caz1agtSSqnpLKNs\nblO31q18hnDuCmFAxwpKIeuaTvV3EAoJL+Su6HFfIWaeKRSgcHNPOmOXy4xXw/75\nXJ/FJu9fZ9ybLaHEAgLObh0Sr9RSPQbZ72ZawPP8+5WCbR+2w90RApHXQL0piSbW\nlIx1NE6o5wQb3vik8z/k5FqLCY2a8++WNyfvS+WWFY5WXGI7ZiDDQk46gnslquH2\nLon5CEn3JlTGQFhxaaa2ivssscf2lA2Rvm2E8o1rdZJS2OpSE0ai4TXY9XnyjZj1\n5usWIwECgYEA+3Mwu03A7PyLEBksS/u3MSo/176S9lF/uXcecQNdhAIalUZ8AgV3\n7HP2yI9ZC0ekA809ZzFjGFostXm9VfUOEZ549jLOMzvBtCdaI0aBUE8icu52fX4r\nfT2NY6hYgz5/fxD8sq1XH/fqNNexABwtViH6YAly/9A1/8M3BOWt72UCgYEA5ag8\nsIfiBUoWd1sS6qHDuugWlpx4ZWYC/59XEJyCN2wioP8qFji/aNZxF1wLfyQe/zaa\nYBFusjsBnSfBU1p4UKCRHWQ9/CnC0DzqTkyKC4Fv8GuxgywNm5W9gPKk7idHP7mw\ne+7Uvf1pOQccqEPh7yltpW+Xw27gfsC2DMAIGa8CgYByv/q5P56PiCCeVB6W/mR3\nl2RTPLEsn7y+EtJdmL+QgrVG8kedVImJ6tHwbRqhvyvmYD9pXGxwrJZCqy/wjkjB\nWaSyFjVrxBV99Yd5Ga/hyntaH+ELHA0UtoZTuHvMSTU9866ei+R6vlSvkM9B0ZoO\n+KqeMTG99HLwKVJudbKO0QKBgQCd33U49XBOqoufKSBr4yAmUH2Ws6GgMuxExUiY\nxr5NUyzK+B36gLA0ZZYAtOnCURZt4x9kgxdRtnZ5jma74ilrY7XeOpbRzfN6KyX3\nBW6wUh6da6rvvUztc5Z+Gk9+18mG6SOFTr04jgfTiCwPD/s06YnSfFAbrRDukZOU\nWD45SQKBgBvjSwl3AbPoJnRjZjGuCUMKQKrLm30xCeorxasu+di/4YV5Yd8VUjaO\nmYyqXW6bQndKLuXT+AXtCd/Xt2sI96z8mc0G5fImDUxQjMUuS3RyQK357cEOu8Zy\nHdI7Pfaf/l0HozAw/Al+LXbpmSBdfmz0U/EGAKRqXMW5+vQ7XHXD\n-----END RSA PRIVATE KEY-----'

@pytest.fixture
def ca_key_enc():
    if False:
        return 10
    return '-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFLTBXBgkqhkiG9w0BBQ0wSjApBgkqhkiG9w0BBQwwHAQIy/O+FhcKBKUCAggA\nMAwGCCqGSIb3DQIJBQAwHQYJYIZIAWUDBAEqBBDtSfZzKh7brkHFw/s6bcbVBIIE\n0JcLyycDhdSPzL7Zm1+ZLavjxiuaGEaHU8hu8ZScqyjcdWbdOfOuqZgu7OzxwfIc\n8Q1bfqMGUfxPcs/JQh13CVOaDYmafeMZYN3rqsNoci11iaHDhTAqgYCM2iVXaFUt\n6ZdfW+/hEk+yHwK5K2R1/ks8buAe0OgjkV0N3DqAif93BPyFP6XT7btVMrorGJjh\n1OJjuw3q0xJ02rn7O5imaZ5NnCIDShkKwWO6sUew3QHhW61/nuCBPyJTsAO0L4+t\n9zjb2jOIIuvTpZUhAty6I+bKgaYLhsii7z5jVYpt+NbYpzIe+9RvAD1psGk9+bGD\nrN70Bnhx29mPEKdmozXVQ8GTqDOSQSYMr9aax+BhSJoTnCtVtGGX0LXE5Dvd/HHy\n+Yw2HFrVglptsPYo4EBKccC3FJlS0mL6yBW5NCpU7MOhDV/iOMbzM4bqwKG+jqaw\nsjIScCg+ljBxGhNrcMa0AEBWukTRe4gERpb8AyGKYOSVN6iZyP5qhN/Abu1asKrj\nc4NRUu3yILleZuxjkDd4w0CwhjlCaKFLsp1XeFE5ZHM5Iezi1/I4QMXFTydB1KnX\nxOSofZ7b7pnvOiBQG2nQzYSjSnBO7E7NQOhjvkRgcxNsdAQWADIqdE3bKZ8qcEZ6\nq1TE0XtcDgwFGwQF/cyuEEXyMAkQV687e8IdCjc+MbyyqUtQA9382JyjOuzavvMD\nnO5s80lB5aa0WHdE+Rg7KiBIwL1CjBSGSiggKvkG01ObeQL4DCQG6gHgz+nvdiNe\ndu2u6hW2/PUuUIOM2ApE98T2TAzCnyu02iMIN5aH4za5y1w5YzaU4Lsl4nzAEA3c\n8EuVIWMutZnqT4ZSCLCq1AtDYkSXxIjGQPwhRslyCJuwtuiaDXLIZIpMRGqMKdGS\nc3q0k5ba92jXppIOVYN/kViNjYeHVZ3KRAi2MqUByqiMBkZo11NsgaU/uPsKsK16\nD0XueVs9EobU55tgBV71Q8g/5BiGG19W5UZVzjiiuGuj44msOfYV4027KqqFf302\nU5RXAwBko9S+v3SuTZrRXK4uuYceR9Uyco8aP/tNAhHEGa8Z73vLngZICp57qD1h\n8smjOrm1volZpu31HP9CWVh47GyuzSZ8BUFrR/uXfa+uqyLqeBKglz5SC6Ak3nL8\neAHu3EK2dVp4vqwYB2oO9DQqs4CN7DKyArNeUzKSf6ZKEYBZCdF5V5HgbSpY5f+e\nxj5cpuMVc7s+Nxv/0bqxNzt8ghe2sDELxK8lo7Q6E+aUNBWt++nHI2b8y5ynaANU\nkQjeoorrPHUScXN8TVrgrIYIfXOqkI14UmroRH5/oyORHXN25JekV1DisKZOtSdV\nVqt3o/hlGFYhaeznIgquBm27trLkLHOfCGx6M2xlKszlWBP03zFLp0PiXE+y07zC\nIwzaiVlj/O+QIsiMmrtc8WXYiNWVN5XDe1elFPs1K2cw0cIeyLgC1Bibxa7dH01G\nZ0Nr+hZN+/EqI3Tu+lWeWtj/lIhjJrKQvUOMM4W1MFZZdK09ZsCdW0Y1fFYn/3Xz\ng1KvGcFoszp0uMptlJUhsxtFooG4xKtgEITmtraRU+hTGU3NZgtk7Qff4tFa0O0h\nA62orBDc+8x+AehfwYSm11dz5/P6aL3QZf+tzr05vbVn\n-----END ENCRYPTED PRIVATE KEY-----'

@pytest.fixture
def rsa_privkey():
    if False:
        print('Hello World!')
    return '-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAzIdEbSkbPIc5F/aewNoqWPsF/YP+DByMwvYs+0K+lehc39P8\n2fL8K2GIaGMBBzqlsX6CplAzGcoEQEBLTouLm+JYC5e1zRjaml4M+Bid8A7qwdjl\nWd0enCAxVB9BMNnj/mokWzh3hAQMBXfmddGRhH0P9KXfjBNh2V09vfHdtS9XMEEQ\njN6vCxaezXqsOMN3bjRTIcUatH7dVUgUpt9cye1mdbD5KVBgJ9MArc2tJ3rmB0lx\njEbAhTEHrNnIkDOJCKE8TaQOW4RyVWlIvSEL+Ov0TPeXXaef8HJlkyIpKrCZ+c4i\n0N7qOlyrJEWTXmKNCj87xgTpY5I7ARISeOQD8QIDAQABAoIBABYNN4l1dyRNiBBX\nXMJ6QzqYxgqRYqC3q02R8UOd7KKJDQps9sQg+KNMvsJSelfnMNo0Q63e08OiDldH\nF1d+oCzMeKW3U7irR1aBcXCGZvDtCs6frgrEVnqK1ga13/d+ZqCVnRngurIXJZyp\nUsW9NK1ONpwwDiwyIsimzvNd0oOoR6ROIN2Fk+AhKQ6bPdgqLM1Swx6BA0J/aaqO\njAqSkYkGOEL970W8ZhnyyDDRcbgPbacUDo7AJnrBeqHoAqrJ1PzJ3jhcWDJl8Xcy\nuVDP1hBeK9yg4nuMcArsqrRQvqL2GuafGYygfzrU1aW96hlXciOv32ov36h2qIJU\nr4JfJGECgYEA7UPD4iRsHV6eMkD98Ev74ygdnFL2TMknqOUEboPNiQzSzr5oVrKa\nKFDhzenUNlMSoeiAaLLI7xaD4xptXuN8xx7sQZVSiEniBfJ7F+9sPNjCXwYbUuWp\nqpp6KfCrjLxDxgSKH9FUIlTvL7M4lmAD2yHn4zXjFz3BOs261JUn6l0CgYEA3K2/\nS2eP3VUL6K4+HNMzXTj9Q8S7LSYnTZVIjfek6pQHMwaMKE8EC7L4XeS9TZ49BKCS\nMh9RI2yBCX6L1uo2zURAI0oDrowDhjaUCD4xxTD27OyMcvjdSzk/+0E+DtsWdgYm\nFGX/l0zTRUsZBbc7ItTG0ksIB+aMM4njBbHubqUCgYAq9llS6pt1Gfv1R5Vz3J5o\nvIvYEaGtt8Lpr0aFKHKgPWUysIG+KSsG39ZzbcLSb2pxTONrkewWdvI8vj1NsE2Y\n1L2dBofiS9aUkxq888qanflcMYPjF9kIHl6+l2jI3BI9mfbU2hes+8ovzfkSKaKp\nHFOb7dcID1Oc7UHGWpfWtQKBgQDC3Y4xOKbaLDJS6iIg9ALETAGgqQUbzjggkzU5\nX7e6CLL+xMZZBcUty4Dz8HuVIakCAAR4zByq6jJbvuofAj0YLy9vufjcVfj6uBEp\n4jmyxhUVi6BOGiHXPhuYc7koByCjYbSYiKUU5psc8j6LRIysqjVTFzxlNZkSHa1h\npwhDnQKBgATpQou7MeAOMHjMPaNx8OCq7QNhocp8Q+goxPb0ND2jF9xSI+gjzRRt\nKpz+xO6tri6wCgWrmE5cJbEe3/EYf3bmbNA9wOQ72kfoy9uO0cCi+5gSJigwaIKM\nDYRTDIS9eg2LF4B64hZvkCLTmP4rLJWdRnWrLosIC4rD1uWgGayC\n-----END RSA PRIVATE KEY-----'

@pytest.fixture
def rsa_privkey_enc():
    if False:
        return 10
    return '-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFLTBXBgkqhkiG9w0BBQ0wSjApBgkqhkiG9w0BBQwwHAQIHU2H6hhL0gYCAggA\nMAwGCCqGSIb3DQIJBQAwHQYJYIZIAWUDBAEqBBD64PydhZIJPW9amw7M8yGvBIIE\n0LHXvvQleCJMlH/Rtml1Vx2nygReVl+1Ag+FjtsNQHtsXYkzVWSDI0zI7nFyDpb9\nKr2+9UOsOhQA5/swka9ude4oJng0YZcV4qgar8yFncWTrMTk/mrvFSNZPz9LMGsq\nin7hzYGAP6XdprHgJfw+wDQfwbwcTQp5DUOPYbhxfnggVQBL84gp/2urCcNnFX+T\nOKGm9C3NfLycrCbaQxaV/2oTo7+UHUaXKwZwY6zKxCqbwGBy7dNcZD16nJyOBmbj\nytOi/OqBcoj03yK4ETIm7EWwem6CRAbPH1GnUAxmb5tG6jzKphbMJur8n72Vv+VK\n9+Gkz5vOq1O1wlK+DfB+Xrgfx3lHHQllxi7FtlQegSFlIbHAacG/muwMRQ5PoMEp\nRaGQkxOhiU7VSaZ3Gdx3TrQMaF5nBqvs90Xw40uWdD9+Kd3Oqkj9OgiqHZwgWPfW\ntxB+jXYGj1ERUvb36T7P8IH/QDa8jwVf3+f1pOpoMe4+6i3rr9bAkDhIjpNDo2a0\nYXvVns18UisnLXHxdAZb9R2V/VoTxhs3IqK3nEb5qnb1RAtJfV4p1ENVsoPiHl5C\npq7xcRO+25hy18CjMWqj8t3PH5MdBL8UMFZyDnIH9z9N019U0ZIaD3NqiiRgGD+U\nCSLkoXq5oni5RkDQCnzJRFo/Vzmx2P5OJyZvHYLtVOUwsp1dW8JFtdKJoGBxNc1M\nkc7eevfwUZEDc2dHxcwxDj0Tas05DaMBib3Oi0D/ipxDdzW+uENQHdCwy7XZf+T+\nig03Ega0/w+c/rdnUevdXK/L1sIO7F8hyDlVG1q0PeoJ8jXnZk+UfNYy820sPWIE\nIwtT1aODvnYgio8vgrDXpB0qVDNi2Ml83gYxznIQuxWg6dCrifvCa8TwCTe9tAhv\ngTkEkYdyBTpvT585z/1x+dra3uOGiMCN0rP3n3JaICDqCwImznvIP8kqNEnalWQj\npUVI3nKZunTtrL9vAegW9jF0Ipvyf+VSQmw+yN5B35Qfy95CwAwtJ/HPjy1sZmJZ\ncarKrlqoD4xdSyrIun3fraGTbM+u4S+USRjikce+pu1cHi70Y3xm4JBAZsRJgPwB\nG/Orf5yC+E2pCK+7rX3rWINgwmX/kk94EtnYbMeES+lhlKOu/mR09K00atuBEDnJ\no0MCM0BWYy5XQ2RAJLKCdcuJ2aWs/+slKRzlTCWnCUgISng6KFpcyA0aS/8r3ZyH\nSKdoSSgOtAieE/TGll0wjvONMIMfoEgR40OBV8BCSF8zWASZBXASTTSlUcu2wQ0q\n/wPFS2KkBdBc+qr+TxDNoeFDX+Rh9Nai25O/xoRtCC7afHsd5aQ4yen5C34/jsR1\n2kuayvZJ2pgYfIobFdgq9qHi637dVeW8n09XRq6HWhZu1ODO5bGX2oLr64MJAmgi\nfA+zu5Dfoe2Q4N1Ja3y0M7Xpfws14jyFxnJ8dR/T6rIJOy1QtHGo3UTai8nSBqCP\nRJ766EKBW7j83/53aYyChHvTXEPf4C29iOur72iMAlT2S06K/SH4fFM3brBzz0Fq\nEykXIgConLXDwj9+87XKYmOQX/0UP2sxAno6gJakdzExIod+u5koXP1o9vL5zMlH\nahZPgPpP2p2uAz1+9MHpVPo2EIrvibm5T89DznwuaEfe\n-----END ENCRYPTED PRIVATE KEY-----'

@pytest.fixture
def rsa_pubkey():
    if False:
        i = 10
        return i + 15
    return '-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzIdEbSkbPIc5F/aewNoq\nWPsF/YP+DByMwvYs+0K+lehc39P82fL8K2GIaGMBBzqlsX6CplAzGcoEQEBLTouL\nm+JYC5e1zRjaml4M+Bid8A7qwdjlWd0enCAxVB9BMNnj/mokWzh3hAQMBXfmddGR\nhH0P9KXfjBNh2V09vfHdtS9XMEEQjN6vCxaezXqsOMN3bjRTIcUatH7dVUgUpt9c\nye1mdbD5KVBgJ9MArc2tJ3rmB0lxjEbAhTEHrNnIkDOJCKE8TaQOW4RyVWlIvSEL\n+Ov0TPeXXaef8HJlkyIpKrCZ+c4i0N7qOlyrJEWTXmKNCj87xgTpY5I7ARISeOQD\n8QIDAQAB\n-----END PUBLIC KEY-----'

@pytest.fixture
def csr():
    if False:
        return 10
    return '-----BEGIN CERTIFICATE REQUEST-----\nMIICRTCCAS0CAQAwADCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMyH\nRG0pGzyHORf2nsDaKlj7Bf2D/gwcjML2LPtCvpXoXN/T/Nny/CthiGhjAQc6pbF+\ngqZQMxnKBEBAS06Li5viWAuXtc0Y2ppeDPgYnfAO6sHY5VndHpwgMVQfQTDZ4/5q\nJFs4d4QEDAV35nXRkYR9D/Sl34wTYdldPb3x3bUvVzBBEIzerwsWns16rDjDd240\nUyHFGrR+3VVIFKbfXMntZnWw+SlQYCfTAK3NrSd65gdJcYxGwIUxB6zZyJAziQih\nPE2kDluEclVpSL0hC/jr9Ez3l12nn/ByZZMiKSqwmfnOItDe6jpcqyRFk15ijQo/\nO8YE6WOSOwESEnjkA/ECAwEAAaAAMA0GCSqGSIb3DQEBCwUAA4IBAQB9PbGDorNt\nTl4xYObUsQwUkMVRPI59MLLYKEJRu/DGSA4sKf/vLK1ypyLIvxNp4gNFgm28nDV2\nt2gQ+DpBvwC1+XZQDZjgL7pPtLvErGCs6O6Y5fW8Lywxx5GqiVTIic/XLKTijKJv\nEecvwPjWv1VgtBKLZxN18KgIIs2Sq/t+GYe+Lu30c92Lc5INbrwTIEDYNTHywKet\n8FTSaYEMU6sGgsrIC5VxNT00EgJHjyjdCVIqQr/LqKyBMqJICWUSPq2ufjwqFsFi\nq1HXd62bA8k27ukX7w8qWsk6fOTwPh5F3883L5jVqcRsL9pqb4RUugTh/aReVlKW\n0WMDRBksXs1E\n-----END CERTIFICATE REQUEST-----'

@pytest.fixture
def ec_privkey():
    if False:
        print('Hello World!')
    return '-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg1lRdFUUOd7WZsydR\neMzFLD5u1Bjxg+NPia6Vznhb4EehRANCAAS+5meGSwViKrRQ3Ni1cfa08WG5dK/u\nldlNqU8U1Lz3ckCGI3TdGZ6nPaL3IT/UNH6C+J86RWSLY18hFHXoeKBD\n-----END PRIVATE KEY-----'

@pytest.fixture
def ec_pubkey():
    if False:
        print('Hello World!')
    return '-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEvuZnhksFYiq0UNzYtXH2tPFhuXSv\n7pXZTalPFNS893JAhiN03Rmepz2i9yE/1DR+gvifOkVki2NfIRR16HigQw==\n-----END PUBLIC KEY-----'

@pytest.fixture
def ed25519_privkey():
    if False:
        i = 10
        return i + 15
    return '-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIFKFjPIOBze2eo9x/EiCL0ni5GacaKIRZdfREBfuEdE9\n-----END PRIVATE KEY-----'

@pytest.fixture
def ed25519_pubkey():
    if False:
        print('Hello World!')
    return '-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAK+1yspaj/3Mb2K7H6y2d0Y+udSF+7sBozMY4aKUBR0I=\n-----END PUBLIC KEY-----'

@pytest.fixture
def ed448_privkey():
    if False:
        return 10
    return '-----BEGIN PRIVATE KEY-----\nMEcCAQAwBQYDK2VxBDsEObnZH0tXF86wbuvvz9Blg9MjUIkyW+Cpz6n4WmaFHIT4\nY2QKHSRG0M1ZUFr/EYH9F9mTgnAwmyp7oA==\n-----END PRIVATE KEY-----'

@pytest.fixture
def ed448_pubkey():
    if False:
        while True:
            i = 10
    return '-----BEGIN PUBLIC KEY-----\nMEMwBQYDK2VxAzoAiIWDcsK9mSaXUL+67ZIdyik8T5Zf0sLEwq3aUf6eysYxjEoZ\nvHv0+Ke3LRlEzGbwroKtP66opn4A\n-----END PUBLIC KEY-----'

@pytest.fixture
def cert_exts():
    if False:
        while True:
            i = 10
    return '\n-----BEGIN CERTIFICATE-----\nMIIEQDCCAyigAwIBAgIUDPVBmE6XZ0e15hwi1lQrVrO0/W8wDQYJKoZIhvcNAQEL\nBQAwKzELMAkGA1UEBhMCVVMxDTALBgNVBAMMBFRlc3QxDTALBgNVBAoMBFNhbHQw\nHhcNMjIxMTE1MTc1MzQwWhcNMjIxMjE1MTc1MzQwWjAAMIIBIjANBgkqhkiG9w0B\nAQEFAAOCAQ8AMIIBCgKCAQEAzIdEbSkbPIc5F/aewNoqWPsF/YP+DByMwvYs+0K+\nlehc39P82fL8K2GIaGMBBzqlsX6CplAzGcoEQEBLTouLm+JYC5e1zRjaml4M+Bid\n8A7qwdjlWd0enCAxVB9BMNnj/mokWzh3hAQMBXfmddGRhH0P9KXfjBNh2V09vfHd\ntS9XMEEQjN6vCxaezXqsOMN3bjRTIcUatH7dVUgUpt9cye1mdbD5KVBgJ9MArc2t\nJ3rmB0lxjEbAhTEHrNnIkDOJCKE8TaQOW4RyVWlIvSEL+Ov0TPeXXaef8HJlkyIp\nKrCZ+c4i0N7qOlyrJEWTXmKNCj87xgTpY5I7ARISeOQD8QIDAQABo4IBhTCCAYEw\nEgYDVR0TAQH/BAgwBgEB/wIBATAOBgNVHQ8BAf8EBAMCAQYwEwYDVR0lBAwwCgYI\nKwYBBQUHAwkwHQYDVR0OBBYEFJC2TjaEH+qfE9rS5jLliRt4kpnkMB8GA1UdIwQY\nMBaAFFzy8fRTKSOe7kBakqO0Ki71potnMBIGA1UdEgQLMAmCB3NhbHQuY2EwMAYI\nKwYBBQUHAQEEJDAiMCAGCCsGAQUFBzABhhRodHRwOi8vb2NzcC5zYWx0LmNhLzAj\nBgNVHREEHDAaggtzdWIuc2FsdC5jYYELc3ViQHNhbHQuY2EwKAYDVR0fBCEwHzAd\noBugGYYXaHR0cDovL3NhbHQuY2EvbXljYS5jcmwwEAYDVR0gBAkwBzAFBgMqBAUw\nDAYDVR0kBAUwA4ABAzAKBgNVHTYEAwIBAjAhBgNVHR4EGjAYoAwwCocIwKgAAP//\nAAChCDAGgQQuY29tMA8GCSsGAQUFBzABBQQCBQAwEQYIKwYBBQUHARgEBTADAgEF\nMA0GCSqGSIb3DQEBCwUAA4IBAQDAw8RirQU2WcDCKGPHHu7yZsrA08Fw/6P0OwLT\nhapKKXEdFcB8jflwEAQiZVge84xEYgdo/LgepRjOnkIc82Vlr3cy+F3A2c2JOwDU\nqf+A7rqJpwLZDHK1v4x9Boh3/JOiwOcyw2LugyQQhvKRqFhVjMAnX+cM3mSm2xn5\npaiBCooGdTl4l66JsTET56oXSsJ5FJ6XKPy86f/MY2n1LRSIQcvKGCP6vF5z7PDr\nsM09tkOYmSGN0coP6Y6PFS92zBnW6wXrzfNe0jvJMfVXJUbne5U0SQCY3mwkIuzB\nIiC+2Um3mhImnIoeRxH/cXTABsOrSE+QzIv7Z3orIUxyMqtm\n-----END CERTIFICATE-----'

@pytest.fixture
def csr_exts():
    if False:
        while True:
            i = 10
    return '-----BEGIN CERTIFICATE REQUEST-----\nMIIDvjCCAqYCAQAwADCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMyH\nRG0pGzyHORf2nsDaKlj7Bf2D/gwcjML2LPtCvpXoXN/T/Nny/CthiGhjAQc6pbF+\ngqZQMxnKBEBAS06Li5viWAuXtc0Y2ppeDPgYnfAO6sHY5VndHpwgMVQfQTDZ4/5q\nJFs4d4QEDAV35nXRkYR9D/Sl34wTYdldPb3x3bUvVzBBEIzerwsWns16rDjDd240\nUyHFGrR+3VVIFKbfXMntZnWw+SlQYCfTAK3NrSd65gdJcYxGwIUxB6zZyJAziQih\nPE2kDluEclVpSL0hC/jr9Ez3l12nn/ByZZMiKSqwmfnOItDe6jpcqyRFk15ijQo/\nO8YE6WOSOwESEnjkA/ECAwEAAaCCAXcwggFzBgkqhkiG9w0BCQ4xggFkMIIBYDAS\nBgNVHRMBAf8ECDAGAQH/AgEBMA4GA1UdDwEB/wQEAwIBBjATBgNVHSUEDDAKBggr\nBgEFBQcDCTAdBgNVHQ4EFgQUkLZONoQf6p8T2tLmMuWJG3iSmeQwEgYDVR0SBAsw\nCYIHc2FsdC5jYTAwBggrBgEFBQcBAQQkMCIwIAYIKwYBBQUHMAGGFGh0dHA6Ly9v\nY3NwLnNhbHQuY2EvMCMGA1UdEQQcMBqCC3N1Yi5zYWx0LmNhgQtzdWJAc2FsdC5j\nYTAoBgNVHR8EITAfMB2gG6AZhhdodHRwOi8vc2FsdC5jYS9teWNhLmNybDAQBgNV\nHSAECTAHMAUGAyoEBTAMBgNVHSQEBTADgAEDMAoGA1UdNgQDAgECMCEGA1UdHgQa\nMBigDDAKhwjAqAAA//8AAKEIMAaBBC5jb20wDwYJKwYBBQUHMAEFBAIFADARBggr\nBgEFBQcBGAQFMAMCAQUwDQYJKoZIhvcNAQELBQADggEBAINICpHFaJaxDfABkbwV\nb3Ji/djatf5dc2jB/A/qP18+M97xIpvJPi/xGTR+sMqffsXLGuZgrhmmkhrbYqIf\nCHi9VPpZ7l0sB/mESJ5+//50J5tRN6I+7UCc3MWTs45HM8/alJQQAKX8Fdx6cZnI\n2lz6raNyT4DUo/eympAtSjJRNnhT62YEiIR+9+Vu4aMjsnRLgLbtOGUraOoyC9do\neY6fyUlpNgz8ny7Ow6nV/J5FNaZfEt/79X+kjHdPkqz7r2A1PEI/Uu+Gksoyizvs\nqFrpUgv3nrP7olcq8rKYbwI9bXj3LMQpWtUZ300Sy2+dzwjoBneJ9VmkaD2U6Njd\nO68=\n-----END CERTIFICATE REQUEST-----'

@pytest.fixture
def cert_exts_read():
    if False:
        for i in range(10):
            print('nop')
    return {'extensions': {'authorityInfoAccess': {'value': [{'OCSP': 'http://ocsp.salt.ca/'}], 'critical': False}, 'authorityKeyIdentifier': {'critical': False, 'issuer': None, 'issuer_sn': None, 'keyid': '5C:F2:F1:F4:53:29:23:9E:EE:40:5A:92:A3:B4:2A:2E:F5:A6:8B:67'}, 'basicConstraints': {'ca': True, 'critical': True, 'pathlen': 1}, 'certificatePolicies': {'critical': False, 'value': [{'1.2.4.5': []}]}, 'crlDistributionPoints': {'critical': False, 'value': [{'crlissuer': [], 'fullname': ['URI:http://salt.ca/myca.crl'], 'reasons': [], 'relativename': None}]}, 'extendedKeyUsage': {'critical': False, 'value': ['OCSPSigning']}, 'inhibitAnyPolicy': {'critical': False, 'value': 2}, 'issuerAltName': {'critical': False, 'value': ['DNS:salt.ca']}, 'keyUsage': {'cRLSign': True, 'critical': True, 'dataEncipherment': False, 'decipherOnly': False, 'digitalSignature': False, 'encipherOnly': False, 'keyAgreement': False, 'keyCertSign': True, 'keyEncipherment': False, 'nonRepudiation': False}, 'nameConstraints': {'critical': False, 'excluded': ['mail:.com'], 'permitted': ['IP:192.168.0.0/16']}, 'noCheck': {'critical': False, 'value': True}, 'policyConstraints': {'critical': False, 'inhibitPolicyMapping': None, 'requireExplicitPolicy': 3}, 'subjectAltName': {'critical': False, 'value': ['DNS:sub.salt.ca', 'mail:sub@salt.ca']}, 'subjectKeyIdentifier': {'critical': False, 'value': '90:B6:4E:36:84:1F:EA:9F:13:DA:D2:E6:32:E5:89:1B:78:92:99:E4'}, 'tlsfeature': {'critical': False, 'value': ['status_request']}}, 'fingerprints': {'md5': '5C:D7:BF:68:AD:09:1A:CA:42:8E:62:10:60:21:13:20', 'sha1': '93:1A:31:45:AC:3D:62:E5:0C:59:E1:D1:8E:45:F2:BD:28:51:20:34', 'sha256': 'E4:EB:84:87:17:80:E4:6D:6E:B8:9C:A0:EE:88:AF:CA:57:C7:8A:86:5A:A8:53:E1:38:DF:7A:43:D7:19:54:E1'}, 'issuer': OrderedDict([('C', 'US'), ('O', 'Salt'), ('CN', 'Test')]), 'issuer_hash': '19:2C:28:89', 'issuer_str': 'O=Salt,CN=Test,C=US', 'key_size': 2048, 'key_type': 'rsa', 'not_after': '2022-12-15 17:53:40', 'not_before': '2022-11-15 17:53:40', 'public_key': '-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzIdEbSkbPIc5F/aewNoq\nWPsF/YP+DByMwvYs+0K+lehc39P82fL8K2GIaGMBBzqlsX6CplAzGcoEQEBLTouL\nm+JYC5e1zRjaml4M+Bid8A7qwdjlWd0enCAxVB9BMNnj/mokWzh3hAQMBXfmddGR\nhH0P9KXfjBNh2V09vfHdtS9XMEEQjN6vCxaezXqsOMN3bjRTIcUatH7dVUgUpt9c\nye1mdbD5KVBgJ9MArc2tJ3rmB0lxjEbAhTEHrNnIkDOJCKE8TaQOW4RyVWlIvSEL\n+Ov0TPeXXaef8HJlkyIpKrCZ+c4i0N7qOlyrJEWTXmKNCj87xgTpY5I7ARISeOQD\n8QIDAQAB\n-----END PUBLIC KEY-----\n', 'serial_number': '0C:F5:41:98:4E:97:67:47:B5:E6:1C:22:D6:54:2B:56:B3:B4:FD:6F', 'signature_algorithm': 'sha256WithRSAEncryption', 'subject': OrderedDict(), 'subject_hash': 'D6:DC:44:F9', 'subject_str': '', 'version': 3}

@pytest.fixture
def csr_exts_read():
    if False:
        i = 10
        return i + 15
    return {'extensions': {'authorityInfoAccess': {'value': [{'OCSP': 'http://ocsp.salt.ca/'}], 'critical': False}, 'basicConstraints': {'ca': True, 'critical': True, 'pathlen': 1}, 'certificatePolicies': {'critical': False, 'value': [{'1.2.4.5': []}]}, 'crlDistributionPoints': {'critical': False, 'value': [{'crlissuer': [], 'fullname': ['URI:http://salt.ca/myca.crl'], 'reasons': [], 'relativename': None}]}, 'extendedKeyUsage': {'critical': False, 'value': ['OCSPSigning']}, 'inhibitAnyPolicy': {'critical': False, 'value': 2}, 'issuerAltName': {'critical': False, 'value': ['DNS:salt.ca']}, 'keyUsage': {'cRLSign': True, 'critical': True, 'dataEncipherment': False, 'decipherOnly': False, 'digitalSignature': False, 'encipherOnly': False, 'keyAgreement': False, 'keyCertSign': True, 'keyEncipherment': False, 'nonRepudiation': False}, 'nameConstraints': {'critical': False, 'excluded': ['mail:.com'], 'permitted': ['IP:192.168.0.0/16']}, 'noCheck': {'critical': False, 'value': True}, 'policyConstraints': {'critical': False, 'inhibitPolicyMapping': None, 'requireExplicitPolicy': 3}, 'subjectAltName': {'critical': False, 'value': ['DNS:sub.salt.ca', 'mail:sub@salt.ca']}, 'subjectKeyIdentifier': {'critical': False, 'value': '90:B6:4E:36:84:1F:EA:9F:13:DA:D2:E6:32:E5:89:1B:78:92:99:E4'}, 'tlsfeature': {'critical': False, 'value': ['status_request']}}, 'key_size': 2048, 'key_type': 'rsa', 'public_key_hash': '90:B6:4E:36:84:1F:EA:9F:13:DA:D2:E6:32:E5:89:1B:78:92:99:E4', 'subject': OrderedDict(), 'subject_hash': 'D6:DC:44:F9', 'subject_str': ''}

@pytest.fixture
def crl():
    if False:
        print('Hello World!')
    return '-----BEGIN X509 CRL-----\nMIIBnTCBhgIBATANBgkqhkiG9w0BAQsFADArMQswCQYDVQQGEwJVUzENMAsGA1UE\nAwwEVGVzdDENMAsGA1UECgwEU2FsdBcNMjIxMTE2MDQxMDA4WhcNMjMwMjI0MDQx\nMDA4WjAnMCUCFAz1QZhOl2dHteYcItZUK1aztP1vFw0yMjExMTYwMzEwMDhaMA0G\nCSqGSIb3DQEBCwUAA4IBAQCnNoVC4rqKd6FIXe3gvCrGSTa5IroFQRCMyVNnOAfZ\nlGD8dJ2zzOwsne3tOg0P9oNZLKPztFiScjAaZG7ZePpaAA8X4HmgKDn9+U2pLR1J\nks4/XI0rSCrRO5TYurCgyw+Zo8E/T2NJgGixcWX8NKAjKk+ZgSLXCOHi8z9cq5Mr\noDN6x8xzk7eYT0PXc1bPzJPsNLSeEECGePqqLeBbpF7e0nwHcRG1Ak1pfASqA0Wm\nArzeIgmP0P2n3oBVEuQK2467rTuqhXpAaJL0lASxS13YXYpMIxfkbELe0r3OHMPP\nzfEPMyxWSMAqcsjSQ+MuF3KCdtaWAk7xTYpBafvRK4pC\n-----END X509 CRL-----'

@pytest.fixture
def crl_all():
    if False:
        return 10
    return '-----BEGIN X509 CRL-----\nMIIDFDCCAfwCAQEwDQYJKoZIhvcNAQEFBQAwXzEjMCEGA1UEChMaU2FtcGxlIFNp\nZ25lciBPcmdhbml6YXRpb24xGzAZBgNVBAsTElNhbXBsZSBTaWduZXIgVW5pdDEb\nMBkGA1UEAxMSU2FtcGxlIFNpZ25lciBDZXJ0Fw0xMzAyMTgxMDMyMDBaFw0xMzAy\nMTgxMDQyMDBaMIIBNjA8AgMUeUcXDTEzMDIxODEwMjIxMlowJjAKBgNVHRUEAwoB\nAzAYBgNVHRgEERgPMjAxMzAyMTgxMDIyMDBaMDwCAxR5SBcNMTMwMjE4MTAyMjIy\nWjAmMAoGA1UdFQQDCgEGMBgGA1UdGAQRGA8yMDEzMDIxODEwMjIwMFowPAIDFHlJ\nFw0xMzAyMTgxMDIyMzJaMCYwCgYDVR0VBAMKAQQwGAYDVR0YBBEYDzIwMTMwMjE4\nMTAyMjAwWjA8AgMUeUoXDTEzMDIxODEwMjI0MlowJjAKBgNVHRUEAwoBATAYBgNV\nHRgEERgPMjAxMzAyMTgxMDIyMDBaMDwCAxR5SxcNMTMwMjE4MTAyMjUxWjAmMAoG\nA1UdFQQDCgEFMBgGA1UdGAQRGA8yMDEzMDIxODEwMjIwMFqgLzAtMB8GA1UdIwQY\nMBaAFL4SAcyq6hGA2i6tsurHtfuf+a00MAoGA1UdFAQDAgEDMA0GCSqGSIb3DQEB\nBQUAA4IBAQBCIb6B8cN5dmZbziETimiotDy+FsOvS93LeDWSkNjXTG/+bGgnrm3a\nQpgB7heT8L2o7s2QtjX2DaTOSYL3nZ/Ibn/R8S0g+EbNQxdk5/la6CERxiRp+E2T\nUG8LDb14YVMhRGKvCguSIyUG0MwGW6waqVtd6K71u7vhIU/Tidf6ZSdsTMhpPPFu\nPUid4j29U3q10SGFF6cCt1DzjvUcCwHGhHA02Men70EgZFADPLWmLg0HglKUh1iZ\nWcBGtev/8VsUijyjsM072C6Ut5TwNyrrthb952+eKlmxLNgT0o5hVYxjXhtwLQsL\n7QZhrypAM1DLYqQjkiDI7hlvt7QuDGTJ\n-----END X509 CRL-----'

@pytest.fixture
def crl_args(tmp_path, ca_cert, ca_key):
    if False:
        for i in range(10):
            print('nop')
    return {'signing_private_key': ca_key, 'signing_cert': ca_cert, 'revoked': []}

@pytest.fixture
def crl_args_exts():
    if False:
        return 10
    return {'authorityKeyIdentifier': 'keyid:always', 'issuerAltName': 'DNS:salt.ca', 'issuingDistributionPoint': {'critical': True, 'fullname': ['URI:http://salt.ca/myca.crl']}, 'CRLNumber': 1}

@pytest.fixture
def crl_revoked():
    if False:
        return 10
    return [{'serial_number': '01337A', 'extensions': {'CRLReason': 'unspecified', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '01337B', 'extensions': {'CRLReason': 'keyCompromise', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '01337C', 'extensions': {'CRLReason': 'cACompromise', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '01337D', 'extensions': {'CRLReason': 'affiliationChanged', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '01337E', 'extensions': {'CRLReason': 'superseded', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '01337F', 'extensions': {'CRLReason': 'cessationOfOperation', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '013380', 'extensions': {'CRLReason': 'certificateHold', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '013381', 'extensions': {'CRLReason': 'privilegeWithdrawn', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '013381', 'extensions': {'CRLReason': 'aACompromise', 'invalidityDate': '2022-11-18 13:37:00'}}, {'serial_number': '013382', 'extensions': {'CRLReason': 'removeFromCRL'}}]

@pytest.mark.parametrize('algo', ['rsa', 'ec', 'ed25519', 'ed448'])
def test_create_certificate_self_signed(x509, algo, request):
    if False:
        while True:
            i = 10
    privkey = request.getfixturevalue(f'{algo}_privkey')
    res = x509.create_certificate(signing_private_key=privkey, CN='success')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_certificate_write_to_path(x509, encoding, rsa_privkey, tmp_path):
    if False:
        return 10
    tgt = tmp_path / 'cert'
    x509.create_certificate(signing_private_key=rsa_privkey, CN='success', encoding=encoding, path=str(tgt))
    assert tgt.exists()
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(tgt.read_bytes(), encoding=encoding)
    assert cert.subject.rfc4514_string() == 'CN=success'

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_certificate_write_to_path_overwrite(x509, encoding, rsa_privkey, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    tgt = tmp_path / 'cert'
    tgt.write_text('occupied')
    assert tgt.exists()
    x509.create_certificate(signing_private_key=rsa_privkey, CN='success', encoding=encoding, path=str(tgt))
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(tgt.read_bytes(), encoding=encoding)
    assert cert.subject.rfc4514_string() == 'CN=success'

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_certificate_write_to_path_overwrite_false(x509, encoding, rsa_privkey, tmp_path):
    if False:
        i = 10
        return i + 15
    tgt = tmp_path / 'cert'
    tgt.write_text('occupied')
    assert tgt.exists()
    x509.create_certificate(signing_private_key=rsa_privkey, CN='success', encoding=encoding, path=str(tgt), overwrite=False)
    assert tgt.read_text() == 'occupied'

def test_create_certificate_raw(x509, rsa_privkey):
    if False:
        print('Hello World!')
    res = x509.create_certificate(signing_private_key=rsa_privkey, CN='success', raw=True)
    assert isinstance(res, bytes)
    assert res.startswith(b'-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

@pytest.mark.parametrize('algo', ['rsa', 'ec', 'ed25519', 'ed448'])
def test_create_certificate_from_privkey(x509, ca_key, ca_cert, algo, request):
    if False:
        for i in range(10):
            print('nop')
    privkey = request.getfixturevalue(f'{algo}_privkey')
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, private_key=privkey, CN='success')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

def test_create_certificate_from_encrypted_privkey(x509, ca_key, ca_cert, rsa_privkey_enc):
    if False:
        for i in range(10):
            print('nop')
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey_enc, private_key_passphrase='hunter2', CN='success')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

def test_create_certificate_from_encrypted_privkey_with_encrypted_privkey(x509, ca_key_enc, ca_cert, rsa_privkey_enc):
    if False:
        i = 10
        return i + 15
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key_enc, signing_private_key_passphrase='correct horse battery staple', private_key=rsa_privkey_enc, private_key_passphrase='hunter2', CN='success')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

@pytest.mark.parametrize('algo', ['rsa', 'ec', 'ed25519', 'ed448'])
def test_create_certificate_from_pubkey(x509, ca_key, ca_cert, algo, request):
    if False:
        return 10
    pubkey = request.getfixturevalue(f'{algo}_pubkey')
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, public_key=pubkey, CN='success')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

def test_create_certificate_from_csr(x509, ca_key, ca_cert, csr):
    if False:
        return 10
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, csr=csr, CN='success')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

def test_create_certificate_from_mismatching_private_key(x509, rsa_privkey, ca_cert, ec_privkey):
    if False:
        return 10
    with pytest.raises(salt.exceptions.SaltInvocationError):
        x509.create_certificate(signing_cert=ca_cert, signing_private_key=rsa_privkey, private_key=rsa_privkey, CN='success')

def test_create_certificate_with_ca_cert_needs_any_pubkey_source(x509, ca_key, ca_cert):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(salt.exceptions.SaltInvocationError):
        x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key)

def test_create_certificate_with_extensions(x509, ca_key, ca_cert, rsa_privkey):
    if False:
        print('Hello World!')
    extensions = {'basicConstraints': 'critical, CA:TRUE, pathlen:1', 'keyUsage': 'critical, cRLSign, keyCertSign', 'extendedKeyUsage': 'OCSPSigning', 'subjectKeyIdentifier': 'hash', 'authorityKeyIdentifier': 'keyid:always', 'issuerAltName': 'DNS:salt.ca', 'authorityInfoAccess': 'OCSP;URI:http://ocsp.salt.ca/', 'subjectAltName': 'DNS:sub.salt.ca,email:sub@salt.ca', 'crlDistributionPoints': 'URI:http://salt.ca/myca.crl', 'certificatePolicies': '1.2.4.5', 'policyConstraints': 'requireExplicitPolicy:3', 'inhibitAnyPolicy': 2, 'nameConstraints': 'permitted;IP:192.168.0.0/255.255.0.0,excluded;email:.com', 'noCheck': True, 'tlsfeature': 'status_request'}
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey, **extensions)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    for x in [cx509.BasicConstraints, cx509.KeyUsage, cx509.ExtendedKeyUsage, cx509.SubjectKeyIdentifier, cx509.AuthorityKeyIdentifier, cx509.IssuerAlternativeName, cx509.AuthorityInformationAccess, cx509.SubjectAlternativeName, cx509.CRLDistributionPoints, cx509.CertificatePolicies, cx509.PolicyConstraints, cx509.InhibitAnyPolicy, cx509.NameConstraints, cx509.OCSPNoCheck, cx509.TLSFeature]:
        cert.extensions.get_extension_for_class(x)

def test_create_certificate_from_csr_with_extensions(x509, ca_key, ca_cert, csr_exts):
    if False:
        return 10
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, csr=csr_exts)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    for x in [cx509.BasicConstraints, cx509.KeyUsage, cx509.ExtendedKeyUsage, cx509.SubjectKeyIdentifier, cx509.SubjectAlternativeName, cx509.CertificatePolicies, cx509.PolicyConstraints, cx509.InhibitAnyPolicy, cx509.NameConstraints, cx509.OCSPNoCheck, cx509.TLSFeature]:
        cert.extensions.get_extension_for_class(x)
    for x in [cx509.IssuerAlternativeName, cx509.AuthorityInformationAccess, cx509.CRLDistributionPoints]:
        with pytest.raises(cx509.ExtensionNotFound):
            cert.extensions.get_extension_for_class(x)

@pytest.mark.parametrize('arg', [{'C': 'US', 'CN': 'Homer', 'L': 'Springfield'}, pytest.param({'subject': ['C=US', 'L=Springfield', 'CN=Homer']}, marks=pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 37, reason='At least cryptography v37 is required for parsing RFC4514 strings.')), pytest.param({'subject': 'CN=Homer,L=Springfield,C=US'}, marks=pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 37, reason='At least cryptography v37 is required for parsing RFC4514 strings.'))])
def test_create_certificate_with_distinguished_name(x509, ca_cert, ca_key, rsa_privkey, arg):
    if False:
        print('Hello World!')
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey, **arg)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=Homer,L=Springfield,C=US'

def test_create_certificate_with_signing_policy(x509, ca_cert, ca_key, rsa_privkey):
    if False:
        print('Hello World!')
    res = x509.create_certificate(signing_policy='testpolicy', CN='from_kwargs', basicConstraints='CA:TRUE', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=from_signing_policy'
    for x in [cx509.BasicConstraints, cx509.KeyUsage, cx509.SubjectKeyIdentifier]:
        ext = cert.extensions.get_extension_for_class(x)
        if x == cx509.BasicConstraints:
            assert not ext.value.ca

def test_create_certificate_with_signing_policy_no_subject_override(x509, ca_cert, ca_key, rsa_privkey):
    if False:
        for i in range(10):
            print('nop')
    '\n    Since `subject` gets precedence, if the signing policy uses direct kwargs\n    for name attributes, ensure that setting `subject` gets ignored.\n    '
    res = x509.create_certificate(signing_policy='testpolicy', subject={'CN': 'from_kwargs', 'SERIALNUMBER': '1234'}, signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=from_signing_policy'

@pytest.mark.parametrize('signing_policy,subject,expected', [('testsubjectdictpolicy', 'CN=from_kwargs,SERIALNUMBER=1234', 'CN=from_signing_policy'), ('testsubjectdictpolicy', ['CN=from_kwargs', 'SERIALNUMBER=1234'], 'CN=from_signing_policy'), ('testsubjectstrpolicy', ['CN=from_kwargs', 'SERIALNUMBER=1234'], 'CN=from_signing_policy'), ('testsubjectstrpolicy', {'CN': 'from_kwargs', 'SERIALNUMBER': '1234'}, 'CN=from_signing_policy'), ('testsubjectlistpolicy', 'CN=from_kwargs,SERIALNUMBER=1234', 'O=Salt Test,L=Salt Lake City,C=US'), ('testsubjectlistpolicy', {'CN': 'from_kwargs', 'SERIALNUMBER': '1234'}, 'O=Salt Test,L=Salt Lake City,C=US')])
@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 37, reason='Parsing of RFC4514 strings requires cryptography >= 37')
def test_create_certificate_with_signing_policy_subject_type_mismatch_no_override(x509, ca_cert, ca_key, rsa_privkey, signing_policy, subject, expected):
    if False:
        return 10
    '\n    When both signing_policy and kwargs have `subject` and the types do not match,\n    force signing_policy\n    '
    res = x509.create_certificate(signing_policy=signing_policy, subject=subject, signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == expected

@pytest.mark.parametrize('signing_policy,subject,expected', [('testsubjectdictpolicy', {'CN': 'from_kwargs', 'O': 'Foo'}, 'CN=from_signing_policy,O=Foo'), ('testsubjectstrpolicy', 'CN=from_kwargs,O=Foo', 'CN=from_signing_policy'), ('testsubjectlistpolicy', ['CN=Test1'], 'CN=Test1,O=Salt Test,L=Salt Lake City,C=US')])
@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 37, reason='Parsing of RFC4514 strings requires cryptography >= 37')
def test_create_certificate_with_signing_policy_subject_merging(x509, ca_cert, ca_key, rsa_privkey, signing_policy, subject, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    When both signing_policy and kwargs have `subject` and the types match,\n    merge them with priority to signing_policy\n    '
    res = x509.create_certificate(signing_policy=signing_policy, subject=subject, signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == expected

@pytest.mark.parametrize('subject,expected', [({'CN': 'from_kwargs', 'O': 'Foo'}, 'CN=from_kwargs,O=Foo'), ('CN=from_kwargs,O=Foo', 'CN=from_kwargs,O=Foo'), (['O=Foo', 'CN=Test1'], 'CN=Test1,O=Foo')])
@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 37, reason='Parsing of RFC4514 strings requires cryptography >= 37')
def test_create_certificate_with_signing_policy_no_subject(x509, ca_cert, ca_key, rsa_privkey, subject, expected):
    if False:
        for i in range(10):
            print('nop')
    '\n    When signing_policy does not enforce `subject` somehow,\n    make sure to follow kwargs\n    '
    res = x509.create_certificate(signing_policy='testnosubjectpolicy', subject=subject, signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == expected

def test_create_certificate_not_before_not_after(x509, ca_cert, ca_key, rsa_privkey):
    if False:
        for i in range(10):
            print('nop')
    fmt = '%Y-%m-%d %H:%M:%S'
    not_before = '2022-12-21 13:37:10'
    not_after = '2032-12-21 13:37:10'
    res = x509.create_certificate(not_before=not_before, not_after=not_after, signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.not_valid_before == datetime.datetime.strptime(not_before, fmt)
    assert cert.not_valid_after == datetime.datetime.strptime(not_after, fmt)

@pytest.mark.parametrize('sn', [3735928559, 'DE:AD:BE:EF', 'deadbeef'])
def test_create_certificate_explicit_serial_number(x509, ca_cert, ca_key, rsa_privkey, sn):
    if False:
        return 10
    res = x509.create_certificate(serial_number=sn, signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.serial_number == 3735928559

def test_create_certificate_as_der(x509, ca_cert, ca_key, rsa_privkey):
    if False:
        for i in range(10):
            print('nop')
    res = x509.create_certificate(encoding='der', CN='success', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    cert = _get_cert(res, 'der')
    assert cert.subject.rfc4514_string() == 'CN=success'

@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 37, reason='PKCS7 serialization requires cryptography v37+')
@pytest.mark.parametrize('typ', ['pem', 'der'])
def test_create_certificate_as_pkcs7(x509, ca_cert, ca_key, rsa_privkey, typ):
    if False:
        print('Hello World!')
    res = x509.create_certificate(encoding=f'pkcs7_{typ}', CN='success', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    cert = _get_cert(res, f'pkcs7_{typ}')
    assert cert[0].subject.rfc4514_string() == 'CN=success'

@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 36, reason='Complete PKCS12 deserialization requires cryptography v36+')
def test_create_certificate_as_pkcs12(x509, ca_cert, ca_key, rsa_privkey):
    if False:
        i = 10
        return i + 15
    res = x509.create_certificate(encoding='pkcs12', pkcs12_friendlyname='foo', CN='success', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    cert = _get_cert(res, 'pkcs12')
    assert cert.cert.certificate.subject.rfc4514_string() == 'CN=success'
    assert cert.cert.friendly_name == b'foo'

@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 36, reason='Complete PKCS12 deserialization requires cryptography v36+')
def test_create_certificate_as_encrypted_pkcs12(x509, ca_cert, ca_key, rsa_privkey_enc):
    if False:
        return 10
    res = x509.create_certificate(encoding='pkcs12', private_key_passphrase='hunter2', pkcs12_passphrase='hunter3', pkcs12_embed_private_key=True, pkcs12_friendlyname='foo', CN='success', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey_enc)
    cert = _get_cert(res, 'pkcs12', 'hunter3')
    assert cert.cert.certificate.subject.rfc4514_string() == 'CN=success'
    assert cert.cert.friendly_name == b'foo'

def test_create_certificate_append_certs_pem(x509, ca_cert, ca_key, rsa_privkey):
    if False:
        return 10
    res = x509.create_certificate(append_certs=[ca_cert], CN='success', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'
    assert res.endswith(ca_cert)

@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 37, reason='PKCS7 serialization requires cryptography v37+')
@pytest.mark.parametrize('typ', ['pem', 'der'])
def test_create_certificate_append_certs_pkcs7(x509, ca_cert, ca_key, rsa_privkey, typ):
    if False:
        while True:
            i = 10
    res = x509.create_certificate(append_certs=[ca_cert], encoding=f'pkcs7_{typ}', CN='success', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    cert = _get_cert(res, f'pkcs7_{typ}')
    assert cert[0].subject.rfc4514_string() == 'CN=success'
    assert cert[1].serial_number == _get_cert(ca_cert).serial_number

@pytest.mark.skipif(CRYPTOGRAPHY_VERSION[0] < 36, reason='Complete PKCS12 deserialization requires cryptography v36+')
def test_create_certificate_append_certs_pkcs12(x509, ca_cert, ca_key, rsa_privkey):
    if False:
        print('Hello World!')
    res = x509.create_certificate(append_certs=[ca_cert], encoding='pkcs12', CN='success', signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey)
    cert = _get_cert(res, 'pkcs12')
    assert cert.cert.certificate.subject.rfc4514_string() == 'CN=success'
    assert cert.additional_certs[0].certificate.serial_number == _get_cert(ca_cert).serial_number

@pytest.mark.parametrize('prepend_cn', [False, True])
def test_create_certificate_copypath(x509, rsa_privkey, ca_cert, ca_key, prepend_cn, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    res = x509.create_certificate(signing_cert=ca_cert, signing_private_key=ca_key, private_key=rsa_privkey, CN='success', copypath=str(tmp_path), prepend_cn=prepend_cn)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'
    prefix = ''
    if prepend_cn:
        prefix = 'success-'
    assert (tmp_path / f'{prefix}{cert.serial_number:x}.crt').exists()

def test_create_crl_empty(x509, crl_args, ca_cert):
    if False:
        print('Hello World!')
    res = x509.create_crl(**crl_args)
    assert res.startswith('-----BEGIN X509 CRL-----')

def test_create_crl(x509, crl_args, crl_revoked, ca_cert):
    if False:
        i = 10
        return i + 15
    crl_args['revoked'] = crl_revoked
    res = x509.create_crl(**crl_args)
    assert res.startswith('-----BEGIN X509 CRL-----')

def test_create_crl_with_exts(x509, crl_args, crl_args_exts, ca_cert):
    if False:
        print('Hello World!')
    crl_args.update({'extensions': crl_args_exts})
    res = x509.create_crl(**crl_args)
    assert res.startswith('-----BEGIN X509 CRL-----')

def test_create_crl_from_certificate(x509, ca_cert, ca_key, cert_exts):
    if False:
        i = 10
        return i + 15
    revoked = [{'certificate': cert_exts}]
    res = x509.create_crl(signing_cert=ca_cert, revoked=revoked, signing_private_key=ca_key)
    assert res.startswith('-----BEGIN X509 CRL-----')

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_crl_write_to_path(x509, encoding, crl_args, tmp_path):
    if False:
        i = 10
        return i + 15
    tgt = tmp_path / 'crl'
    crl_args['encoding'] = encoding
    crl_args['path'] = str(tgt)
    x509.create_crl(**crl_args)
    assert tgt.exists()
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN X509 CRL-----')

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_crl_write_to_path_overwrite(x509, encoding, crl_args, tmp_path):
    if False:
        print('Hello World!')
    tgt = tmp_path / 'cert'
    crl_args['encoding'] = encoding
    crl_args['path'] = str(tgt)
    tgt.write_text('occupied')
    assert tgt.exists()
    x509.create_crl(**crl_args)
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN X509 CRL-----')

def test_create_crl_raw(x509, crl_args):
    if False:
        return 10
    res = x509.create_crl(**crl_args, raw=True)
    assert isinstance(res, bytes)
    assert res.startswith(b'-----BEGIN X509 CRL-----')

@pytest.mark.parametrize('algo', ['rsa', 'ec', 'ed25519', 'ed448'])
def test_create_csr(x509, algo, request):
    if False:
        print('Hello World!')
    privkey = request.getfixturevalue(f'{algo}_privkey')
    res = x509.create_csr(private_key=privkey)
    assert res.startswith('-----BEGIN CERTIFICATE REQUEST-----')

def test_create_csr_der(x509, rsa_privkey):
    if False:
        while True:
            i = 10
    res = x509.create_csr(private_key=rsa_privkey, encoding='der')
    assert base64.b64decode(res)

def test_create_csr_with_extensions(x509, rsa_privkey):
    if False:
        i = 10
        return i + 15
    extensions = {'basicConstraints': 'critical, CA:TRUE, pathlen:1', 'keyUsage': 'critical, cRLSign, keyCertSign', 'extendedKeyUsage': 'OCSPSigning', 'subjectKeyIdentifier': 'hash', 'issuerAltName': 'DNS:salt.ca', 'authorityInfoAccess': 'OCSP;URI:http://ocsp.salt.ca/', 'subjectAltName': 'DNS:sub.salt.ca,email:sub@salt.ca', 'crlDistributionPoints': 'URI:http://salt.ca/myca.crl', 'certificatePolicies': '1.2.4.5', 'policyConstraints': 'requireExplicitPolicy:3', 'inhibitAnyPolicy': 2, 'nameConstraints': 'permitted;IP:192.168.0.0/255.255.0.0,excluded;email:.com', 'noCheck': True, 'tlsfeature': 'status_request'}
    res = x509.create_csr(private_key=rsa_privkey, **extensions)
    assert res.startswith('-----BEGIN CERTIFICATE REQUEST-----')

def test_create_csr_with_wildcard_san(x509, rsa_privkey):
    if False:
        print('Hello World!')
    '\n    Test that wildcards in SAN extension are supported. Issue #65072\n    '
    res = x509.create_csr(private_key=rsa_privkey, subjectAltName='DNS:*.salt.ca')
    assert res.startswith('-----BEGIN CERTIFICATE REQUEST-----')

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_csr_write_to_path(x509, encoding, rsa_privkey, tmp_path):
    if False:
        return 10
    tgt = tmp_path / 'csr'
    x509.create_csr(private_key=rsa_privkey, encoding=encoding, path=str(tgt))
    assert tgt.exists()
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN CERTIFICATE REQUEST-----')

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_csr_write_to_path_overwrite(x509, encoding, rsa_privkey, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    tgt = tmp_path / 'cert'
    tgt.write_text('occupied')
    assert tgt.exists()
    x509.create_csr(private_key=rsa_privkey, encoding=encoding, path=str(tgt))
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN CERTIFICATE REQUEST-----')

def test_create_csr_raw(x509, rsa_privkey):
    if False:
        return 10
    res = x509.create_csr(private_key=rsa_privkey, raw=True)
    assert isinstance(res, bytes)
    assert res.startswith(b'-----BEGIN CERTIFICATE REQUEST-----')

@pytest.mark.slow_test
@pytest.mark.parametrize('algo', ['rsa', 'ec', 'ed25519', 'ed448'])
def test_create_private_key(x509, algo):
    if False:
        while True:
            i = 10
    res = x509.create_private_key(algo=algo)
    assert res.startswith('-----BEGIN PRIVATE KEY-----')

@pytest.mark.slow_test
@pytest.mark.parametrize('algo', ['rsa', 'ec', 'ed25519', 'ed448'])
def test_create_private_key_with_passphrase(x509, algo):
    if False:
        i = 10
        return i + 15
    passphrase = 'hunter2'
    res = x509.create_private_key(algo=algo, passphrase=passphrase)
    assert res.startswith('-----BEGIN ENCRYPTED PRIVATE KEY-----')
    x509.get_private_key_size(res, passphrase=passphrase)

@pytest.mark.slow_test
def test_create_private_key_der(x509):
    if False:
        i = 10
        return i + 15
    res = x509.create_private_key(algo='ec', encoding='der')
    assert base64.b64decode(res)

@pytest.mark.slow_test
@pytest.mark.parametrize('passphrase', [None, 'hunter2'])
def test_create_private_key_pkcs12(x509, passphrase):
    if False:
        i = 10
        return i + 15
    res = x509.create_private_key(algo='ec', encoding='pkcs12', passphrase=passphrase)
    assert base64.b64decode(res)

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_private_key_write_to_path(x509, encoding, tmp_path):
    if False:
        while True:
            i = 10
    tgt = tmp_path / 'pk'
    x509.create_private_key(encoding=encoding, path=str(tgt))
    assert tgt.exists()
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN PRIVATE KEY-----')

def test_create_private_key_write_to_path_encrypted(x509, tmp_path):
    if False:
        i = 10
        return i + 15
    tgt = tmp_path / 'pk'
    x509.create_private_key(path=str(tgt), passphrase='hunter1')
    assert tgt.exists()
    assert tgt.read_text().startswith('-----BEGIN ENCRYPTED PRIVATE KEY-----')

@pytest.mark.parametrize('encoding', ['pem', 'der'])
def test_create_private_key_write_to_path_overwrite(x509, encoding, tmp_path):
    if False:
        while True:
            i = 10
    tgt = tmp_path / 'cert'
    tgt.write_text('occupied')
    assert tgt.exists()
    x509.create_private_key(encoding=encoding, path=str(tgt))
    if encoding == 'pem':
        assert tgt.read_text().startswith('-----BEGIN PRIVATE KEY-----')

def test_create_private_key_raw(x509):
    if False:
        while True:
            i = 10
    res = x509.create_private_key(raw=True)
    assert isinstance(res, bytes)
    assert res.startswith(b'-----BEGIN PRIVATE KEY-----')

@pytest.mark.parametrize('algo,expected', [('rsa', 2048), ('ec', 256), ('ed25519', None), ('ed448', None)])
def test_get_private_key_size(x509, algo, expected, request):
    if False:
        return 10
    privkey = request.getfixturevalue(f'{algo}_privkey')
    res = x509.get_private_key_size(privkey)
    assert res == expected

@pytest.mark.parametrize('source', ['rsa_privkey', 'rsa_pubkey', 'cert_exts', 'csr_exts'])
def test_get_public_key(x509, source, request):
    if False:
        for i in range(10):
            print('nop')
    src = request.getfixturevalue(source)
    res = x509.get_public_key(src)
    assert res.startswith('-----BEGIN PUBLIC KEY-----')

def test_read_certificate(x509, cert_exts, cert_exts_read):
    if False:
        return 10
    res = x509.read_certificate(cert_exts)
    assert res == cert_exts_read

def test_read_crl(x509, crl):
    if False:
        while True:
            i = 10
    res = x509.read_crl(crl)
    assert res
    assert res == {'extensions': {}, 'issuer': OrderedDict([('C', 'US'), ('O', 'Salt'), ('CN', 'Test')]), 'last_update': '2022-11-16 04:10:08', 'next_update': '2023-02-24 04:10:08', 'revoked_certificates': {'0CF541984E976747B5E61C22D6542B56B3B4FD6F': {'extensions': {}, 'revocation_date': '2022-11-16 03:10:08'}}, 'signature_algorithm': 'sha256WithRSAEncryption'}

def test_read_crl_all(x509, crl_all):
    if False:
        for i in range(10):
            print('nop')
    res = x509.read_crl(crl_all)
    assert res == {'extensions': {'authorityKeyIdentifier': {'critical': False, 'issuer': None, 'issuer_sn': None, 'keyid': 'BE:12:01:CC:AA:EA:11:80:DA:2E:AD:B2:EA:C7:B5:FB:9F:F9:AD:34'}, 'cRLNumber': {'critical': False, 'value': 3}}, 'issuer': OrderedDict([('O', 'Sample Signer Organization'), ('OU', 'Sample Signer Unit'), ('CN', 'Sample Signer Cert')]), 'last_update': '2013-02-18 10:32:00', 'next_update': '2013-02-18 10:42:00', 'revoked_certificates': {'147947': {'extensions': {'CRLReason': {'critical': False, 'value': 'affiliationChanged'}, 'invalidityDate': {'critical': False, 'value': '2013-02-18 10:22:00'}}, 'revocation_date': '2013-02-18 10:22:12'}, '147948': {'extensions': {'CRLReason': {'critical': False, 'value': 'certificateHold'}, 'invalidityDate': {'critical': False, 'value': '2013-02-18 10:22:00'}}, 'revocation_date': '2013-02-18 10:22:22'}, '147949': {'extensions': {'CRLReason': {'critical': False, 'value': 'superseded'}, 'invalidityDate': {'critical': False, 'value': '2013-02-18 10:22:00'}}, 'revocation_date': '2013-02-18 10:22:32'}, '14794A': {'extensions': {'CRLReason': {'critical': False, 'value': 'keyCompromise'}, 'invalidityDate': {'critical': False, 'value': '2013-02-18 10:22:00'}}, 'revocation_date': '2013-02-18 10:22:42'}, '14794B': {'extensions': {'CRLReason': {'critical': False, 'value': 'cessationOfOperation'}, 'invalidityDate': {'critical': False, 'value': '2013-02-18 10:22:00'}}, 'revocation_date': '2013-02-18 10:22:51'}}, 'signature_algorithm': 'sha1WithRSAEncryption'}

def test_read_csr(x509, csr_exts, csr_exts_read):
    if False:
        while True:
            i = 10
    res = x509.read_csr(csr_exts)
    assert res == csr_exts_read

def test_verify_crl(x509, crl, ca_cert):
    if False:
        while True:
            i = 10
    assert x509.verify_crl(crl, ca_cert) is True

def test_encode_private_key(x509, rsa_privkey):
    if False:
        i = 10
        return i + 15
    pk = x509.create_private_key()
    res = x509.encode_private_key(pk)
    assert res.strip() == pk.strip()

def test_encode_private_key_encrypted(x509, ca_key, ca_key_enc):
    if False:
        for i in range(10):
            print('nop')
    pk = x509.create_private_key()
    pk_enc = x509.encode_private_key(pk, passphrase='hunter1')
    res = x509.encode_private_key(pk_enc, private_key_passphrase='hunter1')
    assert res.strip() == pk.strip()

@pytest.mark.parametrize('privkey,expected', [('ca_key', True), ('rsa_privkey', False)])
def test_verify_private_key(x509, request, privkey, expected, ca_cert):
    if False:
        return 10
    pk = request.getfixturevalue(privkey)
    assert x509.verify_private_key(pk, ca_cert) is expected

def test_verify_private_key_with_passphrase(x509, ca_key_enc, ca_cert):
    if False:
        return 10
    assert x509.verify_private_key(ca_key_enc, ca_cert, passphrase='correct horse battery staple') is True

@pytest.mark.parametrize('algo', ['rsa', 'ec', 'ed25519', 'ed448'])
def test_verify_signature(x509, algo, request):
    if False:
        while True:
            i = 10
    wrong_privkey = request.getfixturevalue(f'{algo}_privkey')
    privkey = x509.create_private_key(algo=algo)
    cert = x509.create_certificate(signing_private_key=privkey)
    assert x509.verify_signature(cert, privkey)
    assert not x509.verify_signature(cert, wrong_privkey)

def test_get_pem_entry(x509, ca_cert):
    if False:
        return 10
    res = x509.get_pem_entry(ca_cert)
    assert res == ca_cert.encode()

def test_get_pem_entry_newline_fix(x509, ca_cert):
    if False:
        i = 10
        return i + 15
    res = x509.get_pem_entry(ca_cert.replace('\n', ''))
    assert res == ca_cert.encode()

@pytest.fixture
def fresh_cert(x509, ca_key):
    if False:
        i = 10
        return i + 15
    return x509.create_certificate(signing_private_key=ca_key, days_valid=1, CN='fresh')

def test_expires(x509, fresh_cert):
    if False:
        return 10
    assert not x509.expires(fresh_cert)
    assert x509.expires(fresh_cert, 2)

def test_expired(x509, ca_key, fresh_cert, tmp_path):
    if False:
        print('Hello World!')
    tgt = tmp_path / 'pem'
    tgt.write_text(fresh_cert)
    res = x509.expired(str(tgt))
    assert res == {'cn': 'fresh', 'path': str(tgt), 'expired': False}
    old_cert = x509.create_certificate(signing_private_key=ca_key, not_before='2000-01-01 13:37:00', not_after='2000-01-01 13:37:42', CN='expired')
    res = x509.expired(old_cert)
    assert res == {'cn': 'expired', 'expired': True}

def test_will_expire(x509, fresh_cert):
    if False:
        for i in range(10):
            print('nop')
    assert x509.will_expire(fresh_cert, 0) == {'check_days': 0, 'cn': 'fresh', 'will_expire': False}
    assert x509.will_expire(fresh_cert, 2) == {'check_days': 2, 'cn': 'fresh', 'will_expire': True}

def test_write_pem(x509, fresh_cert, tmp_path):
    if False:
        print('Hello World!')
    tgt = tmp_path / 'write_pem'
    x509.write_pem(fresh_cert, str(tgt))
    assert tgt.exists()
    assert tgt.read_text() == fresh_cert

def test_get_pem_entries(x509, fresh_cert, ca_cert, tmp_path):
    if False:
        print('Hello World!')
    ca = tmp_path / 'ca'
    cert = tmp_path / 'cert'
    ca.write_text(ca_cert)
    cert.write_text(fresh_cert)
    res = x509.get_pem_entries(str(tmp_path / '*'))
    assert res
    assert res == {str(ca): ca_cert.encode(), str(cert): fresh_cert.encode()}

def test_read_certificates(x509, cert_exts, cert_exts_read, tmp_path):
    if False:
        return 10
    cert = tmp_path / 'cert'
    cert.write_text(cert_exts)
    res = x509.read_certificates(str(tmp_path / '*'))
    assert res
    assert res == {str(cert): cert_exts_read}

@pytest.mark.parametrize('arg', [{'version': 3}, {'serial_bits': 64}, {'text': True}])
def test_create_certificate_should_not_fail_with_removed_args(x509, arg, rsa_privkey):
    if False:
        while True:
            i = 10
    with pytest.deprecated_call():
        res = x509.create_certificate(signing_private_key=rsa_privkey, CN='success', days_valid=1, **arg)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

def test_create_certificate_warns_about_algorithm_renaming(x509, rsa_privkey):
    if False:
        for i in range(10):
            print('nop')
    with pytest.deprecated_call():
        res = x509.create_certificate(signing_private_key=rsa_privkey, days_valid=1, algorithm='sha512')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert isinstance(cert.signature_hash_algorithm, hashes.SHA512)

def test_create_certificate_warns_about_long_name_attributes(x509, rsa_privkey):
    if False:
        for i in range(10):
            print('nop')
    with pytest.deprecated_call():
        res = x509.create_certificate(signing_private_key=rsa_privkey, days_valid=1, commonName='success')
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert cert.subject.rfc4514_string() == 'CN=success'

def test_create_certificate_warns_about_long_extensions(x509, rsa_privkey):
    if False:
        i = 10
        return i + 15
    kwarg = {'X509v3 Basic Constraints': 'critical CA:TRUE, pathlen:1'}
    with pytest.deprecated_call():
        res = x509.create_certificate(signing_private_key=rsa_privkey, days_valid=1, **kwarg)
    assert res.startswith('-----BEGIN CERTIFICATE-----')
    cert = _get_cert(res)
    assert len(cert.extensions) == 1
    assert isinstance(cert.extensions[0].value, cx509.BasicConstraints)
    assert cert.extensions[0].critical
    assert cert.extensions[0].value.ca
    assert cert.extensions[0].value.path_length == 1

@pytest.mark.parametrize('arg', [{'version': 1}, {'text': True}])
def test_create_csr_should_not_fail_with_removed_args(x509, arg, rsa_privkey):
    if False:
        i = 10
        return i + 15
    with pytest.deprecated_call():
        res = x509.create_csr(private_key=rsa_privkey, CN='success', **arg)
    assert res.startswith('-----BEGIN CERTIFICATE REQUEST-----')

def test_create_csr_warns_about_algorithm_renaming(x509, rsa_privkey):
    if False:
        return 10
    with pytest.deprecated_call():
        res = x509.create_csr(private_key=rsa_privkey, algorithm='sha512')
    assert res.startswith('-----BEGIN CERTIFICATE REQUEST-----')
    csr = cx509.load_pem_x509_csr(res.encode())
    assert isinstance(csr.signature_hash_algorithm, hashes.SHA512)

def test_create_csr_warns_about_long_name_attributes(x509, rsa_privkey):
    if False:
        return 10
    with pytest.deprecated_call():
        res = x509.create_csr(private_key=rsa_privkey, commonName='success')
    assert res.startswith('-----BEGIN CERTIFICATE REQUEST-----')
    csr = cx509.load_pem_x509_csr(res.encode())
    assert csr.subject.rfc4514_string() == 'CN=success'

def test_create_csr_warns_about_long_extensions(x509, rsa_privkey):
    if False:
        while True:
            i = 10
    kwarg = {'X509v3 Basic Constraints': 'critical CA:FALSE'}
    with pytest.deprecated_call():
        res = x509.create_csr(private_key=rsa_privkey, **kwarg)
    assert res.startswith('-----BEGIN CERTIFICATE REQUEST-----')
    csr = cx509.load_pem_x509_csr(res.encode())
    assert len(csr.extensions) == 1
    assert isinstance(csr.extensions[0].value, cx509.BasicConstraints)
    assert csr.extensions[0].critical
    assert csr.extensions[0].value.ca is False
    assert csr.extensions[0].value.path_length is None

@pytest.mark.parametrize('arg', [{'text': True}])
def test_create_crl_should_not_fail_with_removed_args(x509, arg, crl_args):
    if False:
        print('Hello World!')
    crl_args['days_valid'] = 7
    with pytest.deprecated_call():
        res = x509.create_crl(**crl_args, **arg)
    assert res.startswith('-----BEGIN X509 CRL-----')

def test_create_crl_should_recognize_old_style_revoked(x509, crl_args, crl_revoked):
    if False:
        for i in range(10):
            print('nop')
    revoked = [{f'key_{i}': [{'serial_number': rev['serial_number']}]} for (i, rev) in enumerate(crl_revoked)]
    crl_args['revoked'] = revoked
    crl_args['days_valid'] = 7
    with pytest.deprecated_call():
        res = x509.create_crl(**crl_args)
    crl = cx509.load_pem_x509_crl(res.encode())
    assert len(crl) == len(crl_revoked)

def test_create_crl_should_recognize_old_style_reason(x509, crl_args):
    if False:
        return 10
    revoked = [{'key_1': [{'serial_number': '01337A'}, {'reason': 'keyCompromise'}]}]
    crl_args['revoked'] = revoked
    crl_args['days_valid'] = 7
    with pytest.deprecated_call():
        res = x509.create_crl(**crl_args)
    crl = cx509.load_pem_x509_crl(res.encode())
    assert len(crl) == 1
    rev = crl.get_revoked_certificate_by_serial_number(78714)
    assert rev
    assert rev.extensions
    assert len(rev.extensions) == 1
    assert isinstance(rev.extensions[0].value, cx509.CRLReason)

@pytest.mark.parametrize('arg', [{'cipher': 'aes_256_cbc'}, {'verbose': True}, {'text': True}])
def test_create_private_key_should_not_fail_with_removed_args(x509, arg, crl_args):
    if False:
        return 10
    with pytest.deprecated_call():
        res = x509.create_private_key(**arg)
    assert res.startswith('-----BEGIN PRIVATE KEY-----')

def test_create_private_key_warns_about_bits_renaming(x509):
    if False:
        return 10
    with pytest.deprecated_call():
        res = x509.create_private_key(bits=3072)
    pk = load_pem_private_key(res.encode(), None)
    assert pk.key_size == 3072

def test_get_public_key_should_not_fail_with_removed_arg(x509, rsa_privkey):
    if False:
        return 10
    with pytest.deprecated_call():
        res = x509.get_public_key(rsa_privkey, asObj=True)
    assert res.startswith('-----BEGIN PUBLIC KEY-----')

def test_get_signing_policy_warns_about_long_names(x509):
    if False:
        i = 10
        return i + 15
    with pytest.deprecated_call():
        res = x509.get_signing_policy('testdeprecatednamepolicy')
    assert res
    assert 'commonName' not in res
    assert 'CN' in res
    assert res['CN'] == 'deprecated'

def test_get_signing_policy_warns_about_long_exts(x509):
    if False:
        i = 10
        return i + 15
    with pytest.deprecated_call():
        res = x509.get_signing_policy('testdeprecatedextpolicy')
    assert res
    assert 'X509v3 Basic Constraints' not in res
    assert 'basicConstraints' in res
    assert res['basicConstraints'] == 'critical CA:FALSE'

def _get_cert(cert, encoding='pem', passphrase=None):
    if False:
        return 10
    if encoding == 'pem':
        if not isinstance(cert, bytes):
            cert = cert.encode()
        return cx509.load_pem_x509_certificate(cert)
    if encoding == 'der':
        if not isinstance(cert, bytes):
            cert = base64.b64decode(cert)
        return cx509.load_der_x509_certificate(cert)
    if encoding == 'pkcs7_pem':
        if not isinstance(cert, bytes):
            cert = cert.encode()
        return pkcs7.load_pem_pkcs7_certificates(cert)
    if encoding == 'pkcs7_der':
        if not isinstance(cert, bytes):
            cert = base64.b64decode(cert)
        return pkcs7.load_der_pkcs7_certificates(cert)
    if encoding == 'pkcs12':
        if not isinstance(cert, bytes):
            cert = base64.b64decode(cert)
        if passphrase is not None and (not isinstance(passphrase, bytes)):
            passphrase = passphrase.encode()
        return pkcs12.load_pkcs12(cert, passphrase)