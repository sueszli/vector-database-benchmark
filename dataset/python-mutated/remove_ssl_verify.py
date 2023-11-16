import ssl

def remove_ssl_verify():
    if False:
        return 10
    ssl._create_default_https_context = ssl._create_unverified_context