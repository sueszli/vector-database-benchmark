def whcms_license_info(md5hash, datahash, results):
    if False:
        for i in range(10):
            print('nop')
    if md5hash == datahash:
        try:
            return md5hash
        except:
            return results