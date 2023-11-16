from cryptography.hazmat.primitives import hashes

def ex2(user, pwtext):
    if False:
        for i in range(10):
            print('nop')
    md5 = hashes.MD5()
    user.setPassword(md5)