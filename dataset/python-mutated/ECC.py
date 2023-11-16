from MyQR.mylibs.constant import GP_list, ecc_num_per_block, lindex, po2, log

def encode(ver, ecl, data_codewords):
    if False:
        print('Hello World!')
    en = ecc_num_per_block[ver - 1][lindex[ecl]]
    ecc = []
    for dc in data_codewords:
        ecc.append(get_ecc(dc, en))
    return ecc

def get_ecc(dc, ecc_num):
    if False:
        return 10
    gp = GP_list[ecc_num]
    remainder = dc
    for i in range(len(dc)):
        remainder = divide(remainder, *gp)
    return remainder

def divide(MP, *GP):
    if False:
        while True:
            i = 10
    if MP[0]:
        GP = list(GP)
        for i in range(len(GP)):
            GP[i] += log[MP[0]]
            if GP[i] > 255:
                GP[i] %= 255
            GP[i] = po2[GP[i]]
        return XOR(GP, *MP)
    else:
        return XOR([0] * len(GP), *MP)

def XOR(GP, *MP):
    if False:
        i = 10
        return i + 15
    MP = list(MP)
    a = len(MP) - len(GP)
    if a < 0:
        MP += [0] * -a
    elif a > 0:
        GP += [0] * a
    remainder = []
    for i in range(1, len(MP)):
        remainder.append(MP[i] ^ GP[i])
    return remainder