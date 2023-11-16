from MyQR.mylibs import data, ECC, structure, matrix, draw

def get_qrcode(ver, ecl, str, save_place):
    if False:
        i = 10
        return i + 15
    (ver, data_codewords) = data.encode(ver, ecl, str)
    ecc = ECC.encode(ver, ecl, data_codewords)
    final_bits = structure.structure_final_bits(ver, ecl, data_codewords, ecc)
    qrmatrix = matrix.get_qrmatrix(ver, ecl, final_bits)
    return (ver, draw.draw_qrcode(save_place, qrmatrix))