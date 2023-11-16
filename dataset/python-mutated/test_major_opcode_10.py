from ut_helpers_asm import check_instruction

class TestMajor10(object):

    def test_BEQZ(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the BEQZ instruction'
        check_instruction('BEQZ $11, 0xFFFFFF8C', 'ab8c')
        check_instruction('BEQZ $4, 0x6', 'a406')
        check_instruction('BEQZ $TP, 0x4', 'ad04')
        check_instruction('BEQZ $11, 0x4', 'ab04')
        check_instruction('BEQZ $12, 0xA', 'ac0a')
        check_instruction('BEQZ $0, 0x42', 'a042')
        check_instruction('BEQZ $10, 0x6', 'aa06')
        check_instruction('BEQZ $0, 0x8', 'a008')
        check_instruction('BEQZ $12, 0x4', 'ac04')
        check_instruction('BEQZ $1, 0x70', 'a170')

    def test_BNEZ(self):
        if False:
            return 10
        'Test the BNEZ instruction'
        check_instruction('BNEZ $7, 0x46', 'a747')
        check_instruction('BNEZ $0, 0x40', 'a041')
        check_instruction('BNEZ $9, 0x1C', 'a91d')
        check_instruction('BNEZ $0, 0xFFFFFFF6', 'a0f7')
        check_instruction('BNEZ $4, 0xA', 'a40b')
        check_instruction('BNEZ $7, 0xE', 'a70f')
        check_instruction('BNEZ $11, 0xE', 'ab0f')
        check_instruction('BNEZ $10, 0x28', 'aa29')
        check_instruction('BNEZ $9, 0xFFFFFFAE', 'a9af')
        check_instruction('BNEZ $9, 0xE', 'a90f')