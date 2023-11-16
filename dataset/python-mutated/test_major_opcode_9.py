from ut_helpers_asm import check_instruction

class TestMajor9(object):

    def test_ADD3(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the ADD3 instruction'
        check_instruction('ADD3 $10, $4, $0', '940a')
        check_instruction('ADD3 $3, $0, $0', '9003')
        check_instruction('ADD3 $12, $4, $0', '940c')
        check_instruction('ADD3 $7, $12, $0', '9c07')
        check_instruction('ADD3 $TP, $4, $0', '940d')
        check_instruction('ADD3 $4, $1, $9', '9194')
        check_instruction('ADD3 $7, $12, $9', '9c97')
        check_instruction('ADD3 $12, $9, $SP', '99fc')
        check_instruction('ADD3 $12, $TP, $7', '9d7c')
        check_instruction('ADD3 $4, $8, $SP', '98f4')