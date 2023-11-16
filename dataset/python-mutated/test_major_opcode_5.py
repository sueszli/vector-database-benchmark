from ut_helpers_asm import check_instruction

class TestMajor5(object):

    def test_MOV(self):
        if False:
            while True:
                i = 10
        'Test the MOV instruction'
        check_instruction('MOV $2, 0', '5200', multi=2)
        check_instruction('MOV $12, 0', '5c00', multi=2)
        check_instruction('MOV $4, 0', '5400', multi=2)
        check_instruction('MOV $0, 0', '5000', multi=2)
        check_instruction('MOV $0, 3', '5003', multi=2)
        check_instruction('MOV $8, 84', '5854', multi=2)
        check_instruction('MOV $SP, 108', '5f6c', multi=2)
        check_instruction('MOV $12, 80', '5c50', multi=2)
        check_instruction('MOV $TP, 59', '5d3b', multi=2)
        check_instruction('MOV $9, 89', '5959', multi=2)