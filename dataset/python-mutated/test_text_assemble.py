from panda3d import core

def test_text_assemble_null():
    if False:
        while True:
            i = 10
    assembler = core.TextAssembler(core.TextEncoder())
    assembler.set_wtext(u'\x00test')
    assembler.assemble_text()