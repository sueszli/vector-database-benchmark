import unittest
import platform

class SysfontModuleTest(unittest.TestCase):

    def test_create_aliases(self):
        if False:
            return 10
        import pygame.sysfont
        pygame.sysfont.initsysfonts()
        pygame.sysfont.create_aliases()
        self.assertTrue(len(pygame.sysfont.Sysalias) > 0)

    def test_initsysfonts(self):
        if False:
            return 10
        import pygame.sysfont
        pygame.sysfont.initsysfonts()
        self.assertTrue(len(pygame.sysfont.get_fonts()) > 0)

    @unittest.skipIf('Darwin' not in platform.platform(), 'Not mac we skip.')
    def test_initsysfonts_darwin(self):
        if False:
            for i in range(10):
                print('nop')
        import pygame.sysfont
        self.assertTrue(len(pygame.sysfont.get_fonts()) > 10)

    def test_sysfont(self):
        if False:
            for i in range(10):
                print('nop')
        import pygame.font
        pygame.font.init()
        arial = pygame.font.SysFont('Arial', 40)
        self.assertTrue(isinstance(arial, pygame.font.Font))

    @unittest.skipIf('Darwin' in platform.platform() or 'Windows' in platform.platform(), 'Not unix we skip.')
    def test_initsysfonts_unix(self):
        if False:
            print('Hello World!')
        import pygame.sysfont
        self.assertTrue(len(pygame.sysfont.get_fonts()) > 0)

    @unittest.skipIf('Windows' not in platform.platform(), 'Not windows we skip.')
    def test_initsysfonts_win32(self):
        if False:
            for i in range(10):
                print('nop')
        import pygame.sysfont
        self.assertTrue(len(pygame.sysfont.get_fonts()) > 10)
if __name__ == '__main__':
    unittest.main()