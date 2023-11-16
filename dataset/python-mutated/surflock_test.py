import unittest
import sys
import platform
import pygame
IS_PYPY = 'PyPy' == platform.python_implementation()

@unittest.skipIf(IS_PYPY, 'pypy skip known failure')
class SurfaceLockTest(unittest.TestCase):

    def test_lock(self):
        if False:
            while True:
                i = 10
        sf = pygame.Surface((5, 5))
        sf.lock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (sf,))
        sf.lock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (sf, sf))
        sf.unlock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (sf,))
        sf.unlock()
        self.assertEqual(sf.get_locked(), False)
        self.assertEqual(sf.get_locks(), ())

    def test_subsurface_lock(self):
        if False:
            return 10
        sf = pygame.Surface((5, 5))
        subsf = sf.subsurface((1, 1, 2, 2))
        sf2 = pygame.Surface((5, 5))
        sf2.blit(subsf, (0, 0))
        sf2.blit(sf, (0, 0))
        self.assertRaises(pygame.error, sf.blit, subsf, (0, 0))
        sf.lock()
        sf2.blit(subsf, (0, 0))
        self.assertRaises(pygame.error, sf2.blit, sf, (0, 0))
        subsf.lock()
        self.assertRaises(pygame.error, sf2.blit, subsf, (0, 0))
        self.assertRaises(pygame.error, sf2.blit, sf, (0, 0))
        sf.unlock()
        self.assertRaises(pygame.error, sf2.blit, subsf, (0, 0))
        self.assertRaises(pygame.error, sf2.blit, sf, (0, 0))
        sf.unlock()
        self.assertRaises(pygame.error, sf2.blit, sf, (0, 0))
        self.assertRaises(pygame.error, sf2.blit, subsf, (0, 0))
        subsf.unlock()
        sf.lock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (sf,))
        self.assertEqual(subsf.get_locked(), False)
        self.assertEqual(subsf.get_locks(), ())
        subsf.lock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (sf, subsf))
        self.assertEqual(subsf.get_locked(), True)
        self.assertEqual(subsf.get_locks(), (subsf,))
        sf.unlock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (subsf,))
        self.assertEqual(subsf.get_locked(), True)
        self.assertEqual(subsf.get_locks(), (subsf,))
        subsf.unlock()
        self.assertEqual(sf.get_locked(), False)
        self.assertEqual(sf.get_locks(), ())
        self.assertEqual(subsf.get_locked(), False)
        self.assertEqual(subsf.get_locks(), ())
        subsf.lock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (subsf,))
        self.assertEqual(subsf.get_locked(), True)
        self.assertEqual(subsf.get_locks(), (subsf,))
        subsf.lock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (subsf, subsf))
        self.assertEqual(subsf.get_locked(), True)
        self.assertEqual(subsf.get_locks(), (subsf, subsf))

    def test_pxarray_ref(self):
        if False:
            print('Hello World!')
        sf = pygame.Surface((5, 5))
        ar = pygame.PixelArray(sf)
        ar2 = pygame.PixelArray(sf)
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (ar, ar2))
        del ar
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (ar2,))
        ar = ar2[:]
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (ar2,))
        del ar
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(len(sf.get_locks()), 1)

    def test_buffer(self):
        if False:
            return 10
        sf = pygame.Surface((5, 5))
        buf = sf.get_buffer()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (buf,))
        sf.unlock()
        self.assertEqual(sf.get_locked(), True)
        self.assertEqual(sf.get_locks(), (buf,))
        del buf
        self.assertEqual(sf.get_locked(), False)
        self.assertEqual(sf.get_locks(), ())
if __name__ == '__main__':
    unittest.main()