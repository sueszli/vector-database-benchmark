from cython.cimports.shrubbing import Shrubbery
import shrubbing

def main():
    if False:
        i = 10
        return i + 15
    sh: Shrubbery
    sh = shrubbing.standard_shrubbery()
    print('Shrubbery size is', sh.width, 'x', sh.length)