import unittest
from decimal import Decimal
from borb.pdf.canvas.geometry.matrix import Matrix

class TestMatrixDeterminant(unittest.TestCase):
    """
    This test checks some of the methods of the Matrix object.
    """

    def test_matrix_determinant_001(self):
        if False:
            for i in range(10):
                print('nop')
        m0 = Matrix.identity_matrix()
        print(m0)
        assert m0.determinant() == Decimal(1)

    def test_matrix_determinant_002(self):
        if False:
            print('Hello World!')
        m0 = Matrix.identity_matrix()
        m0[0][0] = Decimal(0.453486)
        m0[0][1] = Decimal(0.286607)
        m0[0][2] = Decimal(0.803428)
        m0[1][0] = Decimal(0.69059)
        m0[1][1] = Decimal(0.877364)
        m0[1][2] = Decimal(0.555546)
        m0[2][0] = Decimal(0.00726739)
        m0[2][1] = Decimal(0.0278032)
        m0[2][2] = Decimal(0.421974)
        print(m0)
        det = m0.determinant()
        print(det)
        assert round(det, 10) == round(Decimal(0.08882747052120038), 10)