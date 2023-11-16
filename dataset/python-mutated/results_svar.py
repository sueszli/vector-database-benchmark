"""
Test Results for the SVAR model. Obtained from R using svartest.R
"""

class SVARdataResults:

    def __init__(self):
        if False:
            print('Hello World!')
        self.A = [[1.0, 0.0, 0], [-0.506802245, 1.0, 0], [-5.53605652, 3.04117686, 1.0]]
        self.B = [[0.0075756676, 0.0, 0.0], [0.0, 0.00512051886, 0.0], [0.0, 0.0, 0.020708948]]