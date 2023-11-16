import unittest
import numpy as np
import face_alignment
import sys
import torch
sys.path.append('.')
from face_alignment.utils import get_image

class Tester(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.reference_data = [np.array([[137.0, 240.0, -85.907196], [140.0, 264.0, -81.1443], [143.0, 288.0, -76.25633], [146.0, 306.0, -69.01708], [152.0, 327.0, -53.775352], [161.0, 342.0, -30.029667], [170.0, 348.0, -2.792292], [185.0, 354.0, 23.522688], [212.0, 360.0, 38.664257], [239.0, 357.0, 31.747217], [263.0, 354.0, 12.192401], [284.0, 348.0, -10.0569725], [302.0, 333.0, -29.42916], [314.0, 315.0, -41.675602], [320.0, 297.0, -46.924263], [326.0, 276.0, -50.33218], [335.0, 252.0, -53.945686], [152.0, 207.0, -7.6189857], [164.0, 201.0, 6.1879144], [176.0, 198.0, 16.991247], [188.0, 198.0, 24.690582], [200.0, 201.0, 29.248188], [245.0, 204.0, 37.878166], [257.0, 201.0, 37.420483], [269.0, 201.0, 34.163113], [284.0, 204.0, 28.480812], [299.0, 216.0, 18.31863], [221.0, 225.0, 37.93351], [218.0, 237.0, 48.337395], [215.0, 249.0, 60.502884], [215.0, 261.0, 63.353687], [203.0, 273.0, 40.186855], [209.0, 276.0, 45.057003], [218.0, 276.0, 48.56715], [227.0, 276.0, 47.744766], [233.0, 276.0, 45.01401], [170.0, 228.0, 7.166072], [179.0, 222.0, 17.168053], [188.0, 222.0, 19.775822], [200.0, 228.0, 19.06176], [191.0, 231.0, 20.636724], [179.0, 231.0, 16.125824], [248.0, 231.0, 28.566122], [257.0, 225.0, 33.024036], [269.0, 225.0, 34.384735], [278.0, 231.0, 27.014532], [269.0, 234.0, 32.867023], [257.0, 234.0, 33.34033], [185.0, 306.0, 29.927242], [194.0, 297.0, 42.611233], [209.0, 291.0, 50.563396], [215.0, 291.0, 52.831104], [221.0, 291.0, 52.9225], [236.0, 300.0, 48.32575], [248.0, 309.0, 38.2375], [236.0, 312.0, 48.377922], [224.0, 315.0, 52.63793], [212.0, 315.0, 52.330444], [203.0, 315.0, 49.552994], [194.0, 309.0, 42.64459], [188.0, 303.0, 30.746407], [206.0, 300.0, 46.514435], [215.0, 300.0, 49.611156], [224.0, 300.0, 49.058918], [248.0, 309.0, 38.084103], [224.0, 303.0, 49.817806], [215.0, 303.0, 49.59815], [206.0, 303.0, 47.13894]], dtype=np.float32)]

    def test_predict_points(self):
        if False:
            while True:
                i = 10
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu')
        preds = fa.get_landmarks('test/assets/aflw-test.jpg')
        self.assertEqual(len(preds), len(self.reference_data))
        for (pred, reference) in zip(preds, self.reference_data):
            self.assertTrue(np.allclose(pred, reference))

    def test_predict_batch_points(self):
        if False:
            while True:
                i = 10
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu')
        reference_data = self.reference_data + self.reference_data
        reference_data.append([])
        image = get_image('test/assets/aflw-test.jpg')
        batch = np.stack([image, image, np.zeros_like(image)])
        batch = torch.Tensor(batch.transpose(0, 3, 1, 2))
        preds = fa.get_landmarks_from_batch(batch)
        self.assertEqual(len(preds), len(reference_data))
        for (pred, reference) in zip(preds, reference_data):
            self.assertTrue(np.allclose(pred, reference))

    def test_predict_points_from_dir(self):
        if False:
            for i in range(10):
                print('nop')
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu')
        reference_data = {'test/assets/grass.jpg': None, 'test/assets/aflw-test.jpg': self.reference_data}
        preds = fa.get_landmarks_from_directory('test/assets/')
        for (k, points) in preds.items():
            if isinstance(points, list):
                for (p, p_reference) in zip(points, reference_data[k]):
                    self.assertTrue(np.allclose(p, p_reference))
            else:
                self.assertEqual(points, reference_data[k])
if __name__ == '__main__':
    unittest.main()