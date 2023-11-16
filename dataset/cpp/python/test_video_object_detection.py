import os, sys
from typing import List
from numpy import ndarray
from os.path import dirname
from mock import patch
sys.path.insert(1, os.path.join(dirname(dirname(os.path.abspath(__file__)))))

from imageai.Detection import VideoObjectDetection


test_folder = dirname(os.path.abspath(__file__))

video_file = os.path.join(test_folder, "data-videos", "traffic-micro.mp4")
video_file_output = os.path.join(test_folder, "data-videos", "traffic-micro-detected")



class CallbackFunctions:
    def forFrame(frame_number, output_array, output_count, detected_frame):
        assert isinstance(detected_frame, ndarray)
        assert isinstance(frame_number, int)
        assert isinstance(output_array, list)
        assert isinstance(output_array[0], dict)
        assert isinstance(output_array[0]["name"], str)
        assert isinstance(output_array[0]["percentage_probability"], float)
        assert isinstance(output_array[0]["box_points"], list)

        assert isinstance(output_count, dict)
        for a_key in dict(output_count).keys():
            assert isinstance(a_key, str)
            assert isinstance(output_count[a_key], int)

    def forSecond(second_number, output_arrays, count_arrays, average_output_count, detected_frame):
        assert isinstance(detected_frame, ndarray)
        assert isinstance(second_number, int)
        assert isinstance(output_arrays, list)
        assert isinstance(output_arrays[0], list)

        assert isinstance(output_arrays[0][0], dict)
        assert isinstance(output_arrays[0][0]["name"], str)
        assert isinstance(output_arrays[0][0]["percentage_probability"], float)
        assert isinstance(output_arrays[0][0]["box_points"], list)

        assert isinstance(count_arrays, list)
        assert isinstance(count_arrays[0], dict)
        for a_key in dict(count_arrays[0]).keys():
            assert isinstance(a_key, str)
            assert isinstance(count_arrays[0][a_key], int)

        assert isinstance(average_output_count, dict)
        for a_key2 in dict(average_output_count).keys():
            assert isinstance(a_key2, str)
            assert isinstance(average_output_count[a_key2], int)



def delete_cache(files: List[str]):
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


def test_video_detection_retinanet():

    delete_cache([video_file_output + ".mp4"])

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path=os.path.join(test_folder, "data-models", "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()
    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True)

    assert os.path.exists(video_file_output + ".mp4")
    assert isinstance(video_path, str)

    delete_cache([video_file_output + ".mp4"])


def test_video_detection_retinanet_custom_objects():

    delete_cache([video_file_output + ".mp4"])

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path=os.path.join(test_folder, "data-models", "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(
        person=True,
        bus=True
    )

    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True, custom_objects=custom_objects)

    assert os.path.exists(video_file_output + ".mp4")
    assert isinstance(video_path, str)

    delete_cache([video_file_output + ".mp4"])




def test_video_detection_yolov3():
    delete_cache([video_file_output + ".mp4"])

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(model_path=os.path.join(test_folder, "data-models", "yolov3.pt"))
    detector.loadModel()
    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True)

    assert os.path.exists(video_file_output + ".mp4")
    assert isinstance(video_path, str)
    
    delete_cache([video_file_output + ".mp4"])



def test_video_detection_tiny_yolov3():
    delete_cache([video_file_output + ".mp4"])

    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path=os.path.join(test_folder, "data-models", "tiny-yolov3.pt"))
    detector.loadModel()
    video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True)

    assert os.path.exists(video_file_output + ".mp4")
    assert isinstance(video_path, str)

    delete_cache([video_file_output + ".mp4"])


def test_video_detection_retinanet_analysis():
    delete_cache([video_file_output + ".mp4"])

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path=os.path.join(test_folder, "data-models", "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()

    with patch.object(CallbackFunctions, 'forFrame') as frameFunc:
        with patch.object(CallbackFunctions, 'forSecond') as secondFunc:

            video_path = detector.detectObjectsFromVideo(input_file_path=video_file, output_file_path=video_file_output, save_detected_video=True, frames_per_second=30, log_progress=True, per_frame_function=frameFunc, per_second_function=secondFunc, return_detected_frame=True)

            assert os.path.exists(video_file_output + ".mp4")
            assert isinstance(video_path, str)

            frameFunc.assert_called()
            secondFunc.assert_called()

    delete_cache([video_file_output + ".mp4"])


