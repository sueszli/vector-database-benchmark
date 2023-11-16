import os
import argparse
import lzma
import pickle
from datetime import datetime
import numpy as np
import PIL.Image
import torch
import easyocr

def count_parameters(model):
    if False:
        print('Hello World!')
    return sum([param.numel() for param in model.parameters()])

def get_weight_norm(model):
    if False:
        i = 10
        return i + 15
    with torch.no_grad():
        return sum([param.norm() for param in model.parameters()]).cpu().item()

def replace(list_in, indices, values):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(indices, list):
        indices = [indices]
    if not isinstance(values, list):
        values = [values]
    assert len(indices) == len(values)
    list_out = list_in.copy()
    for (index, value) in zip(indices, values):
        list_out[index] = value
    return list_out

def get_easyocr(language):
    if False:
        i = 10
        return i + 15
    if not isinstance(language, list):
        language = [language]
    return easyocr.Reader(language)

def main(args):
    if False:
        for i in range(10):
            print('nop')
    if args.output is None:
        args.output = 'EasyOcrUnitTestPackage_{}.pickle'.format(datetime.now().strftime('%Y%m%dT%H%M'))
    if args.data_dir is None:
        data_dir = './examples'
    else:
        data_dir = args.data_dir
    image_preprocess = {'english.png': {'tiny': [540, 420, 690, 470], 'mini': [260, 90, 605, 160], 'small': [243, 234, 636, 360]}, 'french.jpg': {'tiny': [184, 615, 425, 732]}, 'chinese.jpg': {'tiny': [181, 78, 469, 157]}, 'korean.png': {'tiny': [130, 84, 285, 180]}}
    if any([file not in os.listdir(data_dir) for file in image_preprocess.keys()]):
        raise FileNotFoundError('Cannot find {} in {}.').format(', '.join([file for file in image_preprocess.keys() if file not in os.listdir(data_dir)], data_dir))
    easyocr_config = {'main_language': 'en'}
    ocr = get_easyocr(easyocr_config['main_language'])
    images = {os.path.splitext(file)[0]: {key: np.asarray(PIL.Image.open(os.path.join(data_dir, file)).crop(crop_box))[:, :, ::-1] for (key, crop_box) in page.items()} for (file, page) in image_preprocess.items()}
    (english_mini_bgr, english_mini_gray) = easyocr.utils.reformat_input(images['english']['mini'])
    (english_small_bgr, english_small_gray) = easyocr.utils.reformat_input(images['english']['small'])
    model_init_test = {'test01': {'description': 'Counting parameters of detector module.', 'method': 'unit_test.count_parameters', 'input': ['unit_test.easyocr.ocr.detector'], 'output': count_parameters(ocr.detector), 'severity': 'Error'}, 'test02': {'description': 'Calculating total norm of parameters in detector module.', 'method': 'unit_test.get_weight_norm', 'input': ['unit_test.easyocr.ocr.detector'], 'output': get_weight_norm(ocr.detector), 'severity': 'Warning'}, 'test03': {'description': 'Counting parameters of recognition module.', 'method': 'unit_test.count_parameters', 'input': ['unit_test.easyocr.ocr.recognizer'], 'output': count_parameters(ocr.recognizer), 'severity': 'Error'}, 'test04': {'description': 'Calculating total norm of parameters in recognition module.', 'method': 'unit_test.get_weight_norm', 'input': ['unit_test.easyocr.ocr.recognizer'], 'output': get_weight_norm(ocr.recognizer), 'severity': 'Warning'}}
    get_textbox_test = {}
    input0 = [ocr.detector, english_mini_bgr, 2560, 1.0, 0.7, 0.4, 0.4, False, 'cuda']
    get_textbox_test.update({'test01': {'description': 'Testing with default input.', 'method': 'unit_test.easyocr.detection.get_textbox', 'input': replace(input0, [0, 1], ['unit_test.easyocr.ocr.detector', 'unit_test.inputs.images.english.mini_bgr']), 'output': easyocr.detection.get_textbox(*input0), 'severity': 'Error'}})
    input0 = [ocr.detector, english_mini_bgr, 1280, 1.2, 0.6, 0.3, 0.3, False, 'cuda']
    get_textbox_test.update({'test02': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.detection.get_textbox', 'input': replace(input0, [0, 1], ['unit_test.easyocr.ocr.detector', 'unit_test.inputs.images.english.mini_bgr']), 'output': easyocr.detection.get_textbox(*input0), 'severity': 'Error'}})
    input0 = [ocr.detector, english_mini_bgr, 640, 0.8, 0.8, 0.5, 0.5, False, 'cuda']
    get_textbox_test.update({'test03': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.detection.get_textbox', 'input': replace(input0, [0, 1], ['unit_test.easyocr.ocr.detector', 'unit_test.inputs.images.english.mini_bgr']), 'output': easyocr.detection.get_textbox(*input0), 'severity': 'Error'}})
    input0 = [ocr.detector, english_mini_bgr, 2560, 1.0, 0.7, 0.4, 0.4, False, 'cuda']
    output0 = easyocr.detection.get_textbox(*input0)
    polys = output0[0]
    group_text_box_test = {}
    input_ = [polys, 0.1, 0.5, 0.5, 1.0, 0.05, True]
    group_text_box_test.update({'test01': {'description': 'Testing with default input.', 'method': 'unit_test.easyocr.utils.group_text_box', 'input': input_, 'output': easyocr.utils.group_text_box(*input_), 'severity': 'Error'}})
    input_ = [polys, 0.05, 0.3, 0.3, 0.8, 0.03, True]
    group_text_box_test.update({'test02': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.utils.group_text_box', 'input': input_, 'output': easyocr.utils.group_text_box(*input_), 'severity': 'Error'}})
    input_ = [polys, 0.12, 0.7, 0.7, 1.2, 0.1, True]
    group_text_box_test.update({'test03': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.utils.group_text_box', 'input': input_, 'output': easyocr.utils.group_text_box(*input_), 'severity': 'Error'}})
    input0 = [None, 20, 0.7, 0.4, 0.4, 2560, 1.0, 0.1, 0.5, 0.5, 0.5, 0.1, True, None]
    detect_test = {}
    input_ = replace(input0, [0, 1], [english_mini_bgr, 20])
    detect_test.update({'test01': {'description': 'Testing with default input.', 'method': 'unit_test.easyocr.ocr.detect', 'input': replace(input_, 0, 'unit_test.inputs.images.english.mini_bgr'), 'output': ocr.detect(*input_), 'severity': 'Error'}})
    input_ = replace(input0, [0, 1], [english_small_bgr, 20])
    detect_test.update({'test02': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.ocr.detect', 'input': replace(input_, 0, 'unit_test.inputs.images.english.small_bgr'), 'output': ocr.detect(*input_), 'severity': 'Error'}})
    input_ = replace(input0, [0, 1], [english_small_bgr, 100])
    detect_test.update({'test03': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.ocr.detect', 'input': replace(input_, 0, 'unit_test.inputs.images.english.small_bgr'), 'output': ocr.detect(*input_), 'severity': 'Error'}})
    get_image_list_test = {}
    output0 = ocr.detect(english_small_bgr)
    input0 = [output0[0][0], output0[1][0], english_small_gray, 64, True]
    input_ = replace(input0, 2, 'unit_test.inputs.images.english.small_gray')
    get_image_list_test.update({'test01': {'description': 'Testing with default input.', 'method': 'unit_test.easyocr.utils.get_image_list', 'input': input_, 'output': easyocr.utils.get_image_list(*input0), 'severity': 'Error'}})
    output0 = ocr.detect(english_mini_bgr)
    input0 = [output0[0][0], output0[1][0], english_mini_gray, 64, True]
    input_ = replace(input0, 2, 'unit_test.inputs.images.english.mini_gray')
    get_image_list_test.update({'test02': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.utils.get_image_list', 'input': input_, 'output': easyocr.utils.get_image_list(*input0), 'severity': 'Error'}})
    output0 = ocr.detect(english_mini_bgr)
    input0 = [output0[0][0], output0[1][0], english_mini_gray, 64, True]
    (image_list, max_width) = easyocr.utils.get_image_list(*input0)
    input0 = [ocr.character, 64, int(max_width), ocr.recognizer, ocr.converter, image_list[:2], '', 'greedy', 5, 1, 0.1, 0.5, 0.003, 1, 'cuda']
    get_text_test = {}
    output_ = easyocr.recognition.get_text(*input0)
    input_ = replace(input0, [0, 3, 4], ['unit_test.easyocr.ocr.character', 'unit_test.easyocr.ocr.recognizer', 'unit_test.easyocr.ocr.converter'])
    get_text_test.update({'test01': {'description': 'Testing with default input.', 'method': 'unit_test.easyocr.recognition.get_text', 'input': input_, 'output': output_, 'severity': 'Error'}})
    input0 = [ocr.character, 64, int(max_width), ocr.recognizer, ocr.converter, image_list[:2], '', 'greedy', 4, 1, 0.05, 0.3, 0.001, 1, 'cuda']
    output_ = easyocr.recognition.get_text(*input0)
    input_ = replace(input0, [0, 3, 4], ['unit_test.easyocr.ocr.character', 'unit_test.easyocr.ocr.recognizer', 'unit_test.easyocr.ocr.converter'])
    get_text_test.update({'test02': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.recognition.get_text', 'input': input_, 'output': output_, 'severity': 'Error'}})
    input0 = [ocr.character, 64, int(max_width), ocr.recognizer, ocr.converter, image_list[:2], '', 'greedy', 6, 4, 0.2, 0.6, 0.005, 1, 'cuda']
    output_ = easyocr.recognition.get_text(*input0)
    input_ = replace(input0, [0, 3, 4], ['unit_test.easyocr.ocr.character', 'unit_test.easyocr.ocr.recognizer', 'unit_test.easyocr.ocr.converter'])
    get_text_test.update({'test03': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.recognition.get_text', 'input': input_, 'output': output_, 'severity': 'Error'}})
    get_paragraph_test = {}
    output0 = ocr.detect(english_mini_bgr)
    input0 = [output0[0][0], output0[1][0], english_mini_gray, 64, True]
    (image_list, max_width) = easyocr.utils.get_image_list(*input0)
    input0 = [ocr.character, 64, int(max_width), ocr.recognizer, ocr.converter, image_list[:2], '', 'greedy', 5, 1, 0.1, 0.5, 0.003, 1, 'cuda']
    output0 = easyocr.recognition.get_text(*input0)
    input_ = [output0, 1, 0.5, 'ltr']
    get_paragraph_test.update({'test01': {'description': 'Testing with default input.', 'method': 'unit_test.easyocr.utils.get_paragraph', 'input': input_, 'output': easyocr.utils.get_paragraph(*input_), 'severity': 'Error'}})
    input_ = [output0, 0.5, 0.3, 'ltr']
    get_paragraph_test.update({'test02': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.utils.get_paragraph', 'input': input_, 'output': easyocr.utils.get_paragraph(*input_), 'severity': 'Error'}})
    input_ = [output0, 1.5, 1, 'ltr']
    get_paragraph_test.update({'test03': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.utils.get_paragraph', 'input': input_, 'output': easyocr.utils.get_paragraph(*input_), 'severity': 'Error'}})
    input_recog = [None, None, None, 'greedy', 5, 1, 0, None, None, 1, None, False, 0.1, 0.5, 0.003, 0.5, 1.0, True, 'standard']
    recognize_test = {}
    (h_list, f_list) = ocr.detect(english_mini_bgr)
    input_ = replace(input_recog, [0, 1, 2], [english_mini_gray, h_list[0], f_list[0]])
    recognize_test.update({'test01': {'description': 'Testing with default input.', 'method': 'unit_test.easyocr.ocr.recognize', 'input': replace(input_, 0, 'unit_test.inputs.images.english.mini_gray'), 'output': ocr.recognize(*input_), 'severity': 'Error'}})
    (h_list, f_list) = ocr.detect(english_small_bgr)
    input_ = replace(input_recog, [0, 1, 2], [english_small_gray, h_list[0], f_list[0]])
    recognize_test.update({'test02': {'description': 'Testing with custom input.', 'method': 'unit_test.easyocr.ocr.recognize', 'input': replace(input_, 0, 'unit_test.inputs.images.english.small_gray'), 'output': ocr.recognize(*input_), 'severity': 'Error'}})
    readtext_test = {}
    input_ = ['unit_test.inputs.images.english.tiny', 'en']
    ocr = get_easyocr('en')
    (_, pred, confidence) = ocr.readtext(images['english']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test01': {'description': 'Reading English text.', 'method': 'unit_test.easyocr_read_as', 'input': input_, 'output': output_, 'severity': 'Error'}})
    input_ = ['unit_test.inputs.images.french.tiny', 'fr']
    ocr = get_easyocr('fr')
    (_, pred, confidence) = ocr.readtext(images['french']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test02': {'description': 'Reading French text.', 'method': 'unit_test.easyocr_read_as', 'input': input_, 'output': output_, 'severity': 'Error'}})
    input_ = ['unit_test.inputs.images.chinese.tiny', 'ch_sim']
    ocr = get_easyocr('ch_sim')
    (_, pred, confidence) = ocr.readtext(images['chinese']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test03': {'description': 'Reading Chinese (simplified) text.', 'method': 'unit_test.easyocr_read_as', 'input': input_, 'output': output_, 'severity': 'Error'}})
    input_ = ['unit_test.inputs.images.korean.tiny', 'ko']
    ocr = get_easyocr('ko')
    (_, pred, confidence) = ocr.readtext(images['korean']['tiny'])[0]
    output_ = [pred, confidence]
    readtext_test.update({'test04': {'description': 'Reading Korean text.', 'method': 'unit_test.easyocr_read_as', 'input': input_, 'output': output_, 'severity': 'Error'}})
    solution_book = {'inputs': {'images': image_preprocess, 'easyocr_config': easyocr_config}, 'tests': {'model initialization': model_init_test, 'get_textbox function': get_textbox_test, 'group_text_box function': group_text_box_test, 'detect method': detect_test, 'get_image_list function': get_image_list_test, 'get_text_test function': get_text_test, 'get_paragraph_test function': get_paragraph_test, 'recognize method': recognize_test, 'readtext method': readtext_test}}
    with lzma.open(args.output, 'wb') as fid:
        pickle.dump(solution_book, fid)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to pack EasyOCR weight.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default=None, help='output path.')
    parser.add_argument('-d', '--data_dir', default=None, help='data directory')
    args = parser.parse_args()
    main(args)