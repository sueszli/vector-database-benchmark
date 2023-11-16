""" voxceleb 1 & 2 """
import hashlib
import os
import subprocess
import sys
import zipfile
import pandas
import soundfile as sf
from absl import logging
SUBSETS = {'vox1_dev_wav': ['https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad'], 'vox1_test_wav': ['https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip'], 'vox2_dev_aac': ['https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag', 'https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah'], 'vox2_test_aac': ['https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip']}
MD5SUM = {'vox1_dev_wav': 'ae63e55b951748cc486645f532ba230b', 'vox2_dev_aac': 'bbc063c46078a602ca71605645c2a402', 'vox1_test_wav': '185fdc63c3c739954633d50379a3d102', 'vox2_test_aac': '0d2b3ea430a821c33263b5ea37ede312'}
USER = {'user': '', 'password': ''}
speaker_id_dict = {}

def download_and_extract(directory, subset, urls):
    if False:
        while True:
            i = 10
    'Download and extract the given split of dataset.\n\n    Args:\n        directory: the directory where to put the downloaded data.\n        subset: subset name of the corpus.\n        urls: the list of urls to download the data file.\n    '
    os.makedirs(directory, exist_ok=True)
    try:
        for url in urls:
            zip_filepath = os.path.join(directory, url.split('/')[-1])
            if os.path.exists(zip_filepath):
                continue
            logging.info('Downloading %s to %s' % (url, zip_filepath))
            subprocess.call('wget %s --user %s --password %s -O %s' % (url, USER['user'], USER['password'], zip_filepath), shell=True)
            statinfo = os.stat(zip_filepath)
            logging.info('Successfully downloaded %s, size(bytes): %d' % (url, statinfo.st_size))
        if '.zip' not in zip_filepath:
            zip_filepath = '_'.join(zip_filepath.split('_')[:-1])
            subprocess.call('cat %s* > %s.zip' % (zip_filepath, zip_filepath), shell=True)
            zip_filepath += '.zip'
        extract_path = zip_filepath.strip('.zip')
        with open(zip_filepath, 'rb') as f_zip:
            md5 = hashlib.md5(f_zip.read()).hexdigest()
        if md5 != MD5SUM[subset]:
            raise ValueError('md5sum of %s mismatch' % zip_filepath)
        with zipfile.ZipFile(zip_filepath, 'r') as zfile:
            zfile.extractall(directory)
            extract_path_ori = os.path.join(directory, zfile.infolist()[0].filename)
            subprocess.call('mv %s %s' % (extract_path_ori, extract_path), shell=True)
    finally:
        pass

def exec_cmd(cmd):
    if False:
        i = 10
        return i + 15
    'Run a command in a subprocess.\n    Args:\n        cmd: command line to be executed.\n    Return:\n        int, the return code.\n    '
    try:
        retcode = subprocess.call(cmd, shell=True)
        if retcode < 0:
            logging.info(f'Child was terminated by signal {retcode}')
    except OSError as e:
        logging.info(f'Execution failed: {e}')
        retcode = -999
    return retcode

def decode_aac_with_ffmpeg(aac_file, wav_file):
    if False:
        for i in range(10):
            print('nop')
    'Decode a given AAC file into WAV using ffmpeg.\n    Args:\n        aac_file: file path to input AAC file.\n        wav_file: file path to output WAV file.\n    Return:\n        bool, True if success.\n    '
    cmd = f'ffmpeg -i {aac_file} {wav_file}'
    logging.info(f'Decoding aac file using command line: {cmd}')
    ret = exec_cmd(cmd)
    if ret != 0:
        logging.error(f'Failed to decode aac file with retcode {ret}')
        logging.error('Please check your ffmpeg installation.')
        return False
    return True

def convert_audio_and_make_label(input_dir, subset, output_dir, output_file):
    if False:
        return 10
    'Optionally convert AAC to WAV and make speaker labels.\n    Args:\n        input_dir: the directory which holds the input dataset.\n        subset: the name of the specified subset. e.g. vox1_dev_wav\n        output_dir: the directory to place the newly generated csv files.\n        output_file: the name of the newly generated csv file. e.g. vox1_dev_wav.csv\n    '
    logging.info('Preprocessing audio and label for subset %s' % subset)
    source_dir = os.path.join(input_dir, subset)
    files = []
    for (root, _, filenames) in os.walk(source_dir):
        for filename in filenames:
            (name, ext) = os.path.splitext(filename)
            if ext.lower() == '.wav':
                (_, ext2) = os.path.splitext(name)
                if ext2:
                    continue
                wav_file = os.path.join(root, filename)
            elif ext.lower() == '.m4a':
                aac_file = os.path.join(root, filename)
                wav_file = aac_file + '.wav'
                if not os.path.exists(wav_file):
                    if not decode_aac_with_ffmpeg(aac_file, wav_file):
                        raise RuntimeError('Audio decoding failed.')
            else:
                continue
            speaker_name = root.split(os.path.sep)[-2]
            if speaker_name not in speaker_id_dict:
                num = len(speaker_id_dict)
                speaker_id_dict[speaker_name] = num
            wav_length = len(sf.read(wav_file)[0])
            files.append((os.path.abspath(wav_file), wav_length, speaker_id_dict[speaker_name], speaker_name))
    csv_file_path = os.path.join(output_dir, output_file)
    df = pandas.DataFrame(data=files, columns=['wav_filename', 'wav_length_ms', 'speaker_id', 'speaker_name'])
    df.to_csv(csv_file_path, index=False, sep='\t')
    logging.info('Successfully generated csv file {}'.format(csv_file_path))

def processor(directory, subset, force_process):
    if False:
        i = 10
        return i + 15
    'download and process'
    urls = SUBSETS
    if subset not in urls:
        raise ValueError(subset, 'is not in voxceleb')
    subset_csv = os.path.join(directory, subset + '.csv')
    if not force_process and os.path.exists(subset_csv):
        return subset_csv
    logging.info('Downloading and process the voxceleb in %s', directory)
    logging.info('Preparing subset %s', subset)
    download_and_extract(directory, subset, urls[subset])
    convert_audio_and_make_label(directory, subset, directory, subset + '.csv')
    logging.info('Finished downloading and processing')
    return subset_csv
if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) != 4:
        print('Usage: python prepare_data.py save_directory user password')
        sys.exit()
    (DIR, USER['user'], USER['password']) = (sys.argv[1], sys.argv[2], sys.argv[3])
    for SUBSET in SUBSETS:
        processor(DIR, SUBSET, False)