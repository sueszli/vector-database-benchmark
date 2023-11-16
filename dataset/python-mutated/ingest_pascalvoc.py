from configargparse import ArgumentParser
from convert_xml_to_json import convert_xml_to_json
import numpy as np
import os
import tarfile

def ingest_pascal(data_dir, out_dir, year='2007', overwrite=False):
    if False:
        return 10
    root_dir = os.path.join(out_dir, 'VOCdevkit', 'VOC' + year)
    manifest_train = os.path.join(root_dir, 'trainval.csv')
    manifest_inference = os.path.join(root_dir, 'val.csv')
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pascalvoc.cfg')
    with open(config_path, 'w') as f:
        f.write('manifest = [train:{}, val:{}]\n'.format(manifest_train, manifest_inference))
        f.write('manifest_root = {}\n'.format(out_dir))
        f.write('epochs = 14\n')
        f.write('height = 1000\n')
        f.write('width = 1000\n')
        f.write('batch_size = 1\n')
        f.write('rng_seed = 0')
    print('Wrote config file to: {}'.format(config_path))
    if not overwrite and os.path.exists(manifest_train) and os.path.exists(manifest_inference):
        print('Found existing manfiest files, skipping ingest,\n              Use --overwrite to rerun ingest anyway.')
        return (manifest_train, manifest_inference)
    tarfiles = [os.path.join(data_dir, tar) for tar in ['VOCtrainval_06-Nov-2007.tar', 'VOCtest_06-Nov-2007.tar']]
    for file in tarfiles:
        with tarfile.open(file, 'r') as t:
            print('Extracting {} to {}'.format(file, out_dir))
            t.extractall(out_dir)
    input_path = os.path.join(root_dir, 'Annotations')
    annot_path = os.path.join(root_dir, 'Annotations-json')
    print('Reading PASCAL XML files from {}'.format(input_path))
    print('Converting XML files to json format, writing to: {}'.format(annot_path))
    convert_xml_to_json(input_path, annot_path, difficult=False)
    annot_path_difficult = os.path.join(root_dir, 'Annotations-json-difficult')
    print('Converting XML files to json format (including objects with difficult flag),')
    print('writing to: {}'.format(annot_path_difficult))
    convert_xml_to_json(input_path, annot_path_difficult, difficult=True)
    img_dir = os.path.join(root_dir, 'JPEGImages')
    index_path = os.path.join(root_dir, 'ImageSets', 'Main', 'trainval.txt')
    create_manifest(manifest_train, index_path, annot_path, img_dir, out_dir)
    index_path = os.path.join(root_dir, 'ImageSets', 'Main', 'test.txt')
    create_manifest(manifest_inference, index_path, annot_path_difficult, img_dir, out_dir)

def create_manifest(manifest_path, index_file, annot_dir, image_dir, root_dir):
    if False:
        while True:
            i = 10
    '\n    Based on a PASCALVOC index file, creates a manifest csv file.\n    If the manifest file already exists, this function will skip writing, unless the\n    overwrite argument is set to True.\n\n    Arguments:\n        manifest_path (string): path to save the manifest file\n        index (string or list): list of images.\n        annot_dir (string): directory of annotations\n        img_dir (string): directory of images\n        root_dir (string): paths will be made relative to this directory\n        ext (string, optional): image extension (default=.jpg)\n    '
    records = [('@FILE', 'FILE')]
    with open(index_file) as f:
        for img in f:
            tag = img.rstrip(os.linesep)
            image = os.path.join(image_dir, tag + '.jpg')
            annot = os.path.join(annot_dir, tag + '.json')
            assert os.path.exists(image), 'Path {} not found'.format(image)
            assert os.path.exists(annot), 'Path {} not found'.format(annot)
            records.append((os.path.relpath(image, root_dir), os.path.relpath(annot, root_dir)))
    np.savetxt(manifest_path, records, fmt='%s\t%s')
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='path to directory with vocdevkit data')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--overwrite', action='store_true', help='overwrite files')
    args = parser.parse_args()
    ingest_pascal(args.input_dir, args.output_dir, overwrite=args.overwrite)