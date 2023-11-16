from datetime import datetime, timezone
import argparse
import logging
import csv
import os
import json
'\nPurpose\nAmazon Rekognition Custom Labels model example used in the service documentation.\nShows how to create an image-level (classification) manifest file from a CSV file.\nYou can specify multiple image level labels per image.\nCSV file format is\nimage,label,label,..\nIf necessary, use the bucket argument to specify the S3 bucket folder for the images.\nhttps://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-gt-cl-transform.html\n'
logger = logging.getLogger(__name__)

def check_duplicates(csv_file, deduplicated_file, duplicates_file):
    if False:
        print('Hello World!')
    '\n    Checks for duplicate images in a CSV file. If duplicate images\n    are found, deduplicated_file is the deduplicated CSV file - only the first\n    occurence of a duplicate is recorded. Other duplicates are recorded in duplicates_file.\n    :param csv_file: The source CSV file.\n    :param deduplicated_file: The deduplicated CSV file to create. If no duplicates are found\n    this file is removed.\n    :param duplicates_file: The duplicate images CSV file to create. If no duplicates are found\n    this file is removed.\n    :return: True if duplicates are found, otherwise false.\n    '
    logger.info('Deduplicating %s', csv_file)
    duplicates_found = False
    with open(csv_file, 'r', newline='', encoding='UTF-8') as f, open(deduplicated_file, 'w', encoding='UTF-8') as dedup, open(duplicates_file, 'w', encoding='UTF-8') as duplicates:
        reader = csv.reader(f, delimiter=',')
        dedup_writer = csv.writer(dedup)
        duplicates_writer = csv.writer(duplicates)
        entries = set()
        for row in reader:
            if not ''.join(row).strip():
                continue
            key = row[0]
            if key not in entries:
                dedup_writer.writerow(row)
                entries.add(key)
            else:
                duplicates_writer.writerow(row)
                duplicates_found = True
    if duplicates_found:
        logger.info('Duplicates found check %s', duplicates_file)
    else:
        os.remove(duplicates_file)
        os.remove(deduplicated_file)
    return duplicates_found

def create_manifest_file(csv_file, manifest_file, s3_path):
    if False:
        while True:
            i = 10
    '\n    Reads a CSV file and creates a Custom Labels classification manifest file.\n    :param csv_file: The source CSV file.\n    :param manifest_file: The name of the manifest file to create.\n    :param s3_path: The S3 path to the folder that contains the images.\n    '
    logger.info('Processing CSV file %s', csv_file)
    image_count = 0
    label_count = 0
    with open(csv_file, newline='', encoding='UTF-8') as csvfile, open(manifest_file, 'w', encoding='UTF-8') as output_file:
        image_classifications = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in image_classifications:
            source_ref = str(s3_path) + row[0]
            image_count += 1
            json_line = {}
            json_line['source-ref'] = source_ref
            for index in range(1, len(row)):
                image_level_label = row[index]
                if image_level_label == '':
                    continue
                label_count += 1
                json_line[image_level_label] = 1
                metadata = {}
                metadata['confidence'] = 1
                metadata['job-name'] = 'labeling-job/' + image_level_label
                metadata['class-name'] = image_level_label
                metadata['human-annotated'] = 'yes'
                metadata['creation-date'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')
                metadata['type'] = 'groundtruth/image-classification'
                json_line[f'{image_level_label}-metadata'] = metadata
            output_file.write(json.dumps(json_line))
            output_file.write('\n')
    output_file.close()
    logger.info('Finished creating manifest file %s\nImages: %s\nLabels: %s', manifest_file, image_count, label_count)
    return (image_count, label_count)

def add_arguments(parser):
    if False:
        for i in range(10):
            print('nop')
    '\n    Adds command line arguments to the parser.\n    :param parser: The command line parser.\n    '
    parser.add_argument('csv_file', help='The CSV file that you want to process.')
    parser.add_argument('--s3_path', help='The S3 bucket and folder path for the images. If not supplied, column 1 is assumed to include the S3 path.', required=False)

def main():
    if False:
        print('Hello World!')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        parser = argparse.ArgumentParser(usage=argparse.SUPPRESS)
        add_arguments(parser)
        args = parser.parse_args()
        s3_path = args.s3_path
        if s3_path is None:
            s3_path = ''
        csv_file = args.csv_file
        file_name = os.path.splitext(csv_file)[0]
        manifest_file = f'{file_name}.manifest'
        duplicates_file = f'{file_name}-duplicates.csv'
        deduplicated_file = f'{file_name}-deduplicated.csv'
        if check_duplicates(csv_file, deduplicated_file, duplicates_file):
            print(f'Duplicates found. Use {duplicates_file} to view duplicates and then update {deduplicated_file}. ')
            print(f'{deduplicated_file} contains the first occurence of a duplicate. Update as necessary with the correct label information.')
            print(f'Re-run the script with {deduplicated_file}')
        else:
            print('No duplicates found. Creating manifest file.')
            (image_count, label_count) = create_manifest_file(csv_file, manifest_file, s3_path)
            print(f'Finished creating manifest file: {manifest_file} \nImages: {image_count}\nLabels: {label_count}')
    except FileNotFoundError as err:
        logger.exception('File not found: %s', err)
        print(f'File not found: {err}. Check your input CSV file.')
if __name__ == '__main__':
    main()