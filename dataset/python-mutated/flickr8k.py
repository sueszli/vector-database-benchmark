import os
import re
from collections import defaultdict
from typing import List
from ludwig.datasets.loaders.dataset_loader import DatasetLoader

class Flickr8kLoader(DatasetLoader):

    def transform_files(self, file_paths: List[str]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        image_to_caption = defaultdict(list)
        with open(f'{self.raw_dataset_dir}/Flickr8k.token.txt') as captions_file:
            image_to_caption = defaultdict(list)
            for line in captions_file:
                line = line.split('#')
                line[1] = line[1].strip('\n01234.\t ')
                line[1] = re.sub('"', '""', line[1])
                line[1] = '"' + line[1] + '"'
                image_to_caption[line[0]].append(line[1])
        with open(os.path.join(self.raw_dataset_dir, 'flickr8k_dataset.csv'), 'w') as output_file:
            output_file.write('image_path,caption0,caption1,caption2,')
            output_file.write('caption3,caption4,split\n')
            splits = ['train', 'dev', 'test']
            for i in range(len(splits)):
                split = splits[i]
                with open(f'{self.raw_dataset_dir}/Flickr_8k.{split}Images.txt') as split_file:
                    for image_name in split_file:
                        image_name = image_name.strip('\n')
                        if image_name in image_to_caption:
                            output_file.write('{},{},{},{},{},{},{}\n'.format(f'{self.raw_dataset_dir}/Flicker8k_Dataset/{image_name}', *image_to_caption[image_name], i))
        return super().transform_files(file_paths)