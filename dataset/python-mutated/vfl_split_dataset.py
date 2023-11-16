from csv import reader, writer
import numpy as np
import click
'\nThis script could split a dataset of csv to 2 parts. The odd columns, e.g. 1th, 3th, 5th, ...\nwould be split to one file and the even, e.g. 2th, 4th, 6th, ... would be split to another\nThis script is for Vertical Federated Learning functionality test, we could split the data\ninto two parts and mock the 2 client (parties) holding data of different features.\n'

@click.command()
@click.argument('file_name', type=str)
@click.argument('num_pieces', type=int)
@click.argument('has_rowkey_index', type=bool)
def vfl_split_dataset(file_name, num_pieces, has_rowkey_index):
    if False:
        for i in range(10):
            print('nop')
    with open(file_name, 'r') as read_obj:
        f = open(file_name, 'r')
        sample = f.readline().split(',')
        f.close()
        print(f'data has {len(sample)} columns')
        csv_reader = reader(read_obj)
        writer_list = []
        col_idx_list = []
        for i in range(num_pieces):
            piece_file_name = file_name.split('/')[-1].split('.')[0]
            writer_list.append(writer(open(f'{piece_file_name}-{i}.csv', 'w'), delimiter=','))
            if has_rowkey_index:
                col_idx_list.append([0] + [i for i in range(1 + i, len(sample), num_pieces)])
            else:
                col_idx_list.append([i for i in range(i, len(sample), num_pieces)])
        for (i, row) in enumerate(csv_reader):
            row = np.array(row)
            for j in range(num_pieces):
                writer_list[j].writerow(row[np.array(col_idx_list[j])])
if __name__ == '__main__':
    vfl_split_dataset()