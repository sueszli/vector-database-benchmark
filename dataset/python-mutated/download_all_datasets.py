from ludwig import datasets

def download_all_datasets():
    if False:
        i = 10
        return i + 15
    'Downloads all datasets to ./downloaded_datasets.'
    dataset_names = datasets.list_datasets()
    print('Datasets: ')
    for name in dataset_names:
        print(f'  {name}')
    print('Downloading all datasets')
    for dataset_name in dataset_names:
        print(f'Downloading {dataset_name}')
        datasets.download_dataset(dataset_name, './downloaded_datasets')
if __name__ == '__main__':
    download_all_datasets()