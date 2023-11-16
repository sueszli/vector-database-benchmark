import os

def create_path_if_not_exists(datapath):
    if False:
        for i in range(10):
            print('nop')
    'Create the new file if not exists and save the data'
    if not os.path.exists(datapath):
        os.makedirs(datapath)
if __name__ == '__main__':
    model_path = 'model/model2/XGBoost/version_1'
    create_path_if_not_exists(model_path)