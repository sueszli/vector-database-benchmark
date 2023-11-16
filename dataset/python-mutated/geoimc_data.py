import logging
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, isspmatrix_csr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from recommenders.utils.python_utils import binarize
from .geoimc_utils import length_normalize, reduce_dims
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('geoimc')

class DataPtr:
    """
    Holds data and its respective indices
    """

    def __init__(self, data, entities):
        if False:
            print('Hello World!')
        'Initialize a data pointer\n\n        Args:\n            data (csr_matrix): The target data matrix.\n            entities (Iterator): An iterator (of 2 elements (ndarray)) containing\n            the features of row, col entities.\n        '
        assert isspmatrix_csr(data)
        self.data = data
        self.entities = entities
        self.data_indices = None
        self.entity_indices = [None, None]

    def get_data(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            csr_matrix: Target matrix (based on the data_indices filter)\n        '
        if self.data_indices is None:
            return self.data
        return self.data[self.data_indices]

    def get_entity(self, of='row'):
        if False:
            for i in range(10):
                print('nop')
        "Get entity\n\n        Args:\n            of (str): The entity, either 'row' or 'col'\n        Returns:\n            numpy.ndarray: Entity matrix (based on the entity_indices filter)\n        "
        idx = 0 if of == 'row' else 1
        if self.entity_indices[idx] is None:
            return self.entities[idx]
        return self.entities[idx][self.entity_indices[idx]]

class Dataset:
    """
    Base class that holds necessary (minimal) information needed
    """

    def __init__(self, name, features_dim=0, normalize=False, target_transform=''):
        if False:
            for i in range(10):
                print('nop')
        "Initialize parameters\n\n        Args:\n            name (str): Name of the dataset\n            features_dim (uint): Dimension of the features. If not 0, PCA is performed\n                on the features as the dimensionality reduction technique\n            normalize (bool): Normalize the features\n            target_transform (str): Transform the target values. Current options are\n                'normalize' (Normalize the values), '' (Do nothing), 'binarize' (convert\n                the values using a threshold defined per dataset)\n\n        "
        self.name = None
        self.training_data = None
        self.test_data = None
        self.entities = None
        self.features_dim = features_dim
        self.feat_normalize = normalize
        self.target_transform = target_transform

    def normalize(self):
        if False:
            print('Hello World!')
        'Normalizes the entity features'
        if self.feat_normalize:
            for i in range(len(self.entities)):
                if isspmatrix_csr(self.entities[i]):
                    logger.info('Normalizing CSR matrix')
                    self.entities[i] = normalize(self.entities[i])
                else:
                    self.entities[i] = length_normalize(self.entities[i])

    def generate_train_test_data(self, data, test_ratio=0.3):
        if False:
            return 10
        'Generate train, test split. The split is performed on the row\n        entities. So, this essentially becomes a cold start row entity test.\n\n        Args:\n            data (csr_matrix): The entire target matrix.\n            test_ratio (float): Ratio of test split.\n\n        '
        self.training_data = DataPtr(data, self.entities)
        self.test_data = DataPtr(data, self.entities)
        (self.training_data.data_indices, self.test_data.data_indices) = train_test_split(np.array(range(0, data.shape[0])), test_size=test_ratio, shuffle=True, random_state=0)
        self.training_data.entity_indices[0] = self.training_data.data_indices
        self.test_data.entity_indices[0] = self.test_data.data_indices

    def reduce_dims(self):
        if False:
            for i in range(10):
                print('nop')
        'Reduces the dimensionality of entity features.'
        if self.features_dim != 0:
            self.entities[0] = reduce_dims(self.entities[0], self.features_dim)
            self.entities[1] = reduce_dims(self.entities[1], self.features_dim)
            logger.info('Dimensionality reduced ...')

class ML_100K(Dataset):
    """
    Handles MovieLens-100K
    """

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(self.__class__.__name__, **kwargs)
        self.min_rating = 1
        self.max_rating = 5

    def df2coo(self, df):
        if False:
            for i in range(10):
                print('nop')
        'Convert the input dataframe into a coo matrix\n\n        Args:\n            df (pandas.DataFrame): DataFrame containing the target matrix information.\n        '
        data = []
        row = list(df['user id'] - 1)
        col = list(df['item id'] - 1)
        for idx in range(0, len(df)):
            val = df['rating'].iloc[idx]
            data += [val]
        if self.target_transform == 'normalize':
            data = data / np.sqrt(np.sum(np.arange(self.min_rating, self.max_rating + 1) ** 2))
        elif self.target_transform == 'binarize':
            data = binarize(np.array(data), 3)
        return coo_matrix((data, (row, col)), shape=(943, 1682))

    def _read_from_file(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Read the traget matrix from file at path.\n\n        Args:\n            path (str): Path to the target matrix\n        '
        df = pd.read_csv(path, delimiter='\t', names=['user id', 'item id', 'rating', 'timestamp'], encoding='ISO-8859-1')
        df.drop(['timestamp'], axis=1, inplace=True)
        return self.df2coo(df)

    def load_data(self, path):
        if False:
            print('Hello World!')
        'Load dataset\n\n        Args:\n            path (str): Path to the directory containing ML100K dataset\n            e1_path (str): Path to the file containing row (user) features of ML100K dataset\n            e2_path (str): Path to the file containing col (movie) features of ML100K dataset\n        '
        self.entities = [self._load_user_features(f'{path}/u.user'), self._load_item_features(f'{path}/u.item')]
        self.normalize()
        self.reduce_dims()
        self.training_data = DataPtr(self._read_from_file(f'{path}/u1.base').tocsr(), self.entities)
        self.test_data = DataPtr(self._read_from_file(f'{path}/u1.test').tocsr(), self.entities)

    def _load_user_features(self, path):
        if False:
            i = 10
            return i + 15
        'Load user features\n\n        Args:\n            path (str): Path to the file containing user features information\n\n        '
        data = pd.read_csv(path, delimiter='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
        features_df = pd.concat([data['user_id'], pd.get_dummies(data['user_id']), pd.get_dummies(data['age']), pd.get_dummies(data['gender']), pd.get_dummies(data['occupation']), pd.get_dummies(data['zip_code'])], axis=1)
        features_df.drop(['user_id'], axis=1, inplace=True)
        user_features = np.nan_to_num(features_df.to_numpy())
        return user_features

    def _load_item_features(self, path):
        if False:
            print('Hello World!')
        'Load item features\n\n        Args:\n            path (str): Path to the file containing item features information\n\n        '
        header = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        data = pd.read_csv(path, delimiter='|', names=header, encoding='ISO-8859-1')
        features_df = pd.concat([pd.get_dummies(data['movie_title']), pd.get_dummies(data['release_date']), pd.get_dummies('video_release_date'), pd.get_dummies('IMDb_URL'), data[header[5:]]], axis=1)
        item_features = np.nan_to_num(features_df.to_numpy())
        return item_features