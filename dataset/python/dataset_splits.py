import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def dataset_splits(ratings, validation_percentage: float, testing_percentage: float, seed=69):

    if testing_percentage != 0.0:
        (user_ids_training_validation, user_ids_test,
         item_ids_training_validation, item_ids_test,
         ratings_training_validation, ratings_test) = train_test_split(ratings['row'],
                                                                       ratings['col'],
                                                                       ratings['data'],
                                                                       test_size=testing_percentage,
                                                                       shuffle=True,
                                                                       random_state=seed)

        urm_test = sp.csr_matrix((ratings_test, (user_ids_test, item_ids_test)))
        urm_training_validation = sp.csr_matrix((ratings_training_validation, (user_ids_training_validation,
                                                                               item_ids_training_validation)))
    else:
        user_ids_training_validation, item_ids_training_validation, ratings_training_validation = ratings['row'], \
                                                                                                  ratings['col'], \
                                                                                                  ratings['data']
        urm_test = np.zeros(1)
        urm_training_validation = np.zeros(1)

    if validation_percentage != 0.0:
        (user_ids_training, user_ids_validation,
         item_ids_training, item_ids_validation,
         ratings_training, ratings_validation) = train_test_split(user_ids_training_validation,
                                                                  item_ids_training_validation,
                                                                  ratings_training_validation,
                                                                  test_size=validation_percentage,
                                                                  random_state=seed)

        urm_validation = sp.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)))
    else:
        user_ids_training, item_ids_training, ratings_training = user_ids_training_validation, \
                                                                 item_ids_training_validation, \
                                                                 ratings_training_validation
        urm_validation = np.zeros(1)


    urm_train = sp.csr_matrix((ratings_training, (user_ids_training, item_ids_training)))

    return urm_training_validation, urm_train, urm_validation, urm_test
