import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from recommenders.evaluation.python_evaluation import ndcg_at_k

class LossHistory(Callback):
    """This class is used for saving the validation loss and the training loss per epoch."""

    def on_train_begin(self, logs={}):
        if False:
            while True:
                i = 10
        'Initialise the lists where the loss of training and validation will be saved.'
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        if False:
            i = 10
            return i + 15
        'Save the loss of training and validation set at the end of each epoch.'
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

class Metrics(Callback):
    """Callback function used to calculate the NDCG@k metric of validation set at the end of each epoch.
    Weights of the model with the highest NDCG@k value is saved."""

    def __init__(self, model, val_tr, val_te, mapper, k, save_path=None):
        if False:
            while True:
                i = 10
        'Initialize the class parameters.\n\n        Args:\n            model: trained model for validation.\n            val_tr (numpy.ndarray, float): the click matrix for the validation set training part.\n            val_te (numpy.ndarray, float): the click matrix for the validation set testing part.\n            mapper (AffinityMatrix): the mapper for converting click matrix to dataframe.\n            k (int): number of top k items per user (optional).\n            save_path (str): Default path to save weights.\n        '
        self.model = model
        self.best_ndcg = 0.0
        self.val_tr = val_tr
        self.val_te = val_te
        self.mapper = mapper
        self.k = k
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        if False:
            print('Hello World!')
        'Initialise the list for validation NDCG@k.'
        self._data = []

    def recommend_k_items(self, x, k, remove_seen=True):
        if False:
            while True:
                i = 10
        'Returns the top-k items ordered by a relevancy score.\n        Obtained probabilities are used as recommendation score.\n\n        Args:\n            x (numpy.ndarray, int32): input click matrix.\n            k (scalar, int32): the number of items to recommend.\n\n        Returns:\n            numpy.ndarray: A sparse matrix containing the top_k elements ordered by their score.\n\n        '
        score = self.model.predict(x)
        if remove_seen:
            seen_mask = np.not_equal(x, 0)
            score[seen_mask] = 0
        top_items = np.argpartition(-score, range(k), axis=1)[:, :k]
        score_c = score.copy()
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0
        top_scores = score - score_c
        return top_scores

    def on_epoch_end(self, batch, logs={}):
        if False:
            i = 10
            return i + 15
        'At the end of each epoch calculate NDCG@k of the validation set.\n        If the model performance is improved, the model weights are saved.\n        Update the list of validation NDCG@k by adding obtained value.\n        '
        top_k = self.recommend_k_items(x=self.val_tr, k=self.k, remove_seen=True)
        top_k_df = self.mapper.map_back_sparse(top_k, kind='prediction')
        test_df = self.mapper.map_back_sparse(self.val_te, kind='ratings')
        NDCG = ndcg_at_k(test_df, top_k_df, col_prediction='prediction', k=self.k)
        if NDCG > self.best_ndcg:
            self.best_ndcg = NDCG
            if self.save_path is not None:
                self.model.save(self.save_path)
        self._data.append(NDCG)

    def get_data(self):
        if False:
            i = 10
            return i + 15
        'Returns a list of the NDCG@k of the validation set metrics calculated\n        at the end of each epoch.'
        return self._data

class AnnealingCallback(Callback):
    """This class is used for updating the value of β during the annealing process.
    When β reaches the value of anneal_cap, it stops increasing.
    """

    def __init__(self, beta, anneal_cap, total_anneal_steps):
        if False:
            while True:
                i = 10
        'Constructor\n\n        Args:\n            beta (float): current value of beta.\n            anneal_cap (float): maximum value that beta can reach.\n            total_anneal_steps (int): total number of annealing steps.\n        '
        self.anneal_cap = anneal_cap
        self.beta = beta
        self.update_count = 0
        self.total_anneal_steps = total_anneal_steps

    def on_train_begin(self, logs={}):
        if False:
            return 10
        'Initialise a list in which the beta value will be saved at the end of each epoch.'
        self._beta = []

    def on_batch_end(self, epoch, logs={}):
        if False:
            while True:
                i = 10
        'At the end of each batch the beta should is updated until it reaches the values of anneal cap.'
        self.update_count = self.update_count + 1
        new_beta = min(1.0 * self.update_count / self.total_anneal_steps, self.anneal_cap)
        K.set_value(self.beta, new_beta)

    def on_epoch_end(self, epoch, logs={}):
        if False:
            while True:
                i = 10
        'At the end of each epoch save the value of beta in _beta list.'
        tmp = K.eval(self.beta)
        self._beta.append(tmp)

    def get_data(self):
        if False:
            while True:
                i = 10
        'Returns a list of the beta values per epoch.'
        return self._beta

class StandardVAE:
    """Standard Variational Autoencoders (VAE) for Collaborative Filtering implementation."""

    def __init__(self, n_users, original_dim, intermediate_dim=200, latent_dim=70, n_epochs=400, batch_size=100, k=100, verbose=1, drop_encoder=0.5, drop_decoder=0.5, beta=1.0, annealing=False, anneal_cap=1.0, seed=None, save_path=None):
        if False:
            print('Hello World!')
        'Initialize class parameters.\n\n        Args:\n            n_users (int): Number of unique users in the train set.\n            original_dim (int): Number of unique items in the train set.\n            intermediate_dim (int): Dimension of intermediate space.\n            latent_dim (int): Dimension of latent space.\n            n_epochs (int): Number of epochs for training.\n            batch_size (int): Batch size.\n            k (int): number of top k items per user.\n            verbose (int): Whether to show the training output or not.\n            drop_encoder (float): Dropout percentage of the encoder.\n            drop_decoder (float): Dropout percentage of the decoder.\n            beta (float): a constant parameter β in the ELBO function,\n                  when you are not using annealing (annealing=False)\n            annealing (bool): option of using annealing method for training the model (True)\n                  or not using annealing, keeping a constant beta (False)\n            anneal_cap (float): maximum value that beta can take during annealing process.\n            seed (int): Seed.\n            save_path (str): Default path to save weights.\n        '
        self.seed = seed
        np.random.seed(self.seed)
        self.n_users = n_users
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose
        self.number_of_batches = self.n_users // self.batch_size
        self.anneal_cap = anneal_cap
        self.annealing = annealing
        if self.annealing:
            self.beta = K.variable(0.0)
        else:
            self.beta = beta
        self.total_anneal_steps = self.number_of_batches * (self.n_epochs - int(self.n_epochs * 0.2)) // self.anneal_cap
        self.drop_encoder = drop_encoder
        self.drop_decoder = drop_decoder
        self.save_path = save_path
        self._create_model()

    def _create_model(self):
        if False:
            return 10
        'Build and compile model.'
        self.x = Input(shape=(self.original_dim,))
        self.dropout_encoder = Dropout(self.drop_encoder)(self.x)
        self.h = Dense(self.intermediate_dim, activation='tanh')(self.dropout_encoder)
        self.z_mean = Dense(self.latent_dim)(self.h)
        self.z_log_var = Dense(self.latent_dim)(self.h)
        self.z = Lambda(self._take_sample, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])
        self.h_decoder = Dense(self.intermediate_dim, activation='tanh')
        self.dropout_decoder = Dropout(self.drop_decoder)
        self.x_bar = Dense(self.original_dim, activation='softmax')
        self.h_decoded = self.h_decoder(self.z)
        self.h_decoded_ = self.dropout_decoder(self.h_decoded)
        self.x_decoded = self.x_bar(self.h_decoded_)
        self.model = Model(self.x, self.x_decoded)
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss=self._get_vae_loss)

    def _get_vae_loss(self, x, x_bar):
        if False:
            print('Hello World!')
        'Calculate negative ELBO (NELBO).'
        reconst_loss = self.original_dim * binary_crossentropy(x, x_bar)
        kl_loss = 0.5 * K.sum(-1 - self.z_log_var + K.square(self.z_mean) + K.exp(self.z_log_var), axis=-1)
        return reconst_loss + self.beta * kl_loss

    def _take_sample(self, args):
        if False:
            print('Hello World!')
        'Sample epsilon ∼ N (0,I) and compute z via reparametrization trick.'
        'Calculate latent vector using the reparametrization trick.\n           The idea is that sampling from N (_mean, _var) is s the same as sampling from _mean+ epsilon * _var\n           where epsilon ∼ N(0,I).'
        (_mean, _log_var) = args
        epsilon = K.random_normal(shape=(K.shape(_mean)[0], self.latent_dim), mean=0.0, stddev=1.0, seed=self.seed)
        return _mean + K.exp(_log_var / 2) * epsilon

    def nn_batch_generator(self, x_train):
        if False:
            return 10
        'Used for splitting dataset in batches.\n\n        Args:\n            x_train (numpy.ndarray): The click matrix for the train set with float values.\n        '
        np.random.seed(self.seed)
        shuffle_index = np.arange(np.shape(x_train)[0])
        np.random.shuffle(shuffle_index)
        x = x_train[shuffle_index, :]
        y = x_train[shuffle_index, :]
        counter = 0
        while 1:
            index_batch = shuffle_index[self.batch_size * counter:self.batch_size * (counter + 1)]
            x_batch = x[index_batch, :]
            y_batch = y[index_batch, :]
            counter += 1
            yield (np.array(x_batch), np.array(y_batch))
            if counter >= self.number_of_batches:
                counter = 0

    def fit(self, x_train, x_valid, x_val_tr, x_val_te, mapper):
        if False:
            return 10
        'Fit model with the train sets and validate on the validation set.\n\n        Args:\n            x_train (numpy.ndarray): The click matrix for the train set.\n            x_valid (numpy.ndarray): The click matrix for the validation set.\n            x_val_tr (numpy.ndarray): The click matrix for the validation set training part.\n            x_val_te (numpy.ndarray): The click matrix for the validation set testing part.\n            mapper (object): The mapper for converting click matrix to dataframe. It can be AffinityMatrix.\n        '
        history = LossHistory()
        metrics = Metrics(model=self.model, val_tr=x_val_tr, val_te=x_val_te, mapper=mapper, k=self.k, save_path=self.save_path)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
        if self.annealing:
            anneal = AnnealingCallback(self.beta, self.anneal_cap, self.total_anneal_steps)
            self.model.fit_generator(generator=self.nn_batch_generator(x_train), steps_per_epoch=self.number_of_batches, epochs=self.n_epochs, verbose=self.verbose, callbacks=[metrics, history, self.reduce_lr, anneal], validation_data=(x_valid, x_valid))
            self.ls_beta = anneal.get_data()
        else:
            self.model.fit_generator(generator=self.nn_batch_generator(x_train), steps_per_epoch=self.number_of_batches, epochs=self.n_epochs, verbose=self.verbose, callbacks=[metrics, history, self.reduce_lr], validation_data=(x_valid, x_valid))
        self.train_loss = history.losses
        self.val_loss = history.val_losses
        self.val_ndcg = metrics.get_data()

    def get_optimal_beta(self):
        if False:
            while True:
                i = 10
        'Returns the value of the optimal beta.'
        index_max_ndcg = np.argmax(self.val_ndcg)
        optimal_beta = self.ls_beta[index_max_ndcg]
        return optimal_beta

    def display_metrics(self):
        if False:
            print('Hello World!')
        'Plots:\n        1) Loss per epoch both for validation and train sets\n        2) NDCG@k per epoch of the validation set\n        '
        plt.figure(figsize=(14, 5))
        sns.set(style='whitegrid')
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, color='b', linestyle='-', label='Train')
        plt.plot(self.val_loss, color='r', linestyle='-', label='Val')
        plt.title('\n')
        plt.xlabel('Epochs', size=14)
        plt.ylabel('Loss', size=14)
        plt.legend(loc='upper left')
        plt.subplot(1, 2, 2)
        plt.plot(self.val_ndcg, color='r', linestyle='-', label='Val')
        plt.title('\n')
        plt.xlabel('Epochs', size=14)
        plt.ylabel('NDCG@k', size=14)
        plt.legend(loc='upper left')
        plt.suptitle('TRAINING AND VALIDATION METRICS HISTORY', size=16)
        plt.tight_layout(pad=2)

    def recommend_k_items(self, x, k, remove_seen=True):
        if False:
            i = 10
            return i + 15
        'Returns the top-k items ordered by a relevancy score.\n\n        Obtained probabilities are used as recommendation score.\n\n        Args:\n            x (numpy.ndarray): Input click matrix, with `int32` values.\n            k (scalar): The number of items to recommend.\n\n        Returns:\n            numpy.ndarray: A sparse matrix containing the top_k elements ordered by their score.\n\n        '
        self.model.load_weights(self.save_path)
        score = self.model.predict(x)
        if remove_seen:
            seen_mask = np.not_equal(x, 0)
            score[seen_mask] = 0
        top_items = np.argpartition(-score, range(k), axis=1)[:, :k]
        score_c = score.copy()
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0
        top_scores = score - score_c
        return top_scores

    def ndcg_per_epoch(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the list of NDCG@k at each epoch.'
        return self.val_ndcg