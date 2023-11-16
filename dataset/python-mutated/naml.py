import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers
from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.layers import AttLayer2
__all__ = ['NAMLModel']

class NAMLModel(BaseModel):
    """NAML model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie,
    Neural News Recommendation with Attentive Multi-View Learning, IJCAI 2019

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        if False:
            i = 10
            return i + 15
        "Initialization steps for NAML.\n        Compared with the BaseModel, NAML need word embedding.\n        After creating word embedding matrix, BaseModel's __init__ method will be called.\n\n        Args:\n            hparams (object): Global hyper-parameters. Some key setttings such as filter_num are there.\n            iterator_creator_train (object): NAML data loader class for train data.\n            iterator_creator_test (object): NAML data loader class for test and validation data\n        "
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams
        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        if False:
            return 10
        input_feat = [batch_data['clicked_title_batch'], batch_data['clicked_ab_batch'], batch_data['clicked_vert_batch'], batch_data['clicked_subvert_batch'], batch_data['candidate_title_batch'], batch_data['candidate_ab_batch'], batch_data['candidate_vert_batch'], batch_data['candidate_subvert_batch']]
        input_label = batch_data['labels']
        return (input_feat, input_label)

    def _get_user_feature_from_iter(self, batch_data):
        if False:
            i = 10
            return i + 15
        'get input of user encoder\n        Args:\n            batch_data: input batch data from user iterator\n\n        Returns:\n            numpy.ndarray: input user feature (clicked title batch)\n        '
        input_feature = [batch_data['clicked_title_batch'], batch_data['clicked_ab_batch'], batch_data['clicked_vert_batch'], batch_data['clicked_subvert_batch']]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
        if False:
            while True:
                i = 10
        'get input of news encoder\n        Args:\n            batch_data: input batch data from news iterator\n\n        Returns:\n            numpy.ndarray: input news feature (candidate title batch)\n        '
        input_feature = [batch_data['candidate_title_batch'], batch_data['candidate_ab_batch'], batch_data['candidate_vert_batch'], batch_data['candidate_subvert_batch']]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _build_graph(self):
        if False:
            print('Hello World!')
        'Build NAML model and scorer.\n\n        Returns:\n            object: a model used to train.\n            object: a model used to evaluate and inference.\n        '
        (model, scorer) = self._build_naml()
        return (model, scorer)

    def _build_userencoder(self, newsencoder):
        if False:
            print('Hello World!')
        'The main function to create user encoder of NAML.\n\n        Args:\n            newsencoder (object): the news encoder of NAML.\n\n        Return:\n            object: the user encoder of NAML.\n        '
        hparams = self.hparams
        his_input_title_body_verts = keras.Input(shape=(hparams.his_size, hparams.title_size + hparams.body_size + 2), dtype='int32')
        click_news_presents = layers.TimeDistributed(newsencoder)(his_input_title_body_verts)
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(click_news_presents)
        model = keras.Model(his_input_title_body_verts, user_present, name='user_encoder')
        return model

    def _build_newsencoder(self, embedding_layer):
        if False:
            for i in range(10):
                print('nop')
        'The main function to create news encoder of NAML.\n        news encoder in composed of title encoder, body encoder, vert encoder and subvert encoder\n\n        Args:\n            embedding_layer (object): a word embedding layer.\n\n        Return:\n            object: the news encoder of NAML.\n        '
        hparams = self.hparams
        input_title_body_verts = keras.Input(shape=(hparams.title_size + hparams.body_size + 2,), dtype='int32')
        sequences_input_title = layers.Lambda(lambda x: x[:, :hparams.title_size])(input_title_body_verts)
        sequences_input_body = layers.Lambda(lambda x: x[:, hparams.title_size:hparams.title_size + hparams.body_size])(input_title_body_verts)
        input_vert = layers.Lambda(lambda x: x[:, hparams.title_size + hparams.body_size:hparams.title_size + hparams.body_size + 1])(input_title_body_verts)
        input_subvert = layers.Lambda(lambda x: x[:, hparams.title_size + hparams.body_size + 1:])(input_title_body_verts)
        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        vert_repr = self._build_vertencoder()(input_vert)
        subvert_repr = self._build_subvertencoder()(input_subvert)
        concate_repr = layers.Concatenate(axis=-2)([title_repr, body_repr, vert_repr, subvert_repr])
        news_repr = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(concate_repr)
        model = keras.Model(input_title_body_verts, news_repr, name='news_encoder')
        return model

    def _build_titleencoder(self, embedding_layer):
        if False:
            print('Hello World!')
        'build title encoder of NAML news encoder.\n\n        Args:\n            embedding_layer (object): a word embedding layer.\n\n        Return:\n            object: the title encoder of NAML.\n        '
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype='int32')
        embedded_sequences_title = embedding_layer(sequences_input_title)
        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(hparams.filter_num, hparams.window_size, activation=hparams.cnn_activation, padding='same', bias_initializer=keras.initializers.Zeros(), kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams.filter_num))(pred_title)
        model = keras.Model(sequences_input_title, pred_title, name='title_encoder')
        return model

    def _build_bodyencoder(self, embedding_layer):
        if False:
            while True:
                i = 10
        'build body encoder of NAML news encoder.\n\n        Args:\n            embedding_layer (object): a word embedding layer.\n\n        Return:\n            object: the body encoder of NAML.\n        '
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams.body_size,), dtype='int32')
        embedded_sequences_body = embedding_layer(sequences_input_body)
        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        y = layers.Conv1D(hparams.filter_num, hparams.window_size, activation=hparams.cnn_activation, padding='same', bias_initializer=keras.initializers.Zeros(), kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(y)
        y = layers.Dropout(hparams.dropout)(y)
        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)
        model = keras.Model(sequences_input_body, pred_body, name='body_encoder')
        return model

    def _build_vertencoder(self):
        if False:
            return 10
        'build vert encoder of NAML news encoder.\n\n        Return:\n            object: the vert encoder of NAML.\n        '
        hparams = self.hparams
        input_vert = keras.Input(shape=(1,), dtype='int32')
        vert_embedding = layers.Embedding(hparams.vert_num, hparams.vert_emb_dim, trainable=True)
        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(hparams.filter_num, activation=hparams.dense_activation, bias_initializer=keras.initializers.Zeros(), kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(vert_emb)
        pred_vert = layers.Reshape((1, hparams.filter_num))(pred_vert)
        model = keras.Model(input_vert, pred_vert, name='vert_encoder')
        return model

    def _build_subvertencoder(self):
        if False:
            i = 10
            return i + 15
        'build subvert encoder of NAML news encoder.\n\n        Return:\n            object: the subvert encoder of NAML.\n        '
        hparams = self.hparams
        input_subvert = keras.Input(shape=(1,), dtype='int32')
        subvert_embedding = layers.Embedding(hparams.subvert_num, hparams.subvert_emb_dim, trainable=True)
        subvert_emb = subvert_embedding(input_subvert)
        pred_subvert = layers.Dense(hparams.filter_num, activation=hparams.dense_activation, bias_initializer=keras.initializers.Zeros(), kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(subvert_emb)
        pred_subvert = layers.Reshape((1, hparams.filter_num))(pred_subvert)
        model = keras.Model(input_subvert, pred_subvert, name='subvert_encoder')
        return model

    def _build_naml(self):
        if False:
            while True:
                i = 10
        "The main function to create NAML's logic. The core of NAML\n        is a user encoder and a news encoder.\n\n        Returns:\n            object: a model used to train.\n            object: a model used to evaluate and predict.\n        "
        hparams = self.hparams
        his_input_title = keras.Input(shape=(hparams.his_size, hparams.title_size), dtype='int32')
        his_input_body = keras.Input(shape=(hparams.his_size, hparams.body_size), dtype='int32')
        his_input_vert = keras.Input(shape=(hparams.his_size, 1), dtype='int32')
        his_input_subvert = keras.Input(shape=(hparams.his_size, 1), dtype='int32')
        pred_input_title = keras.Input(shape=(hparams.npratio + 1, hparams.title_size), dtype='int32')
        pred_input_body = keras.Input(shape=(hparams.npratio + 1, hparams.body_size), dtype='int32')
        pred_input_vert = keras.Input(shape=(hparams.npratio + 1, 1), dtype='int32')
        pred_input_subvert = keras.Input(shape=(hparams.npratio + 1, 1), dtype='int32')
        pred_input_title_one = keras.Input(shape=(1, hparams.title_size), dtype='int32')
        pred_input_body_one = keras.Input(shape=(1, hparams.body_size), dtype='int32')
        pred_input_vert_one = keras.Input(shape=(1, 1), dtype='int32')
        pred_input_subvert_one = keras.Input(shape=(1, 1), dtype='int32')
        his_title_body_verts = layers.Concatenate(axis=-1)([his_input_title, his_input_body, his_input_vert, his_input_subvert])
        pred_title_body_verts = layers.Concatenate(axis=-1)([pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert])
        pred_title_body_verts_one = layers.Concatenate(axis=-1)([pred_input_title_one, pred_input_body_one, pred_input_vert_one, pred_input_subvert_one])
        pred_title_body_verts_one = layers.Reshape((-1,))(pred_title_body_verts_one)
        embedding_layer = layers.Embedding(self.word2vec_embedding.shape[0], hparams.word_emb_dim, weights=[self.word2vec_embedding], trainable=True)
        self.newsencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)
        user_present = self.userencoder(his_title_body_verts)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
        news_present_one = self.newsencoder(pred_title_body_verts_one)
        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation='softmax')(preds)
        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation='sigmoid')(pred_one)
        model = keras.Model([his_input_title, his_input_body, his_input_vert, his_input_subvert, pred_input_title, pred_input_body, pred_input_vert, pred_input_subvert], preds)
        scorer = keras.Model([his_input_title, his_input_body, his_input_vert, his_input_subvert, pred_input_title_one, pred_input_body_one, pred_input_vert_one, pred_input_subvert_one], pred_one)
        return (model, scorer)