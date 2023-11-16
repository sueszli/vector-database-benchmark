import tensorflow.keras as keras
from tensorflow.keras import layers
from recommenders.models.newsrec.models.base_model import BaseModel
from recommenders.models.newsrec.models.layers import AttLayer2, SelfAttention
__all__ = ['NRMSModel']

class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        if False:
            i = 10
            return i + 15
        "Initialization steps for NRMS.\n        Compared with the BaseModel, NRMS need word embedding.\n        After creating word embedding matrix, BaseModel's __init__ method will be called.\n\n        Args:\n            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.\n            iterator_creator_train (object): NRMS data loader class for train data.\n            iterator_creator_test (object): NRMS data loader class for test and validation data\n        "
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        if False:
            while True:
                i = 10
        'get input and labels for trainning from iterator\n\n        Args:\n            batch data: input batch data from iterator\n\n        Returns:\n            list: input feature fed into model (clicked_title_batch & candidate_title_batch)\n            numpy.ndarray: labels\n        '
        input_feat = [batch_data['clicked_title_batch'], batch_data['candidate_title_batch']]
        input_label = batch_data['labels']
        return (input_feat, input_label)

    def _get_user_feature_from_iter(self, batch_data):
        if False:
            while True:
                i = 10
        'get input of user encoder\n        Args:\n            batch_data: input batch data from user iterator\n\n        Returns:\n            numpy.ndarray: input user feature (clicked title batch)\n        '
        return batch_data['clicked_title_batch']

    def _get_news_feature_from_iter(self, batch_data):
        if False:
            for i in range(10):
                print('nop')
        'get input of news encoder\n        Args:\n            batch_data: input batch data from news iterator\n\n        Returns:\n            numpy.ndarray: input news feature (candidate title batch)\n        '
        return batch_data['candidate_title_batch']

    def _build_graph(self):
        if False:
            while True:
                i = 10
        'Build NRMS model and scorer.\n\n        Returns:\n            object: a model used to train.\n            object: a model used to evaluate and inference.\n        '
        (model, scorer) = self._build_nrms()
        return (model, scorer)

    def _build_userencoder(self, titleencoder):
        if False:
            print('Hello World!')
        'The main function to create user encoder of NRMS.\n\n        Args:\n            titleencoder (object): the news encoder of NRMS.\n\n        Return:\n            object: the user encoder of NRMS.\n        '
        hparams = self.hparams
        his_input_title = keras.Input(shape=(hparams.his_size, hparams.title_size), dtype='int32')
        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([click_title_presents] * 3)
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        model = keras.Model(his_input_title, user_present, name='user_encoder')
        return model

    def _build_newsencoder(self, embedding_layer):
        if False:
            for i in range(10):
                print('nop')
        'The main function to create news encoder of NRMS.\n\n        Args:\n            embedding_layer (object): a word embedding layer.\n\n        Return:\n            object: the news encoder of NRMS.\n        '
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype='int32')
        embedded_sequences_title = embedding_layer(sequences_input_title)
        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        model = keras.Model(sequences_input_title, pred_title, name='news_encoder')
        return model

    def _build_nrms(self):
        if False:
            for i in range(10):
                print('nop')
        "The main function to create NRMS's logic. The core of NRMS\n        is a user encoder and a news encoder.\n\n        Returns:\n            object: a model used to train.\n            object: a model used to evaluate and inference.\n        "
        hparams = self.hparams
        his_input_title = keras.Input(shape=(hparams.his_size, hparams.title_size), dtype='int32')
        pred_input_title = keras.Input(shape=(hparams.npratio + 1, hparams.title_size), dtype='int32')
        pred_input_title_one = keras.Input(shape=(1, hparams.title_size), dtype='int32')
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(pred_input_title_one)
        embedding_layer = layers.Embedding(self.word2vec_embedding.shape[0], hparams.word_emb_dim, weights=[self.word2vec_embedding], trainable=True)
        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder
        user_present = self.userencoder(his_input_title)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)
        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation='softmax')(preds)
        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation='sigmoid')(pred_one)
        model = keras.Model([his_input_title, pred_input_title], preds)
        scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)
        return (model, scorer)