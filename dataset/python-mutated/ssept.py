import tensorflow as tf
from recommenders.models.sasrec.model import SASREC, Encoder, LayerNormalization

class SSEPT(SASREC):
    """
    SSE-PT Model

    :Citation:

    Wu L., Li S., Hsieh C-J., Sharpnack J., SSE-PT: Sequential Recommendation
    Via Personalized Transformer, RecSys, 2020.
    TF 1.x codebase: https://github.com/SSE-PT/SSE-PT
    TF 2.x codebase (SASREc): https://github.com/nnkkmto/SASRec-tf2
    """

    def __init__(self, **kwargs):
        if False:
            return 10
        'Model initialization.\n\n        Args:\n            item_num (int): Number of items in the dataset.\n            seq_max_len (int): Maximum number of items in user history.\n            num_blocks (int): Number of Transformer blocks to be used.\n            embedding_dim (int): Item embedding dimension.\n            attention_dim (int): Transformer attention dimension.\n            conv_dims (list): List of the dimensions of the Feedforward layer.\n            dropout_rate (float): Dropout rate.\n            l2_reg (float): Coefficient of the L2 regularization.\n            num_neg_test (int): Number of negative examples used in testing.\n            user_num (int): Number of users in the dataset.\n            user_embedding_dim (int): User embedding dimension.\n            item_embedding_dim (int): Item embedding dimension.\n        '
        super().__init__(**kwargs)
        self.user_num = kwargs.get('user_num', None)
        self.conv_dims = kwargs.get('conv_dims', [200, 200])
        self.user_embedding_dim = kwargs.get('user_embedding_dim', self.embedding_dim)
        self.item_embedding_dim = kwargs.get('item_embedding_dim', self.embedding_dim)
        self.hidden_units = self.item_embedding_dim + self.user_embedding_dim
        self.user_embedding_layer = tf.keras.layers.Embedding(input_dim=self.user_num + 1, output_dim=self.user_embedding_dim, name='user_embeddings', mask_zero=True, input_length=1, embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg))
        self.positional_embedding_layer = tf.keras.layers.Embedding(self.seq_max_len, self.user_embedding_dim + self.item_embedding_dim, name='positional_embeddings', mask_zero=False, embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg))
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(self.num_blocks, self.seq_max_len, self.hidden_units, self.hidden_units, self.attention_num_heads, self.conv_dims, self.dropout_rate)
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(self.seq_max_len, self.hidden_units, 1e-08)

    def call(self, x, training):
        if False:
            while True:
                i = 10
        'Model forward pass.\n\n        Args:\n            x (tf.Tensor): Input tensor.\n            training (tf.Tensor): Training tensor.\n\n        Returns:\n            tf.Tensor, tf.Tensor, tf.Tensor:\n            - Logits of the positive examples.\n            - Logits of the negative examples.\n            - Mask for nonzero targets\n        '
        users = x['users']
        input_seq = x['input_seq']
        pos = x['positive']
        neg = x['negative']
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        (seq_embeddings, positional_embeddings) = self.embedding(input_seq)
        u_latent = self.user_embedding_layer(users)
        u_latent = u_latent * self.user_embedding_dim ** 0.5
        u_latent = tf.tile(u_latent, [1, tf.shape(input_seq)[1], 1])
        seq_embeddings = tf.reshape(tf.concat([seq_embeddings, u_latent], 2), [tf.shape(input_seq)[0], -1, self.hidden_units])
        seq_embeddings += positional_embeddings
        seq_embeddings = self.dropout_layer(seq_embeddings, training=training)
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)
        pos = self.mask_layer(pos)
        neg = self.mask_layer(neg)
        user_emb = tf.reshape(u_latent, [tf.shape(input_seq)[0] * self.seq_max_len, self.user_embedding_dim])
        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)
        pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units])
        neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])
        seq_emb = tf.reshape(seq_attention, [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units])
        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
        pos_logits = tf.expand_dims(pos_logits, axis=-1)
        neg_logits = tf.expand_dims(neg_logits, axis=-1)
        istarget = tf.reshape(tf.cast(tf.not_equal(pos, 0), dtype=tf.float32), [tf.shape(input_seq)[0] * self.seq_max_len])
        return (pos_logits, neg_logits, istarget)

    def predict(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Model prediction for candidate (negative) items\n\n        '
        training = False
        user = inputs['user']
        input_seq = inputs['input_seq']
        candidate = inputs['candidate']
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        (seq_embeddings, positional_embeddings) = self.embedding(input_seq)
        u0_latent = self.user_embedding_layer(user)
        u0_latent = u0_latent * self.user_embedding_dim ** 0.5
        u0_latent = tf.squeeze(u0_latent, axis=0)
        test_user_emb = tf.tile(u0_latent, [1 + self.num_neg_test, 1])
        u_latent = self.user_embedding_layer(user)
        u_latent = u_latent * self.user_embedding_dim ** 0.5
        u_latent = tf.tile(u_latent, [1, tf.shape(input_seq)[1], 1])
        seq_embeddings = tf.reshape(tf.concat([seq_embeddings, u_latent], 2), [tf.shape(input_seq)[0], -1, self.hidden_units])
        seq_embeddings += positional_embeddings
        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)
        seq_emb = tf.reshape(seq_attention, [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units])
        candidate_emb = self.item_embedding_layer(candidate)
        candidate_emb = tf.squeeze(candidate_emb, axis=0)
        candidate_emb = tf.reshape(tf.concat([candidate_emb, test_user_emb], 1), [-1, self.hidden_units])
        candidate_emb = tf.transpose(candidate_emb, perm=[1, 0])
        test_logits = tf.matmul(seq_emb, candidate_emb)
        test_logits = tf.reshape(test_logits, [tf.shape(input_seq)[0], self.seq_max_len, 1 + self.num_neg_test])
        test_logits = test_logits[:, -1, :]
        return test_logits

    def loss_function(self, pos_logits, neg_logits, istarget):
        if False:
            for i in range(10):
                print('nop')
        'Losses are calculated separately for the positive and negative\n        items based on the corresponding logits. A mask is included to\n        take care of the zero items (added for padding).\n\n        Args:\n            pos_logits (tf.Tensor): Logits of the positive examples.\n            neg_logits (tf.Tensor): Logits of the negative examples.\n            istarget (tf.Tensor): Mask for nonzero targets.\n\n        Returns:\n            float: Loss.\n        '
        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]
        loss = tf.reduce_sum(-tf.math.log(tf.math.sigmoid(pos_logits) + 1e-24) * istarget - tf.math.log(1 - tf.math.sigmoid(neg_logits) + 1e-24) * istarget) / tf.reduce_sum(istarget)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        loss += reg_loss
        return loss