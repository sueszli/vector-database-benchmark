from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import BertTokenizer
import re
import unicodedata
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

class TfidfRecommender:
    """Term Frequency - Inverse Document Frequency (TF-IDF) Recommender

    This class provides content-based recommendations using TF-IDF vectorization in combination with cosine similarity.
    """

    def __init__(self, id_col, tokenization_method='scibert'):
        if False:
            i = 10
            return i + 15
        "Initialize model parameters\n\n        Args:\n            id_col (str): Name of column containing item IDs.\n            tokenization_method (str): ['none','nltk','bert','scibert'] option for tokenization method.\n        "
        self.id_col = id_col
        if tokenization_method.lower() not in ['none', 'nltk', 'bert', 'scibert']:
            raise ValueError('Tokenization method must be one of ["none" | "nltk" | "bert" | "scibert"]')
        self.tokenization_method = tokenization_method.lower()
        self.tf = TfidfVectorizer()
        self.tfidf_matrix = dict()
        self.tokens = dict()
        self.stop_words = frozenset()
        self.recommendations = dict()
        self.top_k_recommendations = pd.DataFrame()

    def __clean_text(self, text, for_BERT=False, verbose=False):
        if False:
            print('Hello World!')
        'Clean text by removing HTML tags, symbols, and punctuation.\n\n        Args:\n            text (str): Text to clean.\n            for_BERT (boolean): True or False for if this text is being cleaned for a BERT word tokenization method.\n            verbose (boolean): True or False for whether to print.\n\n        Returns:\n            str: Cleaned version of text.\n        '
        try:
            text_norm = unicodedata.normalize('NFC', text)
            clean = re.sub('<.*?>', '', text_norm)
            clean = clean.replace('\n', ' ')
            clean = clean.replace('\t', ' ')
            clean = clean.replace('\r', ' ')
            clean = clean.replace('Â\xa0', '')
            clean = re.sub('([^\\s\\w]|_)+', '', clean)
            if for_BERT is False:
                clean = clean.lower()
        except Exception:
            if verbose is True:
                print('Cannot clean non-existent text')
            clean = ''
        return clean

    def clean_dataframe(self, df, cols_to_clean, new_col_name='cleaned_text'):
        if False:
            while True:
                i = 10
        "Clean the text within the columns of interest and return a dataframe with cleaned and combined text.\n\n        Args:\n            df (pandas.DataFrame): Dataframe containing the text content to clean.\n            cols_to_clean (list of str): List of columns to clean by name (e.g., ['abstract','full_text']).\n            new_col_name (str): Name of the new column that will contain the cleaned text.\n\n        Returns:\n            pandas.DataFrame: Dataframe with cleaned text in the new column.\n        "
        df = df.replace(np.nan, '', regex=True)
        df[new_col_name] = df[cols_to_clean].apply(lambda cols: ' '.join(cols), axis=1)
        if self.tokenization_method in ['bert', 'scibert']:
            for_BERT = True
        else:
            for_BERT = False
        df[new_col_name] = df[new_col_name].map(lambda x: self.__clean_text(x, for_BERT))
        return df

    def tokenize_text(self, df_clean, text_col='cleaned_text', ngram_range=(1, 3), min_df=0):
        if False:
            i = 10
            return i + 15
        'Tokenize the input text.\n        For more details on the TfidfVectorizer, see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html\n\n        Args:\n            df_clean (pandas.DataFrame): Dataframe with cleaned text in the new column.\n            text_col (str): Name of column containing the cleaned text.\n            ngram_range (tuple of int): The lower and upper boundary of the range of n-values for different n-grams to be extracted.\n            min_df (int): When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.\n\n        Returns:\n            TfidfVectorizer, pandas.Series:\n            - Scikit-learn TfidfVectorizer object defined in `.tokenize_text()`.\n            - Each row contains tokens for respective documents separated by spaces.\n        '
        vectors = df_clean[text_col]
        if self.tokenization_method in ['bert', 'scibert']:
            tf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')
            if self.tokenization_method == 'bert':
                bert_method = 'bert-base-cased'
            elif self.tokenization_method == 'scibert':
                bert_method = 'allenai/scibert_scivocab_cased'
            tokenizer = BertTokenizer.from_pretrained(bert_method)
            vectors_tokenized = vectors.copy()
            for i in range(0, len(vectors)):
                vectors_tokenized[i] = ' '.join(tokenizer.tokenize(vectors[i]))
        elif self.tokenization_method == 'nltk':
            token_dict = {}
            stemmer = PorterStemmer()

            def stem_tokens(tokens, stemmer):
                if False:
                    for i in range(10):
                        print('nop')
                stemmed = []
                for item in tokens:
                    stemmed.append(stemmer.stem(item))
                return stemmed

            def tokenize(text):
                if False:
                    i = 10
                    return i + 15
                tokens = nltk.word_tokenize(text)
                stems = stem_tokens(tokens, stemmer)
                return stems
            tf = TfidfVectorizer(tokenizer=tokenize, analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')
            vectors_tokenized = vectors
        elif self.tokenization_method == 'none':
            tf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=min_df, stop_words='english')
            vectors_tokenized = vectors
        self.tf = tf
        return (tf, vectors_tokenized)

    def fit(self, tf, vectors_tokenized):
        if False:
            return 10
        'Fit TF-IDF vectorizer to the cleaned and tokenized text.\n\n        Args:\n            tf (TfidfVectorizer): sklearn.feature_extraction.text.TfidfVectorizer object defined in .tokenize_text().\n            vectors_tokenized (pandas.Series): Each row contains tokens for respective documents separated by spaces.\n        '
        self.tfidf_matrix = tf.fit_transform(vectors_tokenized)

    def get_tokens(self):
        if False:
            i = 10
            return i + 15
        'Return the tokens generated by the TF-IDF vectorizer.\n\n        Returns:\n            dict: Dictionary of tokens generated by the TF-IDF vectorizer.\n        '
        try:
            self.tokens = self.tf.vocabulary_
        except Exception:
            self.tokens = 'Run .tokenize_text() and .fit_tfidf() first'
        return self.tokens

    def get_stop_words(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the stop words excluded in the TF-IDF vectorizer.\n\n        Returns:\n            list: Frozenset of stop words used by the TF-IDF vectorizer (can be converted to list).\n        '
        try:
            self.stop_words = self.tf.get_stop_words()
        except Exception:
            self.stop_words = 'Run .tokenize_text() and .fit_tfidf() first'
        return self.stop_words

    def __create_full_recommendation_dictionary(self, df_clean):
        if False:
            print('Hello World!')
        'Create the full recommendation dictionary containing all recommendations for all items.\n\n        Args:\n            pandas.DataFrame: Dataframe with cleaned text.\n        '
        cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        sorted_idx = np.argsort(cosine_sim, axis=1)
        data = list(df_clean[self.id_col].values)
        len_df_clean = len(df_clean)
        results = {}
        for (idx, row) in zip(range(0, len_df_clean), data):
            similar_indices = sorted_idx[idx][:-(len_df_clean + 1):-1]
            similar_items = [(cosine_sim[idx][i], data[i]) for i in similar_indices]
            results[row] = similar_items[1:]
        self.recommendations = results

    def __organize_results_as_tabular(self, df_clean, k):
        if False:
            for i in range(10):
                print('nop')
        'Restructures results dictionary into a table containing only the top k recommendations per item.\n\n        Args:\n            df_clean (pandas.DataFrame): Dataframe with cleaned text.\n            k (int): Number of recommendations to return.\n        '
        item_id = list()
        rec_rank = list()
        rec_score = list()
        rec_item_id = list()
        for _item_id in self.recommendations:
            rec_based_on = tmp_item_id = _item_id
            rec_array = self.recommendations.get(rec_based_on)
            tmp_rec_score = list(map(lambda x: x[0], rec_array))
            tmp_rec_id = list(map(lambda x: x[1], rec_array))
            item_id.extend([tmp_item_id] * k)
            rec_rank.extend(list(range(1, k + 1)))
            rec_score.extend(tmp_rec_score[:k])
            rec_item_id.extend(tmp_rec_id[:k])
        output_dict = {self.id_col: item_id, 'rec_rank': rec_rank, 'rec_score': rec_score, 'rec_' + self.id_col: rec_item_id}
        self.top_k_recommendations = pd.DataFrame(output_dict)

    def recommend_top_k_items(self, df_clean, k=5):
        if False:
            i = 10
            return i + 15
        'Recommend k number of items similar to the item of interest.\n\n        Args:\n            df_clean (pandas.DataFrame): Dataframe with cleaned text.\n            k (int): Number of recommendations to return.\n\n        Returns:\n            pandas.DataFrame: Dataframe containing id of top k recommendations for all items.\n        '
        if k > len(df_clean) - 1:
            raise ValueError('Cannot get more recommendations than there are items. Set k lower.')
        self.__create_full_recommendation_dictionary(df_clean)
        self.__organize_results_as_tabular(df_clean, k)
        return self.top_k_recommendations

    def __get_single_item_info(self, metadata, rec_id):
        if False:
            print('Hello World!')
        'Get full information for a single recommended item.\n\n        Args:\n            metadata (pandas.DataFrame): Dataframe containing item info.\n            rec_id (str): Identifier for recommended item.\n\n        Returns:\n            pandas.Series: Single row from dataframe containing recommended item info.\n        '
        rec_info = metadata.iloc[int(np.where(metadata[self.id_col] == rec_id)[0])]
        return rec_info

    def __make_clickable(self, address):
        if False:
            i = 10
            return i + 15
        'Make URL clickable.\n\n        Args:\n            address (str): URL address to make clickable.\n        '
        return '<a href="{0}">{0}</a>'.format(address)

    def get_top_k_recommendations(self, metadata, query_id, cols_to_keep=[], verbose=True):
        if False:
            i = 10
            return i + 15
        "Return the top k recommendations with useful metadata for each recommendation.\n\n        Args:\n            metadata (pandas.DataFrame): Dataframe holding metadata for all public domain papers.\n            query_id (str): ID of item of interest.\n            cols_to_keep (list of str): List of columns from the metadata dataframe to include\n                (e.g., ['title','authors','journal','publish_time','url']).\n                By default, all columns are kept.\n            verbose (boolean): Set to True if you want to print the table.\n\n        Returns:\n            pandas.Styler: Stylized dataframe holding recommendations and associated metadata just for the item of interest (can access as normal dataframe by using df.data).\n        "
        df = self.top_k_recommendations.loc[self.top_k_recommendations[self.id_col] == query_id].reset_index()
        df.drop([self.id_col], axis=1, inplace=True)
        metadata_cols = metadata.columns.values
        df[metadata_cols] = df.apply(lambda row: self.__get_single_item_info(metadata, row['rec_' + self.id_col]), axis=1)
        df.drop([self.id_col], axis=1, inplace=True)
        df = df.rename(columns={'rec_rank': 'rank', 'rec_score': 'similarity_score'})
        if len(cols_to_keep) > 0:
            cols_to_keep.insert(0, 'similarity_score')
            cols_to_keep.insert(0, 'rank')
            df = df[cols_to_keep]
        if 'url' in list(map(lambda x: x.lower(), metadata_cols)):
            format_ = {'url': self.__make_clickable}
            df = df.head().style.format(format_)
        if verbose:
            df
        return df