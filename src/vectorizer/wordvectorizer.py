# -*- coding: utf-8 -*-
##
# @brief word vectorizer class
# @author ss

import numpy as np
import gensim
from sklearn.base import TransformerMixin
from nltk.tokenize import TreebankWordTokenizer
from os import path

_W2V_BINARY_PATH = path.join(path.dirname(path.abspath(__file__)),
                             "../../data/misc/word2vec/GoogleNews-vectors-negative300.bin.gz")


class WordVectorizer(TransformerMixin):
    """ Convert each word of the input sentences into word representation """
    def __init__(self, ner=True, pos=True, dependency=False, embeddings="word2vec",
                 tokenizer=None, w2v_path=_W2V_BINARY_PATH):
        """
        Args:
            ner(bool): add named entity recognition features in the feature vector or not
            pos(bool): add part of speech tagging in the features in the feature vector or not
            dependency(bool): add dependency features or not
            embeddings(bool): name of which word vectors to use, default word2vec
        """
        if tokenizer is None:
            self.tokenize = TreebankWordTokenizer().tokenize
        if embeddings == "word2vec":
            print "loading word2vec model ..."
            self.model = gensim.models.Word2Vec.load_word2vec_format(w2v_path, binary=True)
            print "finished loading word2vec !!"
        self.ner = ner
        self.pos = pos
        self.dependency = dependency

    def transform(self, sentences, **transform_params):
        """
        Args:
            sentences(list of strings): iterator of sentences
            transform_params: Unused
        Returns:
            word_mat(np.ndarray): csr matrix each row contains a word (tokenized using standard tokenizer)
                            in sequence and columns indicating feature vector.
        """
        feature_vector_size = 0
        if self.pos:
            feature_vector_size += 1
        if self.ner:
            feature_vector_size += 1
        if self.model:
            feature_vector_size += self.model.vector_size
        # large matrix containing words per row and features per column
        word_mat = np.zeros((0, feature_vector_size), np.float32)
        word_list = []

        for s in sentences:
            tokens = self.tokenize(s)
            word_list += tokens
            # features for words per sentence # empty now hstack later
            words_features = np.zeros((len(tokens), 0))
            if self.model:
                word_vec = np.zeros((len(tokens), self.model.vector_size), np.float32)
                for i, w in enumerate(tokens):
                    word_vec[i] = self.word2vec(w)
                words_features = np.hstack([words_features, word_vec])
            if self.pos:
                # todo
                pass
                # posvec = np.zeros((len(tokens), 1), np.float32)
                # words_features = np.hstack([words_features, posvec])
            if self.ner:
                # todo
                pass
                # nervec = np.zeros((len(tokens), 1), np.float32)
                # words_features = np.hstack([words_features, nervec])
            if self.dependency:
                # todo
                # depvec = np.zeros((len(tokens), 1), np.float32)
                # words_features = np.hstack([words_features, depvec)
                pass
            # matrix nxm
            # n : number of words in in a sentence
            # m : number of features per word
            word_mat = np.vstack([word_mat, words_features])
        return word_mat, word_list

    def fit(self, X, y=None, **fit_params):
        return self

    def word2vec(self, phrase):
        """
        Using loaded word2vec model given a vector return it's equivalent word2vec representation
         - if word is not existing, replace by zero vector
         - if word contain one or many words inside tokenize and use average representations of all vectors
            (used to generate word vectors for segments of multiple words as well)

        Args:
            phrase (strings): a word or a phrase of multiple words
        Returns:
            raw numpy vector of a word (dtype = float32)
        """
        def lookup(word):
            if word in self.model:
                return self.model[word]
            else:
                return np.zeros(self.model.vector_size, np.float32)

        words = self.tokenize(phrase)
        return np.average([lookup(w) for w in words], axis=0)


