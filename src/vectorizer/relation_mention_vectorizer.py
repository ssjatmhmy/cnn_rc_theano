# -*- coding: utf-8 -*-
##
# @brief RelationMentionVectorizer class
# @author ss

from sklearn.base import TransformerMixin
import wordvectorizer
import numpy as np
from wordvectorizer import WordVectorizer
import threading


class TransformThread(threading.Thread):
    def __init__(self, thread_id, mentions, mats_for_segments, mat_pos_lookup, mentions_out):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.mentions = mentions
        self.mats_for_segments = mats_for_segments
        self.mat_pos_lookup = mat_pos_lookup
        self.mentions_out = mentions_out

    def run(self):
        print "Starting Thread: %s" % self.thread_id
        sentence_ids = [x["sentence_id"] for x in self.mentions]
        for c, i in enumerate(self.mentions):
            if c % 100 == 0:
                print "vectorizing %s out of %s" %(c, len(self.mentions))
            x1 = self.mats_for_segments[sentence_ids[c]]
            # position with respect to ent1
            x2 = self.mat_pos_lookup(i["ent1"])     # dimension self.m * _
            # position with respect to ent2
            x3 = self.mat_pos_lookup(i["ent2"])     # dimension self.m * _
            x = np.hstack([x1, x2, x3])         # merging different parts of vector representation of words
            self.mentions_out[sentence_ids[c], :, :] = x
        print "Exiting Thread:" + self.name


class RelationMentionVectorizer(TransformerMixin):
    """ Transform formatted sentences and others into matrices for CNN input """
    def __init__(self, position_vector=True, mat_position_size=50, ner=False, pos=False, dependency=False, threads=6):
        self.position_vector = position_vector
        # mat_position vectors will be filled when calling fit function
        self.mat_position = None
        self.mat_position_size = mat_position_size
        self.word_vectorizer = WordVectorizer(ner=ner, pos=pos, dependency=dependency)
        self.threads = threads
        # sizes of the output sequence matrix
        # m is number of words in the sequence
        # n is the size of the vector representation of each word in the sequence
        self.m = None
        self.n = self.word_vectorizer.model.vector_size + 2*self.mat_position_size

    def transform(self, mentions, **transform_params):
        """
        Args:
            mentions(list of dict): [{sentence_id:[],segments:[],segment_labels:[],ent1:int, ent2:int}, ]
                segments: list of strings of max length m (padding smaller sizes sequences with zeros)
                segments labels: list of strings
                ent1,ent2: position of entity1 and entity2 in segments    0 <= ent1, ent2 < self.m
            transform_params: Unused
        Returns:
            mentions_out(np.ndarray): vector representation of each segment, l*m*n ndarray
                l = len(mentions), m = self.m, n = self.n
        """
        mentions_out = np.zeros([len(mentions), self.m, self.n], np.float32)
        segments = [x["segments"] for x in mentions]

        mats_for_segments = []
        for seg in segments:
            x1 = [self.word_vectorizer.word2vec(w) for w in seg]
            x1 = np.array(x1, dtype=np.float32)
            # padding with zeros
            pad_size = self.m - x1.shape[0]
            if pad_size > 0:
                temp = np.zeros((pad_size, self.word_vectorizer.model.vector_size))
                x1 = np.vstack([x1, temp])
            mats_for_segments.append(x1)

        thread_obj = []
        for thread_id, block_of_mentions in enumerate(np.array_split(mentions, self.threads)):
            t = TransformThread(thread_id, block_of_mentions, mats_for_segments,
                                self.mat_pos_lookup, mentions_out)
            thread_obj.append(t)
            t.start()
        for t in thread_obj:
            t.join()
        return mentions_out

    def fit(self, mentions_all, y=None, **fit_params):
        """
        Args:
            mentions(list of dict): [{segments:[],segment_labels:[],ent1:int, ent2:int}, ]
                segments: list of strings of max length m (padding smaller sizes sequences with zeros)
                segments labels: list of strings
                ent1,ent2: position of entity1 and entity2 in segments    0 <= ent1, ent2 < m
            y: Unused
            fit_params: Unused
        """
        mentions_train, mentions_valid = mentions_all
        mentions = mentions_train + mentions_valid
        l = max([len(x["segments"]) for x in mentions])
        if (l % 2) != 0:
            l += 1
        self.m = l
        # original index = -l+1,....,0,...l-1
        # array index    = 0,.......,(l-1),...(2xl)-1
        self.mat_position = np.random.rand((2*l)-1, self.mat_position_size)
        return self

    def mat_pos_lookup(self, p):
        """
        Args:
            p: position of entity
        Returns:
            array of dimension self.m * self.mat_position_size

        example : if ent1 = 2 self.m = 10   i.e. : (w0, w1, w2(e1), w3, w4, w5, w6, w7, w8, w9)
                  return: mat_position[-2:8]   === add (l-1) to get indices between (0,2l-1) ===>  mat_position[7:17]
        """
        start = -p + self.m - 1
        end = start + self.m
        return self.mat_position[start:end]

