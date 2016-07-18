# -*- coding: utf-8 -*-
##
# @brief CNN relation extraction demo
# @author ss

import os
import sys
import shutil
import argparse
import pandas as pd
import numpy as np
import cPickle
from sklearn.metrics import (f1_score, accuracy_score, 
                             precision_score, recall_score, 
                             classification_report)
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

fdir = os.path.split(os.path.realpath(__file__))[0]
root = os.path.join(os.path.split(fdir)[0], 'src')
sys.path.append(root)

from cnn.cnn_theano import CNN
from relation_preprocessor import SemevalRelationPreprocessor
from vectorizer.relation_mention_vectorizer import RelationMentionVectorizer


def make_data(param, ShowExample=False):
    """ Preprocessing:
    1) Convert inputs data into the relation mentions:
        mentions: [{sentence_id: int, segments:[string,], segment_labels:[string,],
                              ent1:int, ent2:int},]
    2) Convert relation mentions into word representations:
        X: 4D-ndarray, whose shape is (number of samples, channel=1, sequence length, 
           vector length of word representation)
        y: 1D-ndarray, labels
        
    Args:
        param(dict): set by function get_base_param()
        ShowExample(bool): set to True if you want to see some train data examples
    Returns:
        X_train, y_train, X_valid, y_valid
    """
    print "Running preprocessing process..."
    rp = SemevalRelationPreprocessor(txt_input_dir=param['txt_input_dir'])
    mentions_train, mentions_valid = rp.get_processed_data()
    vectorizer = RelationMentionVectorizer(threads=param['threads'])
    vectorizer.fit((mentions_train, mentions_valid))
    print "Vectorizing data..."
    # Get X
    X_train = vectorizer.transform(mentions_train)
    X_valid = vectorizer.transform(mentions_valid)
    # Get y
    enc = preprocessing.LabelEncoder()    
    df_mentions = pd.DataFrame(mentions_train)
    df_mentions["segment_labels"] = enc.fit_transform(df_mentions["segment_labels"].values)
    y_train = df_mentions["segment_labels"].values
    df_mentions = pd.DataFrame(mentions_valid)
    df_mentions["segment_labels"] = enc.fit_transform(df_mentions["segment_labels"].values)
    y_valid = df_mentions["segment_labels"].values  
    print "Done vectorizing."  
    # show data examples    
    if ShowExample:
        print "Train data examples:"
        for i in xrange(X_train.shape[0]):
            if i % 1000 == 0:
                print "The", i, "sentence:", mentions_train[i]["sentence"]
                print "The", i, "mention:", mentions_train[i]["segments"]
                print "Positions (entity 1, entity 2):", mentions_train[i]["ent1"], mentions_train[i]["ent2"]
                print "Label:", mentions_train[i]["segment_labels"], "Encode:", y_train[i]
                for m in xrange(X_train.shape[1]):
                    print "The ", m, "row"
                    for n in xrange(X_train.shape[2]):
                        print X_train[i, m, n],
                    print
    X_train = np.reshape(X_train, [X_train.shape[0], -1, X_train.shape[1], X_train.shape[2]])
    X_valid = np.reshape(X_valid, [X_valid.shape[0], -1, X_valid.shape[1], X_valid.shape[2]])
    y_train = np.asarray(y_train, dtype=np.int32)
    y_valid = np.asarray(y_valid, dtype=np.int32)
    return X_train, y_train, X_valid, y_valid


def get_base_param():
    param = {'txt_input_dir': "../data/semeval_txt",
             'threads': 4,
             'batch_size': 20,
             # adadelta parameters
             'learning_rate': 0.01,
             'rho': 0.95,
             'epsilon': 1e-6,
             'num_epochs': 200}
    return param
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    param = get_base_param()
    
    if args.make_data:
        X_train, y_train, X_valid, y_valid = make_data(param, ShowExample=True)
        with open('../data/temp/data_train.pkl', 'wb') as f:
            cPickle.dump((X_train, y_train), f, protocol=cPickle.HIGHEST_PROTOCOL)
        with open('../data/temp/data_valid.pkl', 'wb') as f:
            cPickle.dump((X_valid, y_valid), f, protocol=cPickle.HIGHEST_PROTOCOL)
        # print info
        n_samp, max_w, len_feats = X_train.shape[0], X_train.shape[2], X_train.shape[3]
        print "Train data:"
        print "Number:", n_samp, ", Maximum window size:", max_w, ", Vector length of word representation:", len_feats
        n_samp, max_w, len_feats = X_valid.shape[0], X_valid.shape[2], X_valid.shape[3]
        print "Valid data:"
        print "Number:", n_samp, ", Maximum window size:", max_w, ", Vector length of word representation:", len_feats

    # Training the CNN          
    if args.train:    
        with open('../data/temp/data_train.pkl', 'rb') as f:
            X_train, y_train = cPickle.load(f)
        with open('../data/temp/data_valid.pkl', 'rb') as f:
            X_valid, y_valid = cPickle.load(f)
        print "Now training..."
        cnn = CNN(param)
        cnn.fit(X_train, y_train, X_valid, y_valid)
        with open('../data/temp/model.pkl', 'wb') as f:
            cPickle.dump(cnn, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    if args.predict:
        with open('../data/temp/data_valid.pkl', 'rb') as f:
            X_valid, y_valid = cPickle.load(f)
        with open('../data/temp/model.pkl', 'rb') as f:
            cnn = cPickle.load(f)
        y_predicted = cnn.transform(X_valid)
        assert len(y_valid) == len(y_predicted)
        print "Accuracy score:", accuracy_score(y_valid, y_predicted)
        print "Precision score:", precision_score(y_valid, y_predicted, average='macro')
        print "Recall score:", recall_score(y_valid, y_predicted, average='macro')
        print "F1 score:", f1_score(y_valid, y_predicted, average='macro')
        print classification_report(y_valid, y_predicted)


if __name__ == '__main__':
    main()
