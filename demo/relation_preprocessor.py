# -*- coding: utf-8 -*-
##
# @brief RelationPreprocessor class
# @author ss

import re
import os
import pandas as pd
#import nltk


class SemevalRelationPreprocessor(object):
    """
    This class is responsible of converting the brat annotation file format of SelEval task 8
    into standard format to be processed by our RelationMentionVectorizer class
    """
    def __init__(self, txt_input_dir):
        self.txt_input_dir = txt_input_dir
        # Remove stopwords does not improve performance
        #self.stopwords = nltk.corpus.stopwords.words("english")

    def get_processed_data(self):
        """
        Returns:
            mentions_train(list of dict), mentions_valid(list of dict): 
                Both of the form [{sentence_id: int, segments:[string,], 
                segment_labels:[string,], ent1:int, ent2:int},]
                           
        """
        mentions_train = self.get_mentions(datafile="TRAIN_FILE.TXT")
        mentions_valid = self.get_mentions(datafile="TEST_FILE_FULL.TXT")
        return mentions_train, mentions_valid
            
    def get_mentions(self, datafile):
        """
        Returns:
            list of dict: [{sentence_id: int, segments:[string,], segment_labels:[string,],
                           ent1:int, ent2:int},]
        """    
        mentions = []
        ent1, ent2 = -1, -1        
        with open(os.path.join(self.txt_input_dir, datafile)) as f:
            cnt, num = 0, 0
            for line in f:
                if cnt == 0:
                    seq, sentence = line.split('	')
                    sentence = sentence[1:-2]
                    ori_sentence = sentence
                    tmp = ""
                    for ch in sentence:
                        if ch not in [',', '.', '(', ')', '\'', ':']:  # exclude these characters
                            tmp += ch
                    sentence = tmp.lower()
                    # Join words in <>*</> with '_'
                    start = sentence.find('<e1>')
                    end = sentence.find('</e1>')
                    entity = sentence[start:end+5]
                    sentence = sentence.replace(entity, '_'.join(entity.split()))
                    start = sentence.find('<e2>')
                    end = sentence.find('</e2>')
                    entity = sentence[start:end+5]
                    sentence = sentence.replace(entity, '_'.join(entity.split()))
                    # exclude stopwords and pure numbers
                    segments = sentence.split()
                    tmp = []
                    for w in segments:
                        if not w.isdigit():  # (w not in self.stopwords)
                            tmp.append(w)
                    segments = tmp
                    # Get positions of entities and segments
                    for pos, segment in enumerate(segments):
                        m = re.search("<([^>\\s]+)[^>]*>((?:(?!<\\s*\\/\\s*\\1)[\\s\\S])*)<\\s*\\/\\s*\\1\\s*>",
                                      segment)
                        if m:
                            if m.group(1) == "e1":
                                ent1 = pos
                            elif m.group(1) == "e2":
                                ent2 = pos
                            else:
                                assert "Format error"
                            segments[pos] = ' '.join(m.group(2).split('_'))
                elif cnt == 1:
                    segment_labels = line.strip()
                cnt = (cnt+1) % 4
                # Record
                if cnt == 3:
                    mention = {"sentence_id": num, "sentence": ori_sentence, "segments": segments,
                               "segment_labels": segment_labels, "ent1": ent1, "ent2": ent2}
                    mentions.append(mention)
                    num += 1
        return mentions




