# -*- coding: utf-8 -*-

import sys

import os

import os.path

import time

from operator import itemgetter

import numpy as np

import pickle

from random import shuffle


class BatchData:

    def __init__(self, flist, modules, consts, options):

        self.batch_size = len(flist)

        self.x = np.zeros((consts["len_x"], self.batch_size), dtype=np.int64) # use x when copy mechanism is not added

        self.x_ext = np.zeros((consts["len_x"], self.batch_size), dtype=np.int64) # use x_ext when copy mechanism is added

        self.y = np.zeros((consts["len_y"], self.batch_size), dtype=np.int64) # use y when copy mechanism is not added and the unknown words are changed into the unk signal

        self.y_inp = np.zeros((consts["len_y"], self.batch_size), dtype=np.int64) # shifted input

        self.y_ext = np.zeros((consts["len_y"], self.batch_size), dtype=np.int64) # use y_ext when copy mechanism is added

        self.x_mask = np.zeros((consts["len_x"], self.batch_size, 1), dtype=np.int64) # masked metrix

        self.y_mask = np.zeros((consts["len_y"], self.batch_size, 1), dtype=np.int64) # masked metrix

        self.batch_context = np.zeros((consts["len_context"], self.batch_size), dtype=np.int64) # the preceding sentences

        self.context_mask = np.zeros((consts["len_context"], self.batch_size, 1), dtype=np.int64) # masked metrix

        self.batch_downstairs = np.zeros((consts["len_context"], self.batch_size), dtype=np.int64) # the following sentences

        self.downstairs_mask = np.zeros((consts["len_context"], self.batch_size, 1), dtype=np.int64) # masked metrix

        self.len_x = []

        self.len_y = []

        self.len_context = [] # the length of the preceding sentences

        self.len_downstair = [] # the length of the following sentences

        self.original_contents = []

        self.original_summarys = []

        self.original_contexts = []

        self.original_downstairs = []

        self.x_ext_words = []

        self.max_ext_len = 0

        w2i = modules["w2i"] #Word2index

        i2w = modules["i2w"] #Index2word

        dict_size = len(w2i)

        for idx_doc in range(len(flist)):

            if len(flist[idx_doc]) == 4:

                contents, summarys, contexts, downstairs = flist[idx_doc]

            else:

                print("ERROR!")

                return

            content, original_content = contents

            summary, original_summary = summarys

            context, original_context = contexts

            downstair, original_downstair = downstairs

            self.original_contents.append(original_content)

            self.original_summarys.append(original_summary)

            self.original_contexts.append(original_context)

            self.original_downstairs.append(original_downstair)

            xi_oovs = [] # stores the OOVs

            for idx_word in range(len(content)):

                # some sentences in duc is longer than len_x

                if idx_word == consts["len_x"]:
                    break

                w = content[idx_word]

                if w not in w2i:  # OOV

                    if w not in xi_oovs:
                        xi_oovs.append(w)

                    self.x_ext[idx_word, idx_doc] = dict_size + xi_oovs.index(w)  # 500005, 51000

                    w = i2w[modules["lfw_emb"]]

                else:

                    self.x_ext[idx_word, idx_doc] = w2i[w]

                self.x[idx_word, idx_doc] = w2i[w]

                self.x_mask[idx_word, idx_doc, 0] = 1

            self.len_x.append(np.sum(self.x_mask[:, idx_doc, :]))

            self.x_ext_words.append(xi_oovs)

            if self.max_ext_len < len(xi_oovs):
            
                self.max_ext_len = len(xi_oovs)

            if options["has_y"]:

                for idx_word in range(len(summary)):

                    w = summary[idx_word]

                    if w not in w2i:

                        if w in xi_oovs:

                            self.y_ext[idx_word, idx_doc] = dict_size + xi_oovs.index(w)

                        else:

                            self.y_ext[idx_word, idx_doc] = w2i[i2w[modules["lfw_emb"]]]  # unk

                        w = i2w[modules["lfw_emb"]]

                    else:

                        self.y_ext[idx_word, idx_doc] = w2i[w]

                    self.y[idx_word, idx_doc] = w2i[w]

                    if idx_word == 0:
                    
                        self.y_inp[idx_word, idx_doc] = modules["bos_idx"]

                    if idx_word < (len(summary) - 1):
                    
                        self.y_inp[idx_word + 1, idx_doc] = w2i[w]

                    if not options["is_predicting"]:
                        self.y_mask[idx_word, idx_doc, 0] = 1

                self.len_y.append(len(summary))

            else:

                self.y = self.y_mask = None

            if options["has_context"]:

                for idx_word in range(len(context)):

                    if idx_word == consts["len_context"]:
                        break

                    w = context[idx_word]

                    if w not in w2i:

                        self.batch_context[idx_word, idx_doc] = w2i[i2w[modules["lfw_emb"]]]  # unk

                    else:

                        self.batch_context[idx_word, idx_doc] = w2i[w]

                    self.context_mask[idx_word, idx_doc, 0] = 1

                self.len_context.append(np.sum(self.context_mask[:, idx_doc, :]))

                for idx_word in range(len(downstair)):

                    if idx_word == consts["len_context"]:
                        break

                    w = downstair[idx_word]

                    if w not in w2i:

                        self.batch_downstairs[idx_word, idx_doc] = w2i[i2w[modules["lfw_emb"]]]  # unk

                    else:

                        self.batch_downstairs[idx_word, idx_doc] = w2i[w]

                    self.downstairs_mask[idx_word, idx_doc, 0] = 1

                self.len_downstair.append(np.sum(self.downstairs_mask[:, idx_doc, :]))

            else:

                self.batch_context = self.context_mask = None

                self.batch_downstairs = self.downstairs_mask =None

        max_len_x = int(np.max(self.len_x)) # the maximum length of the original sentence of a batch

        max_len_y = int(np.max(self.len_y)) # the minimum length of the simplified sentence of a batch

        max_len_context = int(np.max(self.len_context))

        max_len_downstairs = int(np.max(self.len_downstair))

        self.x = self.x[0:max_len_x, :]

        self.x_ext = self.x_ext[0:max_len_x, :]

        self.x_mask = self.x_mask[0:max_len_x, :, :]

        self.y = self.y[0:max_len_y, :]

        self.y_inp = self.y_inp[0:max_len_y, :]

        self.y_ext = self.y_ext[0:max_len_y, :]

        self.y_mask = self.y_mask[0:max_len_y, :, :]

        self.batch_context = self.batch_context[0:max_len_context, :]

        self.context_mask = self.context_mask[0:max_len_context, :, :]

        self.batch_downstairs = self.batch_downstairs[0:max_len_downstairs, :]

        self.downstairs_mask = self.downstairs_mask[0:max_len_downstairs, :, :]

def get_data(xy_list, modules, consts, options):
    return BatchData(xy_list, modules, consts, options)


def batched(x_size, options, consts):
    batch_size = consts["testing_batch_size"] if options["is_predicting"] else consts["batch_size"]

    if options["is_debugging"]:
        x_size = 12

    ids = [i for i in range(x_size)]

    if not options["is_predicting"]:
        shuffle(ids)

    batch_list = []

    batch_ids = []

    for i in range(x_size):

        idx = ids[i]

        batch_ids.append(idx)

        if len(batch_ids) == batch_size or i == (x_size - 1):
            batch_list.append(batch_ids)

            batch_ids = []

    return batch_list, len(ids), len(batch_list)
