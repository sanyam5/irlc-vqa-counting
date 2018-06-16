from __future__ import print_function
import os
import json
try:
    import _pickle as pkl
except:
    import cPickle as pkl
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pkl.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pkl.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class HMQAFeatureDataset(Dataset):
    def __init__(self, img_id2hqma_idx, image_features, spatial_features, qid2count, qid2count2score,
                 name, dictionary):
        super(HMQAFeatureDataset, self).__init__()

        assert name in ["train", "dev", "test"]
        self.name = name
        self.qid2count = qid2count
        self.qid2count2score = qid2count2score
        self.qids = None
        if self.name == "train":
            self.part_qid2count = self.qid2count["train"]
            self.part_qid2count2score = self.qid2count2score["train"]
        else:
            self.part_qid2count = self.qid2count[self.name]
            self.part_qid2count2score = self.qid2count2score[self.name]

        self.dictionary = dictionary

        self.img_id2hqma_idx = img_id2hqma_idx
        self._features = image_features
        self._spatials = spatial_features

        self.entries = self.load_dataset()

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)

    def prepare_entries(self, questions, part_qid2count, part_qid2count2score, qid_prefix=''):
        entries = []

        set_qids = set(part_qid2count.keys())
        for question in questions:
            question_id = str(question["question_id"])
            if question_id not in set_qids:
                # print("{} is not there".format(question_id))
                # break
                continue

            image_id = question['image_id']
            img_hqma_idx = self.img_id2hqma_idx[image_id]
            count = part_qid2count[question_id]
            score2count = part_qid2count2score[question_id]
            question = question["question"]

            entries.append({
                "question_id": qid_prefix + question_id,
                "image_id": image_id,
                "img_hqma_idx": img_hqma_idx,
                "count": count,
                "score2count": score2count,
                "question": question,
            })

        return entries

    def load_dataset(self):
        """Load entries

        img_id2val: dict {img_id -> val} val can be used to retrieve image or features
        dataroot: root path of dataset
        name: 'train', 'val'
        """

        if self.name == "train":
            fname = "train"
            vqa_question_path = './data/v2_OpenEnded_mscoco_{}2014_questions.json'.format(fname)
            vqa_questions = sorted(json.load(open(vqa_question_path))['questions'], key=lambda x: x['question_id'])

            vqa_entries = self.prepare_entries(
                questions=vqa_questions,
                part_qid2count=self.part_qid2count["vqa"],
                part_qid2count2score=self.part_qid2count2score["vqa"],
                qid_prefix="vqa"
            )

            vgn_questions = sorted(json.load(open("./data/how_many_qa/vgn_ques.json")), key=lambda x: x['question_id'])
            vgn_entries = self.prepare_entries(
                questions=vgn_questions,
                part_qid2count=self.part_qid2count["visual_genome"],
                part_qid2count2score=self.part_qid2count2score["visual_genome"],
                qid_prefix="vgn"
            )

            return vqa_entries + vgn_entries

        elif self.name in ["dev", "test"]:
            fname = "val"
            question_path = './data/v2_OpenEnded_mscoco_{}2014_questions.json'.format(fname)
            questions = sorted(json.load(open(question_path))['questions'], key=lambda x: x['question_id'])

            val_entries = self.prepare_entries(
                questions=questions,
                part_qid2count=self.part_qid2count,
                part_qid2count2score=self.part_qid2count2score,
            )
            return val_entries
        else:
            raise Exception("uknown name '{}'".format(self.name))

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            assert len(tokens) == max_length
            entry['q_token'] = tokens

    def tensorize(self):
        # TODO: uncomment later
        self.features = torch.from_numpy(self._features)
        self.spatials = torch.from_numpy(self._spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            entry["count"] = torch.from_numpy(np.array([entry["count"]]))
            entry["score2count"] = torch.from_numpy(np.array(entry["score2count"])).float()

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['img_hqma_idx']]
        spatials = self.spatials[entry['img_hqma_idx']]

        question = entry['q_token']
        count = entry["count"]
        score2count = entry["score2count"]

        return features, spatials, question, count, score2count

    def __len__(self):
        return len(self.entries)
