'''
@Author: Xiao Tong
@FileName: dataset.py
@CreateTime: 2022-04-29 20:18:39
@Description:

'''

import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from HMCN.utils import get_stopwords, load_khan_data_by_id


class KhanDataset(Dataset):
    def __init__(self, config: dict, fileids: list, word2idx: dict, model_name: str) -> None:
        super(KhanDataset, self).__init__()
        self.config = config
        self.word2idx = word2idx
        self.id2labels = pickle.load(open(config["data"]["id2labels"], "rb"))
        # delete fileids that not in id2labels, which means delete data that doesn't have a label
        self.fileids = list(set(fileids).intersection(set(self.id2labels.keys())))
        self.stopwords = get_stopwords(self.config["data"]["stopwords"])
        self.num_classes = self.config["data"]["num_classes"]
        self.model_name = model_name

    def __getitem__(self, idx):
        fileid = self.fileids[idx]
        filepath = os.path.join(self.config["data"]["sample_dir"], fileid, fileid + ".keyframes.pkl")
        subtitles, lens = load_khan_data_by_id(filepath,
                                               self.word2idx,
                                               self.stopwords,
                                               max_seq_len=self.config[self.model_name]["max_seq_len"],
                                               pad_word=self.config[self.model_name]["pad_word"])

        label = self.id2labels[fileid]
        multi_hot_label = torch.FloatTensor([1. if i in label else 0. for i in range(self.num_classes)])
        return subtitles, lens, multi_hot_label

    def __len__(self):
        return len(self.fileids)


def collate_fn(batch_data):
    batch = {"subtitles": [], "lens": [], "labels": [], "segments": []}
    for data in batch_data:
        subtitles, lens, labels = data
        batch["labels"].append(labels)
        segment = [len(section_subtitles) for section_subtitles in subtitles]
        batch["segments"].append(segment)
        subtitles = [torch.LongTensor(subtitle) for section_subtitles in subtitles for subtitle in section_subtitles]
        subtitles = torch.stack(subtitles, dim=0)
        lens = [length for section_lens in lens for length in section_lens]
        lens = torch.LongTensor(lens)
        batch["subtitles"].append(subtitles)
        batch["lens"].append(lens)
    batch["subtitles"] = torch.cat(batch["subtitles"], dim=0)
    batch["lens"] = torch.cat(batch["lens"], dim=0)
    batch["labels"] = torch.stack(batch["labels"], dim=0)
    return batch
