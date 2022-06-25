'''
@Author: Xiao Tong
@FileName: dataset.py
@CreateTime: 2022-04-30 20:01:55
@Description:

'''

import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from utils import get_stopwords, load_khan_data_by_id


class KhanDataset(Dataset):
    def __init__(self, config: dict, fileids: list, word2idx: dict) -> None:
        super(KhanDataset, self).__init__()
        self.config = config
        self.word2idx = word2idx
        self.id2labels = pickle.load(open(config["data"]["id2labels"], "rb"))
        # delete fileids that not in id2labels, which means delete data that doesn't have a label
        self.fileids = list(set(fileids).intersection(set(self.id2labels.keys())))
        self.stopwords = get_stopwords(self.config["data"]["stopwords"])
        self.num_classes = self.config["data"]["num_classes"]

    def __getitem__(self, idx):
        fileid = self.fileids[idx]
        filepath = os.path.join(self.config["data"]["sample_dir"], fileid, fileid + ".keyframes.pkl")
        images, subtitles, lens = load_khan_data_by_id(filepath,
                                                       self.word2idx,
                                                       self.stopwords,
                                                       max_seq_len=self.config["model"]["max_seq_len"],
                                                       pad_word=self.config["model"]["pad_word"])

        label = self.id2labels[fileid]
        multi_hot_label = torch.FloatTensor([1. if i in label else 0. for i in range(self.num_classes)])
        return images, subtitles, lens, multi_hot_label

    def __len__(self):
        return len(self.fileids)


def collate_fn(batch_data):
    batch = {"images": [], "subtitles": [], "lens": [], "labels": [], "segments": []}
    for data in batch_data:
        images, subtitles, lens, labels = data
        batch["labels"].append(labels)
        segment = [len(section_subtitles) for section_subtitles in subtitles]
        batch["segments"].append(segment)
        images = [torch.from_numpy(np.array(image, dtype=np.float32)) for section_images in images for image in section_images]
        images = torch.stack(images, dim=0).transpose(1, 3)
        subtitles = [torch.LongTensor(subtitle) for section_subtitles in subtitles for subtitle in section_subtitles]
        subtitles = torch.stack(subtitles, dim=0)
        lens = [length for section_lens in lens for length in section_lens]
        lens = torch.LongTensor(lens)
        batch["images"].append(images)
        batch["subtitles"].append(subtitles)
        batch["lens"].append(lens)
    batch["images"] = torch.cat(batch["images"], dim=0)
    batch["subtitles"] = torch.cat(batch["subtitles"], dim=0)
    batch["lens"] = torch.cat(batch["lens"], dim=0)
    batch["labels"] = torch.stack(batch["labels"], dim=0)
    return batch
