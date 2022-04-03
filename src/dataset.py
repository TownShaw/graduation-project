'''
@Author: Xiao Tong
@FileName: dataset.py
@CreateTime: 2022-03-22 20:50:04
@Description:

'''
import torch
import numpy as np
from torch.utils.data import Dataset


class KhanDataset(Dataset):
    def __init__(self, images, subtitles, lens, labels, num_classes) -> None:
        super(KhanDataset, self).__init__()
        self.images = images
        self.subtitles = subtitles
        self.lens = lens
        self.labels = labels
        self.num_classes = num_classes

    def __getitem__(self, idx):
        multi_hot_label = torch.FloatTensor([1. if i in self.labels[idx] else 0. for i in range(self.num_classes)])
        return self.images[idx], self.subtitles[idx], self.lens[idx], multi_hot_label

    def __len__(self):
        return len(self.labels)


def collate_fn(batch_data):
    batch = {"images": [], "subtitles": [], "lens": [], "labels": [], "segments": []}
    for data in batch_data:
        images, subtitles, lens, labels = data
        batch["labels"].append(labels)
        segment = [len(section_images) for section_images in images]
        batch["segments"].append(segment)
        images = [torch.from_numpy(np.array(image)) for section_images in images for image in section_images]
        images = torch.stack(images, dim=0)
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
