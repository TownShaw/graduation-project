'''
@Author: Xiao Tong
@FileName: train.py
@CreateTime: 2022-03-22 22:23:46
@Description:

'''

import os
import sys
import tqdm
import yaml
import torch
import pickle
import numpy as np
import torch.utils.data
import utils.utils as utils
from dataset import KhanDataset, collate_fn
from harnn import HARNN


def train(config: dict):
    logger = utils.getLogger(config["log"]["log_dir"], name=config["log"]["name"])

    word2vec_file = config["data"]["word2vec"]
    logger.info("Loading pretrained word embedding from '{0}'".format(word2vec_file))
    word2idx, pretrained_embedding = utils.load_word2vec_pretrained(word2vec_file)

    sample_dir = config["data"]["sample_dir"]
    logger.info("Loading data from '{0}'".format(sample_dir))
    images, subtitles, lens, labels = utils.load_khan_data(config, word2idx)

    child2parent = pickle.load(open(config["data"]["hierarchy"], "rb"))
    index2know = pickle.load(open(config["data"]["index2know"], "rb"))

    khan_dataset = KhanDataset(images, subtitles, lens, labels, num_classes=config["data"]["num_classes"])
    khan_dataloader = torch.utils.data.DataLoader(khan_dataset, batch_size=config["model"]["batch_size"], shuffle=True, collate_fn=collate_fn)

    for batch in khan_dataloader:
        images, subtitles, lens, labels, segments = batch["images"], batch["subtitles"], batch["lens"], batch["labels"], batch["segments"]


if __name__ == "__main__":
    config_dir = "config"
    config_file = "model.train.yaml"
    config = yaml.load(open(os.path.join(config_dir, config_file), "r", encoding="utf-8"), Loader=yaml.FullLoader)
    train(config=config)
