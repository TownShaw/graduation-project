'''
@Author: Xiao Tong
@FileName: train.py
@CreateTime: 2022-03-22 22:23:46
@Description:

'''

import os
import sys
import tqdm
import json
import yaml
import torch
import numpy as np
import torch.utils.data
import utils.utils as utils
from dataset import KhanDataset, collate_fn
from harnn import HARNN
from sklearn.metrics import average_precision_score


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def count_violate_num(section_scores: np.ndarray, video_labels: np.ndarray, segments: list, threshold: float=0.5):
    violate_num, total_num = 0, 0
    start_idx, end_idx = 0, 0
    for vidx, video in enumerate(segments):
        labels = video_labels[vidx]
        start_idx = end_idx
        end_idx += len(video)
        for section_idx in range(start_idx, end_idx):
            total_num += 1
            precict_labels = (section_scores[section_idx] >= threshold).astype(np.int32)
            diff = precict_labels - labels
            violate_label_num = np.sum((diff > 0).astype(np.int32))
            if violate_label_num > 0:
                violate_num += 1
    return violate_num, total_num


def eval(config: dict):
    word2vec_file = config["data"]["word2vec"]
    word2idx, pretrained_embedding = utils.load_word2vec_pretrained(word2vec_file)

    datasets = json.load(open("data/dataset.json"))
    # train_set = ["00fgAG6VrRQ"]
    # validation_set = os.listdir("data/samples_test")
    validation_set = datasets["validation_set"]
    khan_dataset_validation = KhanDataset(config, validation_set, word2idx=word2idx)
    khan_dataloader_validation = torch.utils.data.DataLoader(khan_dataset_validation,
                                                             batch_size=config["model"]["batch_size"],
                                                             shuffle=False,
                                                             collate_fn=collate_fn)

    best_f1 = 0.0
    model = HARNN(config, len(word2idx), pretrained_word_embedding=pretrained_embedding)
    model_save_path = os.path.join(config["data"]["model_save_dir"], config["data"]["model_name"])
    if not os.path.isfile(model_save_path):
        raise FileNotFoundError("Model not Found at '{}'!".format(model_save_path))
    else:
        print("Loading model from {} ...".format(model_save_path))
        checkpoint = torch.load(model_save_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        best_f1 = checkpoint["best_f1"]
    print("Best-F1: {}".format(best_f1))
    model = model.to(config["device"])

    max_segment_num = 32
    print("Evaluating ...")

    section_violate_num = 0
    section_total_num = 0
    # run an eval epoch before training
    model.eval()
    TP, FP, FN = 0, 0, 0
    total_scores = []
    total_labels = []
    for eval_batch in tqdm.tqdm(khan_dataloader_validation):
        for mini_eval_batch in utils.iter_batch_data(eval_batch, max_segment_num):
            images = mini_eval_batch["images"].to(config["device"])
            subtitles = mini_eval_batch["subtitles"].to(config["device"])
            lens = mini_eval_batch["lens"]
            labels = mini_eval_batch["labels"].to(config["device"])
            segments = mini_eval_batch["segments"]
            image_segments = mini_eval_batch["image_segments"]

            section_scores, video_scores = model(images, (subtitles, lens), segments, image_segments)
            tmp_violate_num, tmp_section_num = count_violate_num(section_scores.cpu().detach().numpy(), labels.cpu().detach().numpy(), segments)
            section_violate_num += tmp_violate_num
            section_total_num += tmp_section_num

            mini_TP, mini_FP, mini_FN = utils.metric(video_scores.cpu().detach().numpy(),
                                                     labels.cpu().detach().numpy(),
                                                     threshold=config["model"]["threshold"],
                                                     num_classes_list=config["data"]["num_classes_list"])
            TP += mini_TP
            FP += mini_FP
            FN += mini_FN
            total_scores.append(video_scores.cpu().detach().numpy())
            total_labels.append(labels.cpu().detach().numpy())
    precision, recall, f1 = utils.calculate(TP, FP, FN)
    total_scores = np.concatenate(total_scores, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    auprc = average_precision_score(total_labels, total_scores, average="micro")
    print("Eval Results: Micro-Precision: {:.4f}, Micro-Recall: {:.4f}, Micro-F1: {:.4f}, AUPRC: {:.4f}, Violate-Ratio: {:.4f}".format(precision, recall, f1, auprc, section_violate_num / section_total_num))


if __name__ == "__main__":
    set_seed(2022)

    config_dir = "config"
    config_file = "HARNN.Khan.yaml"
    config = yaml.load(open(os.path.join(config_dir, config_file), "r", encoding="utf-8"), Loader=yaml.FullLoader)
    eval(config=config)
