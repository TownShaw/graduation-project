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


def dump_prediction(scores: np.ndarray, labels: np.ndarray, out_filename: str, threshold: float=0.5):
    with open(out_filename, "w", encoding="utf-8") as fout:
        for score, label in zip(scores, labels):
            predict_result = (score >= threshold).astype(np.int32)
            predict_indices = np.argsort(predict_result, kind="stable")[::-1][:np.sum(predict_result)][::-1]
            predict_scores = score[predict_indices]
            fout.write(json.dumps({"true_labels": label.tolist(), "predict_labels": predict_indices, "predict_scores": predict_scores}))


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

    best_auprc = 0.0
    model = HARNN(config, len(word2idx), pretrained_word_embedding=pretrained_embedding)
    model_save_path = os.path.join(config["data"]["model_save_dir"], config["data"]["model_name"])
    if not os.path.isfile(model_save_path):
        raise FileNotFoundError("Model not Found at '{}'!".format(model_save_path))
    else:
        print("Loading model from {} ...".format(model_save_path))
        checkpoint = torch.load(model_save_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        best_auprc = checkpoint["best_auprc"]
    print("Best-AUPRC: {}".format(best_auprc))
    model = model.to(config["device"])

    max_segment_num = 40
    print("Evaluating ...")

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
    EMR = utils.metric_EMR(total_scores, total_labels)
    print("Eval Results: Micro-Precision: {:.4f}, Micro-Recall: {:.4f}, Micro-F1: {:.4f}, AUPRC: {:.4f}, EMR: {:.4f}".format(precision, recall, f1, auprc, EMR))

    model_name = ".".join(config["data"]["model_name"].split(".")[:-2])
    dump_prediction(total_scores, total_labels, f"{model_name}.txt", config["model"]["threshold"])


if __name__ == "__main__":
    set_seed(2022)

    config_dir = "config"
    config_file = "HARNN.Khan.yaml"
    config = yaml.load(open(os.path.join(config_dir, config_file), "r", encoding="utf-8"), Loader=yaml.FullLoader)
    eval(config=config)
