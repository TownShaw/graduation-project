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
import torch.utils.data
import utils.utils as utils
from dataset import KhanDataset, collate_fn
from harnn import HARNN, HierarchyLossWithSegments


def train(config: dict):
    logger = utils.getLogger(config["log"]["log_dir"], name=config["log"]["name"])

    word2vec_file = config["data"]["word2vec"]
    logger.info("Loading pretrained word embedding from '{0}'".format(word2vec_file))
    word2idx, pretrained_embedding = utils.load_word2vec_pretrained(word2vec_file)

    sample_dir = config["data"]["sample_dir"]
    logger.info("Loading data from '{0}'".format(sample_dir))
    datasets = json.load(open("data/dataset.json"))
    # train_set = ["00fgAG6VrRQ"]
    # validation_set = os.listdir("data/samples_test")
    train_set, validation_set = datasets["train_set"], datasets["validation_set"]
    train_data = utils.load_khan_data(config, train_set, word2idx)
    validation_data = utils.load_khan_data(config, validation_set, word2idx)

    khan_dataset_train = KhanDataset(*train_data, num_classes=config["data"]["num_classes"])
    khan_dataloader_train = torch.utils.data.DataLoader(khan_dataset_train,
                                                        batch_size=config["model"]["batch_size"],
                                                        shuffle=True,
                                                        collate_fn=collate_fn)
    khan_dataset_validation = KhanDataset(*validation_data, num_classes=config["data"]["num_classes"])
    khan_dataloader_validation = torch.utils.data.DataLoader(khan_dataset_validation,
                                                             batch_size=config["model"]["batch_size"],
                                                             shuffle=False,
                                                             collate_fn=collate_fn)

    best_f1 = 0.0
    model = HARNN(config, len(word2idx), pretrained_label_embedding=pretrained_embedding).to(config["device"])
    model_save_path = os.path.join(config["data"]["model_save_dir"], config["data"]["model_name"])
    if not os.path.isdir(config["data"]["model_save_dir"]):
        os.mkdir(config["data"]["model_save_dir"])
    if os.path.isfile(model_save_path):
        logger.info("Loading model from {} ...".format(model_save_path))
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_f1 = checkpoint["best_f1"]
    logger.info("Best-F1: {}".format(best_f1))

    optim = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    loss_fn = HierarchyLossWithSegments()

    max_segment_num = 32
    logger.info("Start training ...")

    # run an eval epoch before training
    model.eval()
    for eval_batch in tqdm.tqdm(khan_dataloader_validation):
        TP, FP, FN = 0, 0, 0
        for mini_eval_batch in utils.iter_batch_data(eval_batch, max_segment_num):
            images = mini_eval_batch["images"].to(config["device"])
            subtitles = mini_eval_batch["subtitles"].to(config["device"])
            lens = mini_eval_batch["lens"]
            labels = mini_eval_batch["labels"].to(config["device"])
            segments = mini_eval_batch["segments"]
            image_segments = mini_eval_batch["image_segments"]

            _, video_scores = model(images, (subtitles, lens), segments, image_segments)
            mini_TP, mini_FP, mini_FN = utils.metric(video_scores.cpu().detach().numpy(),
                                                     labels.cpu().detach().numpy(),
                                                     threshold=config["model"]["threshold"],
                                                     num_classes_list=config["data"]["num_classes_list"])
            TP += mini_TP
            FP += mini_FP
            FN += mini_FN
    precision, recall, f1 = utils.calculate(TP, FP, FN)
    logger.info("Eval Results: Micro-Precision: {:.4f}, Micro-Recall: {:.4f}, Micro-F1: {:.4f}".format(precision, recall, f1))
    logger.info("Eval Best-F1: {:.4f}".format(best_f1))

    for epoch in range(config["model"]["epochs"]):
        logger.info("Epoch: {}".format(epoch + 1))
        total_loss = 0
        tmp_step = 0
        outputs_list = []
        labels_list = []
        model.train()
        for batch in tqdm.tqdm(khan_dataloader_train):
            for mini_batch in utils.iter_batch_data(batch, max_segment_num):
                images = mini_batch["images"].to(config["device"])
                subtitles = mini_batch["subtitles"].to(config["device"])
                lens = mini_batch["lens"]
                labels = mini_batch["labels"].to(config["device"])
                segments = mini_batch["segments"]
                image_segments = mini_batch["image_segments"]

                optim.zero_grad()
                section_scores, video_scores = model(images, (subtitles, lens), segments, image_segments)
                loss = loss_fn(section_scores, video_scores, labels, segments)
                loss.backward()
                optim.step()

                total_loss += loss.item()
                outputs_list.append(video_scores.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())

            tmp_step += 1
            if tmp_step % 100 == 0:
                outputs = torch.cat(outputs_list, dim=0)
                eval_labels = torch.cat(labels_list, dim=0)
                precision, recall, f1 = utils.metric(outputs,
                                                     eval_labels,
                                                     threshold=config["model"]["threshold"],
                                                     num_classes_list=config["data"]["num_classes_list"])
                logger.info("Epoch: {}, Step: {}, Train Loss: {:.4f}, Precsion: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(epoch + 1,
                                                                                                                           tmp_step,
                                                                                                                           total_loss / 100,
                                                                                                                           precision,
                                                                                                                           recall,
                                                                                                                           f1))
                outputs_list = []
                labels_list = []
                total_loss = 0.0

        model.eval()
        for eval_batch in tqdm.tqdm(khan_dataloader_validation):
            TP, FP, FN = 0, 0, 0
            for mini_eval_batch in utils.iter_batch_data(eval_batch, max_segment_num):
                images = mini_eval_batch["images"].to(config["device"])
                subtitles = mini_eval_batch["subtitles"].to(config["device"])
                lens = mini_eval_batch["lens"]
                labels = mini_eval_batch["labels"].to(config["device"])
                segments = mini_eval_batch["segments"]
                image_segments = mini_eval_batch["image_segments"]

                _, video_scores = model(images, (subtitles, lens), segments, image_segments)
                mini_TP, mini_FP, mini_FN = utils.metric(video_scores.cpu().detach().numpy(),
                                                         labels.cpu().detach().numpy(),
                                                         threshold=config["model"]["threshold"],
                                                         num_classes_list=config["data"]["num_classes_list"])

                TP += mini_TP
                FP += mini_FP
                FN += mini_FN
        precision, recall, f1 = utils.calculate(TP, FP, FN)
        if best_f1 < f1:
            best_f1 = f1
            checkpoint = {"model_state_dict": model.state_dict(), "best_f1": best_f1}
            torch.save(checkpoint, os.path.join(config["data"]["model_save_dir"], config["data"]["model_name"]))
        logger.info("Eval Results: Micro-Precision: {:.4f}, Micro-Recall: {:.4f}, Micro-F1: {:.4f}".format(precision, recall, f1))
        logger.info("Eval Best-F1: {:.4f}".format(best_f1))


if __name__ == "__main__":
    torch.manual_seed(2022)

    config_dir = "config"
    config_file = "model.train.yaml"
    config = yaml.load(open(os.path.join(config_dir, config_file), "r", encoding="utf-8"), Loader=yaml.FullLoader)
    train(config=config)
