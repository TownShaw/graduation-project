'''
@Author: Xiao Tong
@FileName: predict.py
@CreateTime: 2022-04-15 15:59:55
@Description:

'''

import os
import yaml
from harnn import HARNN
from utils.utils import *
from utils.sample_keyframe import extract_keyframes


def static_load(config_dir: str):
    """
    load all data which predict needed
    requires different config file names of different datasets
    config file name MUST be like `[model name].[dataset name].model`
    """
    assert os.path.isdir(config_dir), "Directory {} doesn't exists!".format(config_dir)
    meta_data = {}
    for config_file in os.listdir(config_dir):
        config_path = os.path.join(config_dir, config_file)
        if os.path.isfile(config_path):
            dataset_name = config_file.split(".")[1]
            config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
            meta_data[dataset_name] = {"config": config, "model": None, "word2vec": None, "child2parent": None, "index2know": None}
    for dataset_name in meta_data.keys():
        # load word2vec from gloVe files
        config = meta_data[dataset_name]["config"]
        word2idx, pretrained_embedding = load_word2vec_pretrained(config["data"]["word2vec"])
        meta_data[dataset_name]["word2vec"] = {"word2idx": word2idx, "embedding": pretrained_embedding}

        # load model from pretrained models
        model = HARNN(config, len(word2idx), pretrained_embedding)
        model_save_path = os.path.join(config["data"]["model_save_dir"], config["data"]["model_name"])
        ckpt = torch.load(model_save_path, map_location="cuda:0")
        model.load_state_dict(ckpt["model_state_dict"])
        meta_data[dataset_name]["model"] = model

        # load child2parent & index2know
        meta_data[dataset_name]["child2parent"] = pickle.load(open(config["data"]["hierarchy"], "rb"))
        meta_data[dataset_name]["index2know"] = pickle.load(open(config["data"]["index2know"], "rb"))

    return meta_data


def predict(meta_data: dict, dataset_name: str, subfile: str, videofile: str):
    """
    meta_data: meta data
    dataset_name: dataset name, used for select models & knowledge system
    subfile: subtitles file of video
    videofile: educational video which to be predicted
    """
    config = meta_data[dataset_name]["config"]
    stopwords = get_stopwords(config["data"]["stopwords"])
    sections, keyframes_with_texts = extract_keyframes(subfile, videofile)

    images, subtitles = keyframes_with_texts["keyframes"], keyframes_with_texts["subtitles"]
    lens = []
    for idx, section_subtitles in enumerate(subtitles):
        section_subtitles, section_lens = tokenize_and_pad(section_subtitles,
                                                           stopwords=stopwords,
                                                           word2idx=meta_data[dataset_name]["word2vec"]["word2idx"],
                                                           max_seq_len=config["model"]["max_seq_len"],
                                                           pad_word=config["model"]["pad_word"])
        subtitles[idx] = section_subtitles
        lens.append(section_lens)
    image_segment = [[len(section_images) for section_images in images]]
    segment = [[len(section_subtitles) for section_subtitles in subtitles]]
    images = [torch.from_numpy(np.array(image, dtype=np.float32)) for section_images in images for image in section_images]
    images = torch.stack(images, dim=0).transpose(1, 3)
    subtitles = [torch.LongTensor(subtitle) for section_subtitles in subtitles for subtitle in section_subtitles]
    subtitles = torch.stack(subtitles, dim=0)
    lens = torch.LongTensor([length for section_lens in lens for length in section_lens])

    section_scores, video_scores = meta_data[dataset_name]["model"](images, (subtitles, lens), segment, image_segment)

    section_labels_indices, section_labels_names = get_labels_by_threshold(section_scores.cpu().detach().numpy(),
                                                                           index2know=meta_data[dataset_name]["index2know"],
                                                                           threshold=meta_data[dataset_name]["config"]["model"]["threshold"])
    video_labels_indices, video_labels_names = get_labels_by_threshold(video_scores.cpu().detach().numpy(),
                                                                       index2know=meta_data[dataset_name]["index2know"],
                                                                       threshold=meta_data[dataset_name]["config"]["model"]["threshold"])
    return sections, section_labels_names, video_labels_names


if __name__ == "__main__":
    pass
