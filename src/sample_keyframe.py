'''
@Author: Xiao Tong
@FileName: sample_keyframe.py
@CreateTime: 2022-03-04 10:52:49
@Description:
Extract keyframes from videos

1. 去背景

   - 取当前帧 8 角用于确定背景

2. 从前往后, 以增量式为原理, 找出非背景像素急剧减少的情况作为关键帧.
'''


import os
import sys
import cv2
import tqdm
import math
import webvtt
import random
import pickle
import multiprocessing
import numpy as np
from PIL import Image

# now on server or my laptop
on_server = True

if on_server:
    root_dir = "/share/tongxiao/graduation-project"
else:
    root_dir = "/home/xt/Codes/graduation-project"
data_dir = "data"
subtitle_dir = "data/subtitles"
video_dir = "data/videos"


def get_background(image: np.ndarray, edge_length: int) -> int:
    """
    @param: image
    @param: edge_length
    output: background pixel
    """
    height, width = image.shape[:2]
    middle_height = height // 2
    middle_width = width // 2
    half_edge_length = edge_length // 2
    all_mean = np.mean(image)
    top_left = np.mean(image[:edge_length, :edge_length])
    top_middle = np.mean(image[:edge_length, middle_width - half_edge_length:middle_width + half_edge_length])
    top_right = np.mean(image[:edge_length, width - edge_length:])
    middle_left = np.mean(image[middle_height - half_edge_length:middle_height + half_edge_length, :edge_length])
    middle_right = np.mean(image[middle_height - half_edge_length:middle_height + half_edge_length, width - edge_length:])
    down_left = np.mean(image[height - edge_length:, :edge_length])
    down_middle = np.mean(image[height - edge_length:, middle_width - half_edge_length:middle_width + half_edge_length])
    down_right = np.mean(image[height - edge_length:, width - edge_length:])
    background_pixel = np.mean(np.array([top_left, top_middle, top_right, middle_left, middle_right, down_left, down_middle, down_right, all_mean]))
    return background_pixel


def wipe_background(image: Image, edge_length: int=20, background_threshold: int=15) -> Image:
    """
    @param: image
    @param: background_pixel
    @param: background_threshold: background +- background_threshold 内的像素最低为 0, 即也视为 background
    """
    image = np.array(image)
    background_pixel = get_background(image, edge_length)
    diff_image = image.astype(np.int16) - background_pixel
    final_image = np.where(abs(diff_image) < background_threshold, np.zeros_like(diff_image), abs(diff_image))
    final_image = final_image.astype(np.uint8)
    return Image.fromarray(final_image)


def seek_frame_by_idx(capture, index: int, edge_length: int=20, background_threshold: int=15):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = capture.read()
    if ret == True:
        gray_frame = Image.fromarray(frame).resize((224, 224)).convert("L")
        return wipe_background(gray_frame, edge_length, background_threshold)
    else:
        return None


def is_incremental(image1: Image, image2: Image, background_threshold: int=15, minus_threshold: int=2000) -> bool:
    """
    判断 image2 是否由 image1 增量而来
    return: Bool
    """
    image1, image2 = np.array(image1).astype(np.int16), np.array(image2).astype(np.int16)
    diff = image2 - image1
    diff = np.where(abs(diff) > background_threshold, diff, np.zeros_like(diff))
    # print(np.sum((diff > 0).astype(np.int16)))
    # print(np.sum((diff < 0).astype(np.int16)))
    minus_num = np.sum((diff < 0).astype(np.int16))
    return minus_num < minus_threshold


def split_sections(capture, edge_length: int=20, background_threshold: int=15, minus_threshold: int=2000):
    """
    
    """
    # 初步搜索时间跨度为 20s
    interval = 20
    fps = capture.get(cv2.CAP_PROP_FPS)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 在两个帧之间进行二分查找, 找到关键帧
    low_index = 0
    high_index = int(min(interval * fps + low_index, num_frames - 1))
    low_frame = seek_frame_by_idx(capture, low_index, edge_length, background_threshold)
    high_frame = seek_frame_by_idx(capture, high_index, edge_length, background_threshold)
    # 视频首尾帧为关键帧
    keyframes = [low_index]
    while low_index != high_index:
        if not is_incremental(low_frame, high_frame, background_threshold, minus_threshold):
            # 若前后帧之间相差 5s 之内则直接将前帧当作关键帧, 减少搜索
            if high_index - low_index <= int(fps * 5):
                if low_index != 0:      # 因为 0 已经默认加入到 keyframes 中
                    keyframes.append(low_index)
                low_index = high_index
                low_frame = high_frame
                high_index = int(min(interval * fps + low_index, num_frames - 1))
                high_frame = seek_frame_by_idx(capture, high_index, edge_length, background_threshold)
                continue
            middle_index = low_index + (high_index - low_index) // 2
            middle_frame = seek_frame_by_idx(capture, middle_index, edge_length, background_threshold)
            if is_incremental(low_frame, middle_frame, background_threshold, minus_threshold):
                low_index = middle_index
                low_frame = middle_frame
            else:
                high_index = middle_index
                high_frame = middle_frame
            continue
        else:
            low_index = high_index
            low_frame = high_frame
            high_index = int(min(interval * fps + low_index, num_frames - 1))
            high_frame = seek_frame_by_idx(capture, high_index, edge_length, background_threshold)
    keyframes.append(num_frames - 1)
    return keyframes


def extract_keyframes(fileid: str, edge_length: int=20, background_threshold: int=15, minus_threshold: int=2000, max_seq_len: int=250) -> tuple:
    videopath = os.path.join(root_dir, video_dir, fileid + ".mp4")
    subpath = os.path.join(root_dir, subtitle_dir, fileid + ".en.vtt")

    # split sections by video only
    capture = cv2.VideoCapture(videopath)
    fps = capture.get(cv2.CAP_PROP_FPS)
    section_keyframes = split_sections(capture,
                                       edge_length=edge_length,
                                       background_threshold=background_threshold,
                                       minus_threshold=minus_threshold)

    # read subtitles
    vtt = webvtt.read(subpath)
    sub_segments = []
    for item in vtt:
        start_idx = int(item.start_in_seconds * fps)
        end_idx = int(item.end_in_seconds * fps)
        sub_segments.append({"start": start_idx, "end": end_idx, "text": item.text})

    # 根据 max_seq_len 和 sections 来分割 segments
    segment_idx = 0
    section_idx = 0
    keyframes_with_text = {"keyframes": [], "subtitles": []}
    keyframes = []
    subtitles = []
    tmp_sub = ""
    start_idx = section_keyframes[0]
    keyframes.append(start_idx)
    while segment_idx < len(sub_segments):
        # 存在一些数据字幕文件中的时长大于视频时长, 会出现 segment 边界大于 section 边界, 所以这里加一行直接跳过
        # 至于为什么我也不知道 :(
        try:
            t = section_keyframes[section_idx + 1]
        except IndexError:
            break

        if len(tmp_sub.split()) + len(sub_segments[segment_idx]["text"].split()) <= max_seq_len and sub_segments[segment_idx]["end"] <= section_keyframes[section_idx + 1]:
            tmp_sub += ' ' + sub_segments[segment_idx]["text"]
            segment_idx += 1
            continue
        # segment 末端已超过现在的 section 末端
        if sub_segments[segment_idx]["end"] > section_keyframes[section_idx + 1]:
            keyframes.append(section_keyframes[section_idx + 1])
            # 若该 segment 并未全在下一个 section 中, 则将对应的字幕文本算在现在的 section 中
            if not sub_segments[segment_idx]["start"] >= section_keyframes[section_idx + 1]:
                tmp_sub += ' ' + sub_segments[segment_idx]["text"]
                segment_idx += 1
            subtitles.append(tmp_sub)
            keyframes_with_text["keyframes"].append(keyframes)
            keyframes_with_text["subtitles"].append(subtitles)
            #
            keyframes, subtitles = [], []
            tmp_sub = ""
            keyframes.append(section_keyframes[section_idx + 1])
            section_idx += 1
        # 说明此时一定是 section 内长度超过 max_seq_len, 则分 segment
        else:
            keyframes.append(sub_segments[segment_idx]["end"])
            subtitles.append(tmp_sub)
            tmp_sub = ""
            segment_idx += 1
    # 加上最后一个 segment
    if keyframes[0] != section_keyframes[-1]:
        keyframes.append(section_keyframes[-1])
        subtitles.append(tmp_sub)
        keyframes_with_text["keyframes"].append(keyframes)
        keyframes_with_text["subtitles"].append(subtitles)
    # 过滤 section 中没有文本的
    tmp = {key: [] for key in keyframes_with_text.keys()}
    for frames, subs in zip(keyframes_with_text["keyframes"], keyframes_with_text["subtitles"]):
        if len(subs) == 1 and subs[0] == "":
            continue
        tmp["keyframes"].append(frames)
        tmp["subtitles"].append(subs)
    keyframes_with_text = tmp
    del tmp
    for section_idx in range(len(keyframes_with_text["keyframes"])):
        for idx, frame_index in enumerate(keyframes_with_text["keyframes"][section_idx]):
            keyframes_with_text["keyframes"][section_idx][idx] = seek_frame_by_idx(capture, frame_index)
    return section_keyframes, keyframes_with_text


def batched_extract(fileids: list, index: int, edge_length: int=20, background_threshold: int=15, minus_threshold: int=2000, max_seq_len: int=250):
    for fileid in tqdm.tqdm(fileids, file=sys.stdout, position=index):
        save_dir = os.path.join(root_dir, data_dir, "samples", fileid)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        section_save_file = os.path.join(save_dir, fileid + ".sections.pkl")
        keyframe_save_file = os.path.join(save_dir, fileid + ".keyframes.pkl")

        if os.path.isfile(section_save_file) and os.path.isfile(keyframe_save_file):
            continue
        sections, keyframes_with_text = extract_keyframes(fileid,
                                                          edge_length=edge_length,
                                                          background_threshold=background_threshold,
                                                          minus_threshold=minus_threshold,
                                                          max_seq_len=max_seq_len)
        pickle.dump(sections, open(section_save_file, "wb"))
        pickle.dump(keyframes_with_text, open(keyframe_save_file, "wb"))


if __name__ == "__main__":
    # 用于获取图像背景像素. 在图像 8 角处取 8 个边长为 edge_length 的正方形, 这些正方形内像素平均值作为背景像素
    edge_length = 20
    # 用于获取图像背景像素. 与背景像素相差 background_threshold 以内的像素都视为背景
    background_threshold = 15
    # 用于判定两图像之间是否为 incremental: 若将前后图像作 diff, 若为负值的像素个数大于 minus_threshold 则认为不是 incremental
    minus_threshold = 2000
    # 用于在 section 内细分 segment, 目的是充分利用文本信息. 在 section 内按文本最长为 max_seq_len 来分 segment
    max_seq_len = 250

    processes = 8
    fileids = [filename.split(".")[0] for filename in os.listdir(os.path.join(root_dir, subtitle_dir))]
    random.shuffle(fileids)
    num_per_process = math.ceil(len(fileids) / processes)

    process_list = []
    for idx in range(processes):
        process = multiprocessing.Process(target=batched_extract, args=(fileids[idx * num_per_process:(idx + 1) * num_per_process],
                                                                        idx,
                                                                        edge_length,
                                                                        background_threshold,
                                                                        minus_threshold,
                                                                        max_seq_len))
        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()
        process.close()

    print("Done!")
