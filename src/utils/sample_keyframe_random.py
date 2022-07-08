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
import datetime
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


def seek_frame_by_idx(capture, index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = capture.read()
    if ret == True:
        frame = Image.fromarray(frame).resize((224, 224))
        return frame
    else:
        return None


def split_sections(capture, section_len: float=300, max_section_num: int=8):
    """
    
    """
    fps = capture.get(cv2.CAP_PROP_FPS)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 在两个帧之间进行二分查找, 找到关键帧
    low_index = 0
    # 视频首尾帧为关键帧
    keyframes = []
    while low_index != num_frames - 1:
        high_index = min(low_index + math.ceil(section_len * fps), num_frames - 1)
        keyframes.append(high_index)
        low_index = high_index
    keyframes = [0] + keyframes[:max_section_num - 1]
    if keyframes[-1] != num_frames - 1:
        keyframes.append(num_frames - 1)
    return keyframes


def extract_keyframes(subpath: str,
                      videopath: str,
                      section_len: float=300,
                      max_section_num: int=8) -> tuple:
    # split sections by video only
    capture = cv2.VideoCapture(videopath)
    fps = capture.get(cv2.CAP_PROP_FPS)
    section_keyframes = split_sections(capture,
                                       section_len=section_len,
                                       max_section_num=max_section_num)

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
    # if len(sub_segments) == 0:
    #     keyframes_with_text["keyframes"] = section_keyframes
    #     for idx in range(len(section_keyframes) - 1):
    #         keyframes_with_text["subtitles"].append([""])
    #     return section_keyframes, keyframes_with_text
    keyframes = []
    subtitles = []
    tmp_sub = ""
    subtitle_idx = 0
    for section_idx in range(len(section_keyframes) - 1):
        start_idx = section_keyframes[section_idx]
        end_idx = section_keyframes[section_idx + 1]
        while subtitle_idx < len(sub_segments) and sub_segments[subtitle_idx]["end"] <= end_idx:
            tmp_sub = tmp_sub + " " + sub_segments[subtitle_idx]["text"]
            subtitle_idx += 1
        keyframes_with_text["keyframes"].append([start_idx, end_idx])
        keyframes_with_text["subtitles"].append([tmp_sub])
        tmp_sub = ""

    for section_idx in range(len(keyframes_with_text["keyframes"])):
        for idx, frame_index in enumerate(keyframes_with_text["keyframes"][section_idx]):
            keyframes_with_text["keyframes"][section_idx][idx] = seek_frame_by_idx(capture, frame_index)
    test_segmentation(fileid, section_keyframes, keyframes_with_text)
    return section_keyframes, keyframes_with_text


def test_segmentation(fileid: str, section_keyframes: list, keyframes_with_text: dict):
    # Assertions
    assert len(keyframes_with_text["keyframes"]) == len(section_keyframes) - 1, "{} length of sections mismatch!".format(fileid)
    assert len(keyframes_with_text["keyframes"]) == len(keyframes_with_text["subtitles"]), \
        "{} length of section_keyframes mismatch length of section_subtitles!".format(fileid)

    assert len(keyframes_with_text["keyframes"]) > 0, "{} length of sections NOT bigger than 0!".format(fileid)
    assert len(keyframes_with_text["keyframes"]) <= 8, "{} section number bigger than 8!".format(fileid)
    for section_keyframes, section_subtitles in zip(*keyframes_with_text.values()):
        assert len(section_keyframes) == len(section_subtitles) + 1, "{} length of keyframes mismatch length of subtitles!".format(fileid)
        assert len(section_keyframes) > 0, "{} length of keyframes = 0!".format(fileid)
        # assert len(section_keyframes) == len(set(section_keyframes)), "{} exists duplicate keyframe!".format(fileid)
        for frame in section_keyframes:
            assert frame.size == (224, 224), "{} keyframe size not equal to (224, 224)!".format(fileid)


def batched_extract(dest_dir: str,
                    fileids: list,
                    index: int,
                    section_len: float=300,
                    max_section_num: int=8):
    for fileid in tqdm.tqdm(fileids, file=sys.stdout, position=index):
        save_dir = os.path.join(dest_dir, fileid)
        section_save_file = os.path.join(save_dir, fileid + ".sections.pkl")
        keyframe_save_file = os.path.join(save_dir, fileid + ".keyframes.pkl")

        try:
            subpath = os.path.join(root_dir, subtitle_dir, fileid + ".en.vtt")
            videopath = os.path.join(root_dir, video_dir, fileid + ".mp4")
            sections, keyframes_with_text = extract_keyframes(subpath,
                                                              videopath,
                                                              section_len=section_len,
                                                              max_section_num=max_section_num)
        except Exception as e:
            print(fileid, e, file=sys.stderr)
            continue
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
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
    max_seq_len = 200
    # 每个 video 内最多 section 个数
    max_section_num = 12
    section_len = 100.0

    # test case
    # fileid = "Nue0DINMRPM"
    # fileid = "Osl6HEDZZ-E"
    # subpath = os.path.join("data/subtitles", fileid + ".en.vtt")
    # videopath = os.path.join("data/videos", fileid + ".mp4")
    # sections, keyframes = extract_keyframes(subpath, videopath, section_len, max_section_num)
    # print("Done.")

    dest_dir = os.path.join(root_dir, data_dir, "random_samples_100")
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    # 确定需要进行分段的 ids, 以下为默认进行分段的 ids
    fileids = [filename.split(".")[0] for filename in os.listdir(os.path.join(root_dir, subtitle_dir))]

    boundary_time = float(datetime.datetime(2022, 6, 27, 0, 0).strftime("%s"))
    removed_ids = []
    for fileid in fileids:
        section_file = os.path.join(dest_dir, fileid, fileid + ".sections.pkl")
        keyframe_file = os.path.join(dest_dir, fileid, fileid + ".keyframes.pkl")

        # 若文件存在, 但是生成时间早于给定的时间, 则重新生成
        if os.path.isfile(section_file) and os.path.isfile(keyframe_file):
            if os.path.getmtime(section_file) >= boundary_time \
            and os.path.getmtime(keyframe_file) >= boundary_time \
            and fileid in fileids:
                removed_ids.append(fileid)     # 从 ids 列表中删除, 表示不需要重新生成
    for fileid in removed_ids:
        fileids.remove(fileid)

    # 需要自定义分段 ids 在下面定义, 将上面直接全部注释即可
    # fileids = ["Onkd8tChC2A"]

    processes = 8
    random.seed(2022)
    random.shuffle(fileids)
    num_per_process = math.ceil(len(fileids) / processes)

    process_list = []
    for idx in range(processes):
        process = multiprocessing.Process(target=batched_extract, args=(dest_dir,
                                                                        fileids[idx * num_per_process:(idx + 1) * num_per_process],
                                                                        idx,
                                                                        section_len,
                                                                        max_section_num))
        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()
        process.close()

    print("Done!")
