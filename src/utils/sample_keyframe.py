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


def seek_frame_by_idx(capture, index: int, edge_length: int=20, background_threshold: int=15, wipe: bool=False):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = capture.read()
    if ret == True:
        frame = Image.fromarray(frame).resize((224, 224))
        if wipe:
            return wipe_background(frame.convert("L"), edge_length, background_threshold)
        else:
            return frame
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
    return minus_num < minus_threshold, minus_num


def split_sections(capture, edge_length: int=20, background_threshold: int=15, minus_threshold: int=2000, max_section_num: int=8):
    """
    
    """
    # 初步搜索时间跨度为 20s
    interval = 20
    fps = capture.get(cv2.CAP_PROP_FPS)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 在两个帧之间进行二分查找, 找到关键帧
    low_index = 0
    high_index = int(min(interval * fps + low_index, num_frames - 1))
    low_frame = seek_frame_by_idx(capture, low_index, edge_length, background_threshold, wipe=True)
    high_frame = seek_frame_by_idx(capture, high_index, edge_length, background_threshold, wipe=True)
    # 视频首尾帧为关键帧
    keyframes = []
    while low_index != high_index:
        is_incre, minus_num = is_incremental(low_frame, high_frame, background_threshold, minus_threshold)
        if not is_incre:
            # 若前后帧之间相差 5s 之内则直接将前帧当作关键帧, 减少搜索
            if high_index - low_index <= int(fps * 5):
                if low_index != 0:      # 因为 0 已经默认加入到 keyframes 中
                    keyframes.append((low_index, minus_num))
                low_index = high_index
                low_frame = high_frame
                high_index = int(min(interval * fps + low_index, num_frames - 1))
                high_frame = seek_frame_by_idx(capture, high_index, edge_length, background_threshold, wipe=True)
                continue
            middle_index = low_index + (high_index - low_index) // 2
            middle_frame = seek_frame_by_idx(capture, middle_index, edge_length, background_threshold, wipe=True)
            is_incre, _ = is_incremental(low_frame, middle_frame, background_threshold, minus_threshold)
            if is_incre:
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
            high_frame = seek_frame_by_idx(capture, high_index, edge_length, background_threshold, wipe=True)
    keyframes = sorted(keyframes, key=lambda x: x[1])[:max_section_num - 1]
    keyframes = [item[0] for item in keyframes]
    keyframes += [0, num_frames - 1]
    keyframes.sort()
    return keyframes


def extract_keyframes(subpath: str,
                      videopath: str,
                      edge_length: int=20,
                      background_threshold: int=15,
                      minus_threshold: int=2000,
                      max_seq_len: int=200,
                      max_section_num: int=8) -> tuple:
    # split sections by video only
    capture = cv2.VideoCapture(videopath)
    fps = capture.get(cv2.CAP_PROP_FPS)
    section_keyframes = split_sections(capture,
                                       edge_length=edge_length,
                                       background_threshold=background_threshold,
                                       minus_threshold=minus_threshold,
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
    start_idx = section_keyframes[0]
    keyframes.append(start_idx)
    while segment_idx < len(sub_segments):
        # 存在一些数据字幕文件中的时长大于视频时长, 会出现 segment 边界大于 section 边界, 所以这里加一行直接跳过
        # 至于为什么我也不知道 :(
        try:
            t = section_keyframes[section_idx + 1]
        except IndexError:
            break

        if len(tmp_sub.split()) + len(sub_segments[segment_idx]["text"].split()) <= max_seq_len and sub_segments[segment_idx]["end"] < section_keyframes[section_idx + 1]:
            tmp_sub += ' ' + sub_segments[segment_idx]["text"]
            segment_idx += 1
            continue
        # segment 末端已超过现在的 section 末端
        if sub_segments[segment_idx]["end"] >= section_keyframes[section_idx + 1]:
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
            # segment_idx += 1
    # 加上后面的 segment & sections                                   原注释: (最后一个 segment)?
    if keyframes[-1] != section_keyframes[-1]:
        # 找到大于且距离 keyframes[-1] 最近的 section_keyframe
        nearest_keyframe = section_keyframes[-1]
        for frame_idx in section_keyframes:
            if frame_idx > keyframes[-1]:
                nearest_keyframe = frame_idx
                break
        # 加上该 section 中的最后一个 segment
        keyframes.append(nearest_keyframe)
        subtitles.append(tmp_sub)
        keyframes_with_text["keyframes"].append(keyframes)
        keyframes_with_text["subtitles"].append(subtitles)
        # 若后面还有无字的 sections
        if nearest_keyframe != section_keyframes[-1]:
            index = section_keyframes.index(nearest_keyframe)
            for idx in range(index, len(section_keyframes) - 1):
                keyframes_with_text["keyframes"].append([section_keyframes[idx], section_keyframes[idx + 1]])
                keyframes_with_text["subtitles"].append([""])
    # 过滤 section 中没有文本的
    # tmp = {key: [] for key in keyframes_with_text.keys()}
    # for frames, subs in zip(keyframes_with_text["keyframes"], keyframes_with_text["subtitles"]):
    #     if len(subs) == 1 and subs[0] == "":
    #         continue
    #     tmp["keyframes"].append(frames)
    #     tmp["subtitles"].append(subs)
    # keyframes_with_text = tmp
    # del tmp

    for section_idx in range(len(keyframes_with_text["keyframes"])):
        for idx, frame_index in enumerate(keyframes_with_text["keyframes"][section_idx]):
            keyframes_with_text["keyframes"][section_idx][idx] = seek_frame_by_idx(capture, frame_index)

    fileid = os.path.basename(subpath).split(".")[0]
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
                    edge_length: int=20,
                    background_threshold: int=15,
                    minus_threshold: int=2000,
                    max_seq_len: int=200,
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
                                                              edge_length=edge_length,
                                                              background_threshold=background_threshold,
                                                              minus_threshold=minus_threshold,
                                                              max_seq_len=max_seq_len,
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
    max_section_num = 8

    # test case
    # fileid = "Nue0DINMRPM"
    # fileid = "0nKP1FdSzEg"
    # sections, keyframes = extract_keyframes(fileid, edge_length, background_threshold, minus_threshold, max_seq_len, max_section_num)
    # print("Done.")

    dest_dir = os.path.join(root_dir, data_dir, "samples")
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    # 确定需要进行分段的 ids, 以下为默认进行分段的 ids
    fileids = [filename.split(".")[0] for filename in os.listdir(os.path.join(root_dir, subtitle_dir))]

    boundary_time = float(datetime.datetime(2022, 4, 14, 18, 0, 0).strftime("%s"))
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
                                                                        edge_length,
                                                                        background_threshold,
                                                                        minus_threshold,
                                                                        max_seq_len,
                                                                        max_section_num))
        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()
        process.close()

    print("Done!")
