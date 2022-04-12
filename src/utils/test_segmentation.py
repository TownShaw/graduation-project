import os
import sys
import tqdm
import pickle

root_dir = "/share/tongxiao/graduation-project/data/samples"
total_subtitle_num = 0
exceed_max_seq_len_num = 0
segment_exceed_32 = 0
segment_exceed_32_list = []

def test_segmentation(fileid: str):
    sections = pickle.load(open(os.path.join(root_dir, fileid, fileid + ".sections.pkl"), "rb"))
    keyframes = pickle.load(open(os.path.join(root_dir, fileid, fileid + ".keyframes.pkl"), "rb"))

    # 测试最终分段个数
    assert len(sections) <= 9, "{} section number bigger than 8!".format(fileid)

    global segment_exceed_32
    global segment_exceed_32_list
    segment_num = 0
    for section_keyframes, section_subtitles in zip(*keyframes.values()):
        segment_num += len(section_subtitles)
        assert len(section_keyframes) == len(section_subtitles) + 1, "{} length of keyframes mismatch length of subtitles!".format(fileid)
        for frame in section_keyframes:
            assert frame.size == (224, 224), "{} keyframe size not equal to (224, 224)!".format(fileid)
        for subtitle in section_subtitles:
            global exceed_max_seq_len_num
            global total_subtitle_num
            if len(subtitle.split()) > 200:
                exceed_max_seq_len_num += 1
            total_subtitle_num += 1
    if segment_num > 32:
        segment_exceed_32 += 1
        segment_exceed_32_list.append((segment_num, fileid))


if __name__ == "__main__":
    for fileid in tqdm.tqdm(os.listdir(root_dir), file=sys.stdout):
        test_segmentation(fileid)
    print(exceed_max_seq_len_num, total_subtitle_num, exceed_max_seq_len_num / total_subtitle_num)
    print(segment_exceed_32)
    print(segment_exceed_32_list)
    print("Done!")
