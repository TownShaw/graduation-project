'''
@Author: Xiao Tong
@FileName: main.py
@CreateTime: 2022-04-15 16:00:43
@Description:

'''

import os
import sys
import cv2
import json
import datetime
from tornado import web, ioloop
from predict import static_load, predict


class MainHandler(web.RequestHandler):
    def set_default_headers(self):
        # http://localhost:4000
        self.set_header("Access-Control-Allow-Origin", "http://localhost:4000")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.set_header("Access-Control-Allow-Credentials", "true")
        self.set_header('Access-Control-Allow-Methods', "POST, GET, OPTION")

    def get(self):
        return self.write(json.dumps({"datasets": list(meta_data.keys())}, ensure_ascii=False))

    def post(self):
        tmp_dir = ".tmp"
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        dataset_name = self.request.arguments["datasetName"][0].decode("utf-8")
        video_content = self.request.files["video"][0]["body"]
        sub_content = self.request.files["subtitle"][0]["body"]

        subfile = os.path.join(tmp_dir, "temp.vtt")
        videofile = os.path.join(tmp_dir, "temp.mp4")

        fvtt = open(subfile, "w")
        fvtt.write(sub_content.decode("utf-8"))
        fvideo = open(videofile, "wb")
        fvideo.write(video_content)
        fvtt.close()
        fvideo.close()

        sections, section_labels_names, video_labels_names = [], [], []
        try:
            sections, section_labels_names, video_labels_names = predict(meta_data, dataset_name, subfile, videofile)
        except Exception as e:
            print(e, file=sys.stderr)
            pass
        fps = cv2.VideoCapture(videofile).get(cv2.CAP_PROP_FPS)
        sections = [str(datetime.timedelta(seconds=int(timestamp / fps + 0.5))) for timestamp in sections]
        os.remove(subfile)
        os.remove(videofile)
        self.write(json.dumps({"sections": sections, "section_labels": section_labels_names, "video_labels": video_labels_names}, ensure_ascii=False))

    def options(self, *args):
        # no body
        # `*args` is for route with `path arguments` supports
        self.set_status(204)
        self.finish()


def make_app():
    return web.Application([
        (r"/", MainHandler)
    ])


if __name__ == "__main__":
    config_dir = "config"
    meta_data = static_load(config_dir=config_dir)
    app = make_app()
    app.listen(5000)
    print('service started, listening port 5000')
    ioloop.IOLoop.current().start()
