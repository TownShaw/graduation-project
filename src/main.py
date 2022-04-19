'''
@Author: Xiao Tong
@FileName: main.py
@CreateTime: 2022-04-15 16:00:43
@Description:

'''

import os
import json
from tornado import web, ioloop
from predict import static_load, predict
from requests_toolbelt.multipart import decoder


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

        req_data = self.request.body
        content_type = self.request.headers._dict["Content-Type"]
        multipart_data = decoder.MultipartDecoder(req_data, content_type)
        dataset_name = "Khan"
        subfile = os.path.join(tmp_dir, "temp.vtt")
        videofile = os.path.join(tmp_dir, "temp.mp4")
        for part in multipart_data.parts:
            content_type = ""
            for k, v in part.headers.items():
                if k.decode("utf-8") == "Content-Type":
                    content_type = v.decode("utf-8")
            content_type = content_type.split("/")[0]
            if content_type != "text":
                fvideo = open(videofile, "wb")
                fvideo.write(part.content)
                fvideo.close()
            else:
                fvtt = open(subfile, "w")
                fvtt.write(part.content.decode("utf-8"))
                fvtt.close()
        sections, section_labels_names, video_labels_names = [], [], []
        try:
            sections, section_labels_names, video_labels_names = predict(meta_data, dataset_name, subfile, videofile)
        except Exception:
            pass
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
