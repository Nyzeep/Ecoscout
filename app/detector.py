

import os
import cv2
import time
import random
import base64
import numpy as np
from datetime import datetime


ALL_CLASSES = {
    0: {"name": "垃圾桶",   "en": "garbage_bin", "color": (50, 200, 50),  "alert": False, "icon": ""},
    1: {"name": "垃圾溢出", "en": "overflow",    "color": (0,   0,   255), "alert": True,  "icon": ""},
    2: {"name": "散落垃圾", "en": "garbage",     "color": (0,   80,  255), "alert": True,  "icon": ""},
    3: {"name": "火焰",     "en": "fire",        "color": (0,   0,   200), "alert": True,  "icon": ""},
    4: {"name": "烟雾",     "en": "smoke",       "color": (255, 128, 0),  "alert": True,  "icon": ""},
}


ALERT_ID_SET = {cid for cid, info in ALL_CLASSES.items() if info["alert"]}


BIN_TYPES = {
    "recyclable": {"name": "可回收垃圾桶", "color": "#2196F3", "classes": [0]},
    "hazardous":  {"name": "有害垃圾桶",   "color": "#F44336", "classes": [3]},
    "other":      {"name": "其他垃圾桶",   "color": "#9E9E9E", "classes": [2]},
    "overflow":   {"name": "溢出警报",     "color": "#FF5722", "classes": [1]},
}


class MyDetector:


    def __init__(self,
                 garbage_model_path=None,
                 fire_model_path=None,
                 smoke_model_path=None,
                 conf_threshold=0.5,
                 iou_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold


        self.garbage_model = None
        self.fire_model    = None
        self.smoke_model   = None


        self.models_loaded = {
            "garbage": False,
            "fire":    False,
            "smoke":   False,
        }

        self._load_models(garbage_model_path, fire_model_path, smoke_model_path)

    def _load_models(self, garbage_path, fire_path, smoke_path):

        try:
            from ultralytics import YOLO


            if garbage_path and os.path.exists(garbage_path):
                try:
                    self.garbage_model = YOLO(garbage_path)
                    self.models_loaded["garbage"] = True
                    print("[检测模块] 垃圾分类模型加载完成")
                except Exception as err:
                    print("[检测模块] 垃圾分类模型加载失败:", err)


            if fire_path and os.path.exists(fire_path):
                try:
                    self.fire_model = YOLO(fire_path)
                    self.models_loaded["fire"] = True
                    print("[检测模块] 火焰检测模型加载完成")
                except Exception as err:
                    print("[检测模块] 火焰检测模型加载失败:", err)


            if smoke_path and os.path.exists(smoke_path):
                try:
                    self.smoke_model = YOLO(smoke_path)
                    self.models_loaded["smoke"] = True
                    print("[检测模块] 烟雾检测模型加载完成")
                except Exception as err:
                    print("[检测模块] 烟雾检测模型加载失败:", err)

            if not any(self.models_loaded.values()):
                print("[检测模块] 没有找到模型文件，使用演示模式")

        except ImportError:
            print("[检测模块] ultralytics 未安装，使用演示模式")
            print("  安装命令: pip install ultralytics")

    def detect(self, img):


        if any(self.models_loaded.values()):
            return self._run_yolo(img)
        else:
            return self._fake_detect(img)

    def _run_yolo(self, img):

        result_list = []
        h, w = img.shape[:2]
        min_box_area = w * h * 0.005


        if self.models_loaded["garbage"] and self.garbage_model is not None:
            yolo_out = self.garbage_model(img, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            for box in yolo_out.boxes:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue
                info = ALL_CLASSES.get(cid, {"name": "未知目标", "alert": False, "color": (128, 128, 128), "icon": ""})
                result_list.append({
                    "class_id":     cid,
                    "class_name":   info["name"],
                    "confidence":   round(conf, 3),
                    "bbox":         [x1, y1, x2, y2],
                    "alert":        info["alert"],
                    "color":        info["color"],
                    "icon":         info.get("icon", ""),
                    "source_model": "garbage",
                })


        if self.models_loaded["fire"] and self.fire_model is not None:
            yolo_out = self.fire_model(img, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            for box in yolo_out.boxes:
                cid = int(box.cls[0])
                if cid != 0:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue
                info = ALL_CLASSES[3]
                result_list.append({
                    "class_id":     3,
                    "class_name":   info["name"],
                    "confidence":   round(conf, 3),
                    "bbox":         [x1, y1, x2, y2],
                    "alert":        info["alert"],
                    "color":        info["color"],
                    "icon":         info.get("icon", ""),
                    "source_model": "fire",
                })


        if self.models_loaded["smoke"] and self.smoke_model is not None:
            yolo_out = self.smoke_model(img, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            for box in yolo_out.boxes:
                cid = int(box.cls[0])
                if cid != 0:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue
                info = ALL_CLASSES[4]
                result_list.append({
                    "class_id":     4,
                    "class_name":   info["name"],
                    "confidence":   round(conf, 3),
                    "bbox":         [x1, y1, x2, y2],
                    "alert":        info["alert"],
                    "color":        info["color"],
                    "icon":         info.get("icon", ""),
                    "source_model": "smoke",
                })

        return result_list

    def _fake_detect(self, img):

        h, w = img.shape[:2]
        result_list = []
        seed_val = int(img[h // 2, w // 2, 0]) if img.ndim == 3 else 0
        my_rng   = random.Random(seed_val + int(time.time() // 3))

        all_cids   = [0, 1, 2, 3, 4]
        num_objs   = my_rng.randint(1, 4)
        chosen_cids = my_rng.sample(all_cids, min(num_objs, len(all_cids)))

        for cid in chosen_cids:
            info   = ALL_CLASSES[cid]
            margin = 30
            x1     = my_rng.randint(margin, w // 2)
            y1     = my_rng.randint(margin, h // 2)
            x2     = my_rng.randint(w // 2, w - margin)
            y2     = my_rng.randint(h // 2, h - margin)
            conf   = round(my_rng.uniform(0.55, 0.96), 3)
            result_list.append({
                "class_id":     cid,
                "class_name":   info["name"],
                "confidence":   conf,
                "bbox":         [x1, y1, x2, y2],
                "alert":        info["alert"],
                "color":        info["color"],
                "icon":         info.get("icon", ""),
                "source_model": "demo",
            })
        return result_list

    def draw_boxes(self, img, det_list):

        output_img = img.copy()

        en_label_map = {
            "垃圾桶":   "GarbageBin",
            "垃圾溢出": "Overflow",
            "散落垃圾": "Garbage",
            "火焰":     "FIRE",
            "烟雾":     "SMOKE",
            "未知目标": "Unknown",
        }
        for det in det_list:
            x1, y1, x2, y2 = det["bbox"]
            box_color = det.get("color", (0, 255, 0))
            en_name   = en_label_map.get(det["class_name"], det["class_name"])
            label_str = "{} {:.0%}".format(en_name, det["confidence"])
            line_w    = 3 if det["alert"] else 2

            cv2.rectangle(output_img, (x1, y1), (x2, y2), box_color, line_w)
            (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output_img, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), box_color, -1)
            cv2.putText(output_img, label_str, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if det["alert"]:
                cv2.putText(output_img, "! ALERT", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return output_img

    def check_scene(self, det_list):


        alarm_list  = [d for d in det_list if d["alert"]]
        alarm_types = list({d["class_name"] for d in alarm_list})
        cid_set     = {d["class_id"] for d in det_list}

        scene_status = "normal"
        if 3 in cid_set:
            scene_status = "fire"
        elif 4 in cid_set:
            scene_status = "smoke"
        elif 1 in cid_set:
            scene_status = "overflow"
        elif alarm_list:
            scene_status = "warning"

        return {
            "status":       scene_status,
            "alert_count":  len(alarm_list),
            "alert_types":  alarm_types,
            "normal_count": len([d for d in det_list if not d["alert"]]),
            "total":        len(det_list),
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


    def draw_results(self, img, det_list):
        return self.draw_boxes(img, det_list)

    def analyze_scene(self, det_list):
        return self.check_scene(det_list)


UnifiedDetector = MyDetector
GarbageDetector = MyDetector
GARBAGE_CLASSES = ALL_CLASSES


def frame_to_base64(img):

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def base64_to_frame(b64_str):

    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
