from __future__ import annotations

from typing import Any


ALL_CLASSES: dict[int, dict[str, Any]] = {
    0: {"name": "垃圾桶", "en": "garbage_bin", "color": (50, 200, 50), "alert": False, "icon": ""},
    1: {"name": "垃圾溢出", "en": "overflow", "color": (0, 0, 255), "alert": True, "icon": ""},
    2: {"name": "散落垃圾", "en": "garbage", "color": (0, 80, 255), "alert": True, "icon": ""},
    3: {"name": "火焰", "en": "fire", "color": (0, 0, 200), "alert": True, "icon": ""},
    4: {"name": "烟雾", "en": "smoke", "color": (255, 128, 0), "alert": True, "icon": ""},
}

ALERT_ID_SET = {cid for cid, info in ALL_CLASSES.items() if info["alert"]}

BIN_TYPES = {
    "recyclable": {"name": "可回收垃圾桶", "color": "#2196F3", "classes": [0]},
    "hazardous": {"name": "有害垃圾", "color": "#F44336", "classes": [3]},
    "other": {"name": "其他垃圾", "color": "#9E9E9E", "classes": [2]},
    "overflow": {"name": "溢出告警", "color": "#FF5722", "classes": [1]},
}

VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "flv", "wmv"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

