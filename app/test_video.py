

import os
import cv2


print("=" * 50)
print("测试1: OpenCV视频支持")
print("=" * 50)


fourcc_values = ['mp4v', 'XVID', 'MJPG', 'X264', 'H264']
print("支持的编码器:")
for codec in fourcc_values:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    print(f"  {codec}: {fourcc}")

print("\n" + "=" * 50)
print("测试2: 检查uploads目录")
print("=" * 50)
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
print(f"上传目录: {uploads_dir}")
print(f"目录存在: {os.path.exists(uploads_dir)}")
print(f"可写: {os.access(uploads_dir, os.W_OK)}")

print("\n" + "=" * 50)
print("测试3: 检测模型导入")
print("=" * 50)
try:
    from detector import MyDetector
    print("detector.py 导入成功")
except Exception as e:
    print(f"detector.py 导入失败: {e}")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
