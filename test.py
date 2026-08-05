from utils.format_converter import convert_to_yolo
# 替换为你的实际图像路径和标注
#逐行读取原始标注 → convert_to_yolo 按 1360×765 像素做归一化 → 输出 YOLO 行 + 统计。
lines = open('D:/Data/VisDrone/train/annotations/0000003_00231_d_0000016.txt').readlines()
yolo_lines, stats = convert_to_yolo(lines, 1360, 765)
print('过滤统计:', stats)
print('转换后标注:')
for l in yolo_lines[:10]: print(l)