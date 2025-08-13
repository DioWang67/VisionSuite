import cv2
import os

# 設定你的來源目錄
input_dir = 'target\pcba_fail'
# 設定轉換後要放的目錄
output_dir = "target\pcba_fail"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".bmp"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"讀取失敗: {img_path}")
            continue

        # 轉成 jpg 的路徑
        jpg_filename = os.path.splitext(filename)[0] + ".png"
        jpg_path = os.path.join(output_dir, jpg_filename)

        # 儲存 jpg
        cv2.imwrite(jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"已轉換: {jpg_path}")
