import cv2
import numpy as np
import os
from pathlib import Path

def load_reference_images(ref_folder, input_formats):
    """讀取正常圖像資料夾並生成參考模板（平均圖像）"""
    ref_images = []
    for file in os.listdir(ref_folder):
        if any(file.lower().endswith(fmt) for fmt in input_formats):
            img_path = os.path.join(ref_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ref_images.append(img)
    
    if not ref_images:
        raise ValueError("正常圖像資料夾中沒有可用的圖像！")
    
    ref_images = [img.astype(np.float32) for img in ref_images]
    ref_avg = np.mean(ref_images, axis=0).astype(np.uint8)
    return ref_avg

def generate_anomaly_mask(ref_img, test_img, threshold=30):
    """生成異常mask：比較測試圖像與參考圖像的差異"""
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))
    
    diff = cv2.absdiff(ref_img, test_img)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def process_anomaly_detection(config):
    """處理測試圖像資料夾中的所有圖像並生成異常mask"""
    anomaly_config = config['anomaly_detection']
    ref_folder = Path(anomaly_config['reference_folder'])
    test_folder = Path(anomaly_config['test_folder'])
    output_folder = Path(anomaly_config['output_folder'])
    threshold = anomaly_config['threshold']
    input_formats = anomaly_config['input_formats']
    
    os.makedirs(output_folder, exist_ok=True)
    ref_img = load_reference_images(ref_folder, input_formats)
    
    test_images = [os.path.join(test_folder, file) for file in os.listdir(test_folder) 
                   if any(file.lower().endswith(fmt) for fmt in input_formats)]
    
    if not test_images:
        raise ValueError("測試圖像資料夾中沒有可用的圖像！")
    
    for test_path in test_images:
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        if test_img is None:
            print(f"無法讀取測試圖像：{test_path}")
            continue
        
        mask = generate_anomaly_mask(ref_img, test_img, threshold)
        output_filename = os.path.join(output_folder, f"{Path(test_path).stem}.png")
        cv2.imwrite(output_filename, mask)
        print(f"已生成mask並保存至：{output_filename}")