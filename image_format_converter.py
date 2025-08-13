import cv2
import os
from pathlib import Path

def convert_format(format_config):
    """將指定資料夾中符合格式的圖像轉換為目標格式"""
    # 直接使用 format_config，而不是 config['format_conversion']
    input_dir = Path(format_config['input_dir'])
    output_dir = Path(format_config['output_dir'])
    input_formats = format_config['input_formats']
    output_format = format_config['output_format']
    quality = format_config.get('quality', 95)
    
    # 檢查輸入資料夾是否存在
    if not input_dir.exists():
        raise FileNotFoundError(f"輸入資料夾 {input_dir} 不存在")
    
    # 確保輸出資料夾存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍歷輸入資料夾中的檔案
    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(fmt) for fmt in input_formats):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"讀取失敗: {img_path}")
                continue
            
            output_filename = os.path.splitext(filename)[0] + output_format
            output_path = os.path.join(output_dir, output_filename)
            
            if output_format.lower() in ('.jpg', '.jpeg'):
                cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            else:
                cv2.imwrite(output_path, img)
            
            print(f"已轉換: {output_path}")

if __name__ == "__main__":
    config = {
        "input_dir": "./pcba_fail",
        "output_dir": "./pcba_fail_png",
        "input_formats": [".bmp"],
        "output_format": ".png",
        "quality": 95
    }
    convert_format(config)