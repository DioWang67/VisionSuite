import cv2
import albumentations as A
import os
from multiprocessing import Pool, cpu_count
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

class DataAugmentor:
    def __init__(self, config_path: str = None):
        """初始化數據增強器"""
        self._setup_logging()
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self._setup_output_dirs()
        self.augmentations = self._create_augmentations()

    def _setup_logging(self):
        """設置日誌系統"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('augmentation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_output_dirs(self):
        """建立必要的輸出目錄"""
        try:
            output_img_dir = Path(self.config['output']['image_dir'])
            output_img_dir.mkdir(parents=True, exist_ok=True)
            
            output_label_dir = Path(self.config['output']['label_dir'])
            output_label_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"創建輸出目錄成功: {output_img_dir}, {output_label_dir}")
        except Exception as e:
            self.logger.error(f"創建輸出目錄失敗: {e}")
            raise

    def _load_config(self, config_path: str) -> dict:
        """從YAML文件加載配置"""
        try:
            if config_path and Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"成功從 {config_path} 加載配置")
                return config
            else:
                self.logger.warning(f"配置文件 {config_path} 不存在，使用默認配置")
                return self._default_config()
        except Exception as e:
            self.logger.error(f"加載配置文件失敗: {e}")
            return self._default_config()

    def _default_config(self) -> dict:
        """返回默認配置"""
        return {
            'input': {
                'image_dir': 'A/img',
                'label_dir': 'A/label'
            },
            'output': {
                'image_dir': 'output/images',
                'label_dir': 'output/labels'
            },
            'augmentation': {
                'num_images': 5,
                'num_operations': (3, 5),
                'operations': {
                    'flip': {'probability': 0.5},
                    'rotate': {'angle': (-10, 10)},
                    'multiply': {'range': (0.8, 1.2)},
                    'scale': {'range': (0.8, 1.2)},
                    'contrast': {'range': (0.75, 1.5)},
                    'hue': {'range': (-10, 10)},
                    'noise': {'scale': (0, 0.05)},
                    'perspective': {'scale': (0.01, 0.05)},
                    'blur': {'kernel': (3, 5)}
                }
            },
            'processing': {
                'batch_size': 10,
                'num_workers': None
            }
        }

    def _resize_with_aspect_ratio_and_gray_padding(self, image, target_size=640):
        """
        將圖片等比例縮放並用灰色填充至目標尺寸
        """
        h, w = image.shape[:2]
        
        # 計算縮放比例
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 縮放圖片
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 創建灰色背景
        padded = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
        
        # 計算居中位置
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # 將縮放後的圖片放在灰色背景上
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 計算座標變換係數
        scale_x = new_w / w
        scale_y = new_h / h
        offset_x = x_offset / target_size
        offset_y = y_offset / target_size
        
        return padded, scale_x, scale_y, offset_x, offset_y

    def _transform_yolo_bbox(self, bbox, scale_x, scale_y, offset_x, offset_y):
        """
        轉換YOLO格式的邊界框座標
        """
        x_center, y_center, width, height = bbox
        
        # 轉換中心點
        new_x_center = x_center * scale_x + offset_x
        new_y_center = y_center * scale_y + offset_y
        
        # 轉換寬高
        new_width = width * scale_x
        new_height = height * scale_y
        
        # 確保座標在有效範圍內
        new_x_center = np.clip(new_x_center, 0, 1)
        new_y_center = np.clip(new_y_center, 0, 1)
        new_width = np.clip(new_width, 0, 1)
        new_height = np.clip(new_height, 0, 1)
        
        return [new_x_center, new_y_center, new_width, new_height]

    def _create_augmentations(self) -> A.Compose:
        """根據配置創建增強序列（不包含縮放，因為我們會手動處理）"""
        aug_config = self.config['augmentation']
        ops_config = aug_config['operations']
        
        aug_list = []

        # 移除自動縮放，改為手動處理
        # aug_list.append(A.LongestMaxSize(max_size=640, interpolation=cv2.INTER_AREA, always_apply=True))

        # 建立增強操作列表
        if ops_config.get('flip'):
            aug_list.append(A.HorizontalFlip(p=ops_config['flip']['probability']))
        
        if ops_config.get('rotate'):
            angle = ops_config['rotate']['angle']
            limit = angle if isinstance(angle, (int, float)) else (angle[0], angle[1])
            # 移除不支持的參數
            aug_list.append(A.Rotate(limit=limit, border_mode=cv2.BORDER_CONSTANT, p=0.5))
        
        if ops_config.get('multiply'):
            multiply_range = ops_config['multiply']['range']
            aug_list.append(A.RandomBrightnessContrast(
                brightness_limit=(multiply_range[0]-1, multiply_range[1]-1), p=0.5))
        
        if ops_config.get('scale'):
            scale_range = ops_config['scale']['range']
            # 移除不支持的參數
            aug_list.append(A.RandomScale(
                scale_limit=(scale_range[0]-1, scale_range[1]-1), p=0.5))
        
        if ops_config.get('contrast'):
            contrast_range = ops_config['contrast']['range']
            aug_list.append(A.RandomBrightnessContrast(
                contrast_limit=(contrast_range[0]-1, contrast_range[1]-1), p=0.5))
        
        if ops_config.get('hue'):
            hue_range = ops_config['hue']['range']
            if isinstance(hue_range, (int, float)):
                hue_limit = (-hue_range, hue_range)
            else:
                hue_limit = hue_range
            aug_list.append(A.HueSaturationValue(
                hue_shift_limit=hue_limit, sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20), p=0.5))
        
        if ops_config.get('noise'):
            noise_scale = ops_config['noise']['scale']
            # 移除不支持的參數
            aug_list.append(A.GaussNoise(mean=0, var_limit=(0, noise_scale[1]*255), p=0.0))
        
        if ops_config.get('perspective'):
            perspective_scale = ops_config['perspective']['scale']
            # 移除不支持的參數
            aug_list.append(A.Perspective(scale=perspective_scale[1], p=0.5))
        
        if ops_config.get('blur'):
            blur_kernel = ops_config['blur']['kernel']
            blur_limit = blur_kernel if isinstance(blur_kernel, int) else blur_kernel
            aug_list.append(A.MotionBlur(blur_limit=blur_limit, p=0))

        # 移除自動填充，改為手動處理
        # aug_list.append(A.PadIfNeeded(...))

        # 記錄增強管道以供調試
        self.logger.info(f"增強管道: {aug_list}")

        return A.Compose(
            aug_list,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    def _process_single_image(self, img_file: str) -> None:
        """處理單張圖片的增強"""
        try:
            img_path = Path(self.config['input']['image_dir']) / img_file
            label_path = Path(self.config['input']['label_dir']) / (Path(img_file).stem + '.txt')

            # 讀取圖片並轉為 RGB
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                self.logger.error(f"無法讀取圖片: {img_path}")
                return

            # 確保圖片為 RGB 三通道
            if len(image.shape) == 2:  # 灰階圖
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA 圖片
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:  # BGR 轉 RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 記錄原始圖片尺寸
            self.logger.info(f"圖片 {img_file} 原始尺寸: {image.shape}")

            # 讀取標註
            try:
                with open(label_path, 'r') as f:
                    annotations = [line.strip().split() for line in f.readlines()]
            except Exception as e:
                self.logger.error(f"讀取標註檔案失敗 {label_path}: {e}")
                return

            # 解析標註
            bboxes = []
            class_labels = []
            for ann in annotations:
                cls, x_center, y_center, width, height = map(float, ann)
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(int(cls))

            # 在批次處理增強時加入標註數量檢查
            for i in range(self.config['augmentation']['num_images']):
                try:
                    # 先進行數據增強（在原始尺寸上）
                    if len(aug_list := self.augmentations.transforms) > 0:
                        transformed = self.augmentations(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )
                        augmented_image = transformed['image']
                        augmented_bboxes = transformed['bboxes']
                        augmented_labels = transformed['class_labels']
                    else:
                        # 如果沒有增強操作，直接使用原圖
                        augmented_image = image.copy()
                        augmented_bboxes = bboxes.copy()
                        augmented_labels = class_labels.copy()

                    # 檢查邊界框數量是否匹配
                    if len(augmented_bboxes) != len(bboxes):
                        self.logger.warning(f"增強後邊界框數量與原始標註數量不一致，跳過此增強: {img_file}")
                        continue

                    # 手動進行等比例縮放和灰色填充
                    final_image, scale_x, scale_y, offset_x, offset_y = self._resize_with_aspect_ratio_and_gray_padding(
                        augmented_image, 640)

                    # 轉換邊界框座標
                    final_bboxes = []
                    for bbox in augmented_bboxes:
                        transformed_bbox = self._transform_yolo_bbox(bbox, scale_x, scale_y, offset_x, offset_y)
                        final_bboxes.append(transformed_bbox)

                    # 檢查最終圖片尺寸
                    if final_image.shape[:2] != (640, 640):
                        self.logger.warning(f"圖片 {img_file}_aug_{i + 1} 尺寸不正確，實際尺寸: {final_image.shape[:2]}")

                    # 檢查邊框顏色（多個角落）
                    border_pixels = [
                        final_image[0, 0],  # 左上角
                        final_image[0, -1],  # 右上角
                        final_image[-1, 0],  # 左下角
                        final_image[-1, -1]  # 右下角
                    ]
                    expected_border = [128, 128, 128]
                    all_borders_correct = True
                    for idx, pixel in enumerate(border_pixels):
                        if not np.allclose(pixel, expected_border, atol=1):
                            self.logger.warning(f"圖片 {img_file}_aug_{i + 1} 的邊框（位置 {idx}）未正確填充為灰色，實際值為 {pixel.tolist()}")
                            all_borders_correct = False

                    if all_borders_correct:
                        self.logger.info(f"圖片 {img_file}_aug_{i + 1} 邊框填充正確為灰色")

                    # 保存增強後的圖片
                    aug_img_filename = f"{Path(img_file).stem}_aug_{i + 1}.png"
                    aug_img_path = Path(self.config['output']['image_dir']) / aug_img_filename
                    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_path), final_image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    self.logger.info(f"生成增強圖片: {aug_img_path}")

                    # 保存增強後的標註
                    aug_label_filename = f"{Path(img_file).stem}_aug_{i + 1}.txt"
                    aug_label_path = Path(self.config['output']['label_dir']) / aug_label_filename
                    
                    with open(aug_label_path, 'w') as f:
                        for bbox, label in zip(final_bboxes, augmented_labels):
                            x_center = min(max(bbox[0], 0), 1)
                            y_center = min(max(bbox[1], 0), 1)
                            width = min(max(bbox[2], 0), 1)
                            height = min(max(bbox[3], 0), 1)
                            f.write(f"{int(label)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    self.logger.info(f"生成增強標註: {aug_label_path}")

                except Exception as e:
                    self.logger.error(f"處理增強時出錯 {img_file}, 索引 {i}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"處理圖片時出錯 {img_file}: {e}")

    def process_dataset(self):
        """處理整個數據集"""
        start_time = time.time()
        
        input_img_dir = Path(self.config['input']['image_dir'])
        if not input_img_dir.exists():
            self.logger.error(f"輸入圖片目錄不存在: {input_img_dir}")
            return
            
        img_files = [f for f in os.listdir(input_img_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        
        if not img_files:
            self.logger.error("沒有找到圖片文件")
            return

        self.logger.info(f"開始處理 {len(img_files)} 張圖片")

        num_workers = self.config['processing'].get('num_workers', None) or cpu_count()
        
        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap(self._process_single_image, img_files),
                total=len(img_files),
                desc="處理進度"
            ))

        elapsed_time = time.time() - start_time
        self.logger.info(f"處理完成! 耗時: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    augmentor = DataAugmentor()
    augmentor.process_dataset()