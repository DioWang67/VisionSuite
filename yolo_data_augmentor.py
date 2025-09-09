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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
                logging.FileHandler('augmentation_fixed.log'),
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
            
            # 創建驗證目錄
            debug_dir = Path(self.config['output'].get('debug_dir', 'debug_visualizations'))
            debug_dir.mkdir(parents=True, exist_ok=True)
            
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
        """針對小目標優化的默認配置"""
        return {
            'input': {
                'image_dir': 'A/img',
                'label_dir': 'A/label'
            },
            'output': {
                'image_dir': 'output/images',
                'label_dir': 'output/labels',
                'debug_dir': 'debug_visualizations'  # 新增調試目錄
            },
            'augmentation': {
                'num_images': 5,
                'target_size': 640,  # 針對小目標使用更大解析度
                'operations': {
                    'flip': {'probability': 0.3},
                    'rotate': {'angle': (-5, 5)},  # 減小旋轉角度避免小目標變形
                    'multiply': {'range': (0.9, 1.1)},  # 減小亮度變化
                    'scale': {'range': (0.95, 1.05)},  # 減小縮放範圍
                    'contrast': {'range': (0.9, 1.3)},
                    'hue': {'range': (-5, 5)},  # 極小的色調變化保護綠色
                    'noise': {'scale': (0, 0.02)},  # 輕微噪聲
                    'perspective': {'scale': (0, 0)},  # 關閉透視變換
                    'blur': {'kernel': (0, 0)}  # 關閉模糊
                }
            },
            'processing': {
                'batch_size': 10,
                'num_workers': None,
                'debug_mode': True  # 開啟調試模式
            }
        }

    def yolo_to_absolute(self, yolo_bbox, img_width, img_height):
        """將YOLO格式轉換為絕對座標"""
        x_center, y_center, width, height = yolo_bbox
        
        # 轉換為絕對座標
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = width * img_width
        abs_height = height * img_height
        
        # 轉換為左上角座標
        x1 = abs_x_center - abs_width / 2
        y1 = abs_y_center - abs_height / 2
        x2 = abs_x_center + abs_width / 2
        y2 = abs_y_center + abs_height / 2
        
        return [x1, y1, x2, y2]

    def absolute_to_yolo(self, abs_bbox, img_width, img_height):
        """將絕對座標轉換為YOLO格式"""
        x1, y1, x2, y2 = abs_bbox
        
        # 計算中心點和尺寸
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # 轉換為相對座標
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # 確保座標在有效範圍內
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        
        return [x_center, y_center, width, height]

    def visualize_bboxes(self, image, bboxes, class_labels, save_path=None, title=""):
        """可視化邊界框以供調試"""
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        ax.set_title(title)
        
        img_height, img_width = image.shape[:2]
        
        for bbox, label in zip(bboxes, class_labels):
            if len(bbox) == 4 and all(0 <= coord <= 1 for coord in bbox):  # YOLO格式
                x_center, y_center, width, height = bbox
                
                # 轉換為像素座標
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height
                
                # 計算左上角座標
                x1 = x_center_px - width_px / 2
                y1 = y_center_px - height_px / 2
                
                # 創建矩形
                rect = patches.Rectangle((x1, y1), width_px, height_px,
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # 添加標籤
                ax.text(x1, y1 - 5, f'Class {int(label)}', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _create_augmentations(self) -> A.Compose:
        """創建針對小目標優化的增強序列"""
        aug_config = self.config['augmentation']
        ops_config = aug_config['operations']
        
        aug_list = []

        # 幾何變換（最容易導致標註偏移的操作）
        if ops_config.get('flip', {}).get('probability', 0) > 0:
            aug_list.append(A.HorizontalFlip(p=ops_config['flip']['probability']))
        
        if ops_config.get('rotate') and ops_config['rotate']['angle'] != (0, 0):
            angle = ops_config['rotate']['angle']
            if isinstance(angle, (list, tuple)):
                limit = max(abs(angle[0]), abs(angle[1]))
            else:
                limit = abs(angle)
            aug_list.append(A.Rotate(limit=limit, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=128))

        # 顏色變換（不影響座標）
        if ops_config.get('multiply'):
            multiply_range = ops_config['multiply']['range']
            brightness_limit = (multiply_range[0] - 1, multiply_range[1] - 1)
            aug_list.append(A.RandomBrightnessContrast(
                brightness_limit=brightness_limit, contrast_limit=0, p=0.5))
        
        if ops_config.get('contrast'):
            contrast_range = ops_config['contrast']['range']
            contrast_limit = (contrast_range[0] - 1, contrast_range[1] - 1)
            aug_list.append(A.RandomBrightnessContrast(
                brightness_limit=0, contrast_limit=contrast_limit, p=0.5))
        
        if ops_config.get('hue'):
            hue_range = ops_config['hue']['range']
            if isinstance(hue_range, (list, tuple)):
                hue_limit = hue_range
            else:
                hue_limit = (-abs(hue_range), abs(hue_range))
            aug_list.append(A.HueSaturationValue(
                hue_shift_limit=hue_limit, sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10), p=0.3))
        
        # 噪聲（不影響座標）
        if ops_config.get('noise') and ops_config['noise']['scale'][1] > 0:
            noise_scale = ops_config['noise']['scale']
            aug_list.append(A.GaussNoise(var_limit=(0, noise_scale[1] * 255), p=0.2))

        self.logger.info(f"增強管道包含 {len(aug_list)} 個操作: {[type(aug).__name__ for aug in aug_list]}")

        return A.Compose(
            aug_list,
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
        )

    def resize_with_padding(self, image, target_size=640):
        """
        等比例縮放並填充到目標尺寸，返回變換參數
        """
        h, w = image.shape[:2]
        
        # 計算縮放比例
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 縮放圖片
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 創建目標尺寸的圖片（灰色背景）
        result = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
        
        # 計算居中位置
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # 將縮放後的圖片放置到中心
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 返回變換參數
        transform_params = {
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': y_offset,
            'new_width': new_w,
            'new_height': new_h,
            'target_size': target_size
        }
        
        return result, transform_params

    def transform_bboxes_after_resize(self, bboxes, original_width, original_height, transform_params):
        """
        在縮放和填充後變換邊界框座標
        """
        transformed_bboxes = []
        
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            
            # 轉換為原圖的絕對座標
            abs_x_center = x_center * original_width
            abs_y_center = y_center * original_height
            abs_width = width * original_width
            abs_height = height * original_height
            
            # 應用縮放
            scaled_x_center = abs_x_center * transform_params['scale']
            scaled_y_center = abs_y_center * transform_params['scale']
            scaled_width = abs_width * transform_params['scale']
            scaled_height = abs_height * transform_params['scale']
            
            # 應用偏移
            final_x_center = scaled_x_center + transform_params['x_offset']
            final_y_center = scaled_y_center + transform_params['y_offset']
            
            # 轉換為新圖的相對座標
            target_size = transform_params['target_size']
            new_x_center = final_x_center / target_size
            new_y_center = final_y_center / target_size
            new_width = scaled_width / target_size
            new_height = scaled_height / target_size
            
            # 確保座標在有效範圍內
            new_x_center = np.clip(new_x_center, 0, 1)
            new_y_center = np.clip(new_y_center, 0, 1)
            new_width = np.clip(new_width, 0, 1)
            new_height = np.clip(new_height, 0, 1)
            
            transformed_bboxes.append([new_x_center, new_y_center, new_width, new_height])
        
        return transformed_bboxes

    def _process_single_image(self, img_file: str) -> None:
        """處理單張圖片的增強（修正版）"""
        try:
            img_path = Path(self.config['input']['image_dir']) / img_file
            label_path = Path(self.config['input']['label_dir']) / (Path(img_file).stem + '.txt')

            # 讀取圖片
            image = cv2.imread(str(img_path))
            if image is None:
                self.logger.error(f"無法讀取圖片: {img_path}")
                return

            # 轉換為RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            
            self.logger.info(f"處理圖片 {img_file}, 原始尺寸: {original_width}x{original_height}")

            # 讀取標註
            try:
                with open(label_path, 'r') as f:
                    annotations = [line.strip().split() for line in f.readlines() if line.strip()]
            except Exception as e:
                self.logger.error(f"讀取標註檔案失敗 {label_path}: {e}")
                return

            if not annotations:
                self.logger.warning(f"標註文件為空: {label_path}")
                return

            # 解析標註
            bboxes = []
            class_labels = []
            for ann in annotations:
                if len(ann) >= 5:
                    cls, x_center, y_center, width, height = map(float, ann[:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(int(cls))

            if not bboxes:
                self.logger.warning(f"沒有有效的標註: {label_path}")
                return

            # 調試模式：保存原始圖片和標註
            if self.config['processing'].get('debug_mode', False):
                debug_dir = Path(self.config['output'].get('debug_dir', 'debug_visualizations'))
                original_viz_path = debug_dir / f"{Path(img_file).stem}_original.png"
                self.visualize_bboxes(image, bboxes, class_labels, 
                                    str(original_viz_path), f"Original: {img_file}")

            # 生成增強版本
            target_size = self.config['augmentation'].get('target_size', 640)
            
            for i in range(self.config['augmentation']['num_images']):
                try:
                    # 第一步：應用 Albumentations 增強（保持原始尺寸）
                    if len(self.augmentations.transforms) > 0:
                        transformed = self.augmentations(
                            image=image,
                            bboxes=bboxes,
                            class_labels=class_labels
                        )
                        augmented_image = transformed['image']
                        augmented_bboxes = transformed['bboxes']
                        augmented_labels = transformed['class_labels']
                    else:
                        augmented_image = image.copy()
                        augmented_bboxes = bboxes.copy()
                        augmented_labels = class_labels.copy()

                    # 檢查增強後是否還有標註
                    if len(augmented_bboxes) == 0:
                        self.logger.warning(f"增強後沒有剩餘標註，跳過: {img_file}_aug_{i+1}")
                        continue

                    # 第二步：縮放和填充
                    final_image, transform_params = self.resize_with_padding(augmented_image, target_size)
                    
                    # 第三步：變換邊界框座標
                    final_bboxes = self.transform_bboxes_after_resize(
                        augmented_bboxes, original_width, original_height, transform_params)

                    # 驗證最終的邊界框
                    valid_bboxes = []
                    valid_labels = []
                    for bbox, label in zip(final_bboxes, augmented_labels):
                        # 檢查邊界框是否有效（面積 > 0，座標在合理範圍內）
                        if (bbox[2] > 0.001 and bbox[3] > 0.001 and  # 寬高大於最小值
                            0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1):  # 中心點在圖片內
                            valid_bboxes.append(bbox)
                            valid_labels.append(label)

                    if len(valid_bboxes) == 0:
                        self.logger.warning(f"沒有有效的最終邊界框，跳過: {img_file}_aug_{i+1}")
                        continue

                    # 調試模式：保存增強後的可視化
                    if self.config['processing'].get('debug_mode', False):
                        debug_viz_path = debug_dir / f"{Path(img_file).stem}_aug_{i+1}.png"
                        self.visualize_bboxes(final_image, valid_bboxes, valid_labels,
                                            str(debug_viz_path), f"Augmented: {img_file}_aug_{i+1}")

                    # 保存增強後的圖片
                    aug_img_filename = f"{Path(img_file).stem}_aug_{i + 1}.png"
                    aug_img_path = Path(self.config['output']['image_dir']) / aug_img_filename
                    final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_path), final_image_bgr)

                    # 保存增強後的標註
                    aug_label_filename = f"{Path(img_file).stem}_aug_{i + 1}.txt"
                    aug_label_path = Path(self.config['output']['label_dir']) / aug_label_filename
                    
                    with open(aug_label_path, 'w') as f:
                        for bbox, label in zip(valid_bboxes, valid_labels):
                            f.write(f"{int(label)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                    self.logger.info(f"成功生成: {aug_img_filename} (包含 {len(valid_bboxes)} 個標註)")

                except Exception as e:
                    self.logger.error(f"處理增強時出錯 {img_file}_aug_{i+1}: {e}")
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

        # 先處理一張圖片測試
        if self.config['processing'].get('debug_mode', False):
            self.logger.info("調試模式：先處理第一張圖片")
            self._process_single_image(img_files[0])
            self.logger.info("請檢查 debug_visualizations 目錄中的結果，確認無誤後再處理全部圖片")
            return

        # 正常批次處理
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
    # 使用修正版的增強器
    augmentor = FixedDataAugmentor()
    augmentor.process_dataset()