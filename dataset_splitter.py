import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(config):
    """將圖像和標註分割為訓練、驗證和測試集"""
    split_config = config['train_test_split']
    image_dir = Path(split_config['input']['image_dir'])
    label_dir = Path(split_config['input']['label_dir'])
    output_dir = Path(split_config['output']['output_dir'])
    train_ratio = split_config['split_ratios']['train']
    val_ratio = split_config['split_ratios']['val']
    test_ratio = split_config['split_ratios']['test']
    input_formats = split_config['input_formats']
    label_format = split_config['label_format']
    
    images = sorted([str(p) for p in image_dir.glob("*") if p.suffix.lower() in input_formats])
    labels = sorted([str(p) for p in label_dir.glob(f"*{label_format}")])
    
    print(f"圖像數量: {len(images)}, 標籤數量: {len(labels)}")
    assert len(images) == len(labels), "圖片數量和標籤數量不一致"
    
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=(val_ratio + test_ratio), random_state=42
    )
    
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=test_ratio/(val_ratio + test_ratio), random_state=42
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    def copy_files(images, labels, split):
        for img, lbl in zip(images, labels):
            shutil.copy(img, output_dir / split / "images" / Path(img).name)
            shutil.copy(lbl, output_dir / split / "labels" / Path(lbl).name)
    
    copy_files(train_images, train_labels, "train")
    copy_files(val_images, val_labels, "val")
    copy_files(test_images, test_labels, "test")
    
    print("文件已成功分配並複製到訓練、驗證和測試目錄中")