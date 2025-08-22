import logging
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def split_dataset(config, log_file=None, logger=None):
    """將圖像和標註分割為訓練、驗證和測試集

    Args:
        config: 分割設定。
        log_file: 選用的 log 檔案路徑。
        logger: 可傳入外部 logger 以集中管理訊息。
    """
    logger = logger or logging.getLogger(__name__)
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    split_config = config["train_test_split"]
    image_dir = Path(split_config["input"]["image_dir"])
    label_dir = Path(split_config["input"]["label_dir"])
    output_dir = Path(split_config["output"]["output_dir"])
    train_ratio = split_config["split_ratios"]["train"]
    val_ratio = split_config["split_ratios"]["val"]
    test_ratio = split_config["split_ratios"]["test"]
    input_formats = split_config["input_formats"]
    label_format = split_config["label_format"]

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split_ratios 必須相加為 1")
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("輸入影像或標註目錄不存在")

    image_dict = {
        p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in input_formats
    }
    label_dict = {
        p.stem: p for p in label_dir.glob("*") if p.suffix.lower() == label_format
    }

    common_keys = image_dict.keys() & label_dict.keys()
    missing_images = label_dict.keys() - image_dict.keys()
    missing_labels = image_dict.keys() - label_dict.keys()

    for key in sorted(missing_images):
        logger.warning(f"找不到影像檔案 {label_dict[key]}")
    for key in sorted(missing_labels):
        logger.warning(f"找不到標籤檔案 {image_dict[key]}")

    paired_images = [image_dict[k] for k in sorted(common_keys)]
    paired_labels = [label_dict[k] for k in sorted(common_keys)]

    train_images, temp_images, train_labels, temp_labels = train_test_split(
        paired_images,
        paired_labels,
        test_size=val_ratio + test_ratio,
        random_state=42,
    )

    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images,
        temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_files(images, labels, split):
        for img, lbl in zip(images, labels):
            shutil.copy(img, output_dir / split / "images" / img.name)
            shutil.copy(lbl, output_dir / split / "labels" / lbl.name)

    copy_files(train_images, train_labels, "train")
    copy_files(val_images, val_labels, "val")
    copy_files(test_images, test_labels, "test")

    logger.info("文件已成功分配並複製到訓練、驗證和測試目錄中")

