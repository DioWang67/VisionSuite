import cv2
import numpy as np
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

# ---------------------------- 基本設定 ----------------------------

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

@dataclass(frozen=True)
class AnomalyConfig:
    reference_folder: Path
    test_folder: Path
    output_folder: Path
    # 原本的 threshold（絕對差）仍保留，改為 abs_diff_thresh
    abs_diff_thresh: int = 30
    # 新增：Z-score 門檻（任一達標即算異常）
    z_thresh: float = 3.0
    # 對齊模式：'auto'（先 ECC 失敗再 ORB）、'ecc'、'orb'、'none'
    align: str = "auto"
    # 形態學與濾波參數
    open_ksize: int = 3
    close_ksize: int = 5
    dilate_ksize: int = 0
    min_area: int = 50
    # 其他
    input_formats: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    recursive: bool = True
    save_overlay: bool = False
    overlay_alpha: float = 0.6
    quiet: bool = False

# ---------------------------- 輔助工具 ----------------------------

def _setup_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

def _norm_exts(exts: Iterable[str]) -> List[str]:
    # 接受有無點的寫法（jpg / .jpg）
    out = []
    for e in exts:
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        out.append(e)
    return out

def _list_images(folder: Path, exts: Iterable[str], recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in folder.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    else:
        files = [folder / f for f in os.listdir(folder) if (folder / f).suffix.lower() in exts]
        files = [p for p in files if p.is_file()]
    files.sort()
    return files

def _imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"讀取圖像失敗：{path}")
    return img

def _to_bgr(img_gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# ---------------------------- 對齊邏輯 ----------------------------

def _align_ecc(src_gray: np.ndarray, ref_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref32 = ref_gray.astype(np.float32) / 255.0
    src32 = src_gray.astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
    try:
        _cc, warp = cv2.findTransformECC(
            ref32, src32, warp, motionType=cv2.MOTION_AFFINE, criteria=criteria, gaussFiltSize=5
        )
    except cv2.error as e:
        raise RuntimeError(f"ECC 對齊失敗: {e}")
    aligned = cv2.warpAffine(src_gray, warp, (ref_gray.shape[1], ref_gray.shape[0]), flags=cv2.INTER_LINEAR)
    return aligned, warp

def _align_orb(src_gray: np.ndarray, ref_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(ref_gray, None)
    if des1 is None or des2 is None:
        raise RuntimeError("ORB 無法找到足夠特徵")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        raise RuntimeError("ORB 配對不足")
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("找不到同質矩陣")
    aligned = cv2.warpPerspective(src_gray, H, (ref_gray.shape[1], ref_gray.shape[0]), flags=cv2.INTER_LINEAR)
    return aligned, H

def _align_image(src_gray: np.ndarray, ref_gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        # 只在大小不同時做 resize，避免無意義的形變
        if src_gray.shape != ref_gray.shape:
            return cv2.resize(src_gray, (ref_gray.shape[1], ref_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        return src_gray
    # 先試 ECC，再退 ORB（auto）；或強制某一種
    if mode in ("ecc", "auto"):
        try:
            aligned, _ = _align_ecc(src_gray, ref_gray)
            return aligned
        except Exception:
            if mode == "ecc":
                raise
    aligned, _ = _align_orb(src_gray, ref_gray)
    return aligned

# ---------------------------- 正常模型（mean/std） ----------------------------

class NormalModel:
    """逐像素 mean/std（float32），ref_gray 是模板大小。"""
    def __init__(self, ref_gray: np.ndarray, mean: np.ndarray, std: np.ndarray, n: int) -> None:
        self.ref_gray = ref_gray
        self.mean = mean
        self.std = std
        self.n = n

def _build_normal_model(ref_folder: Path, exts: Iterable[str], recursive: bool, align: str) -> NormalModel:
    paths = _list_images(ref_folder, exts, recursive)
    if not paths:
        raise ValueError("正常圖像資料夾中沒有可用的圖像！")
    # 第一張當模板
    ref0 = _imread_gray(paths[0])
    ref_gray = ref0
    mean = np.zeros_like(ref_gray, dtype=np.float32)
    m2 = np.zeros_like(ref_gray, dtype=np.float32)
    n = 0

    for i, p in enumerate(paths):
        g = _imread_gray(p)
        if i > 0:
            g = _align_image(g, ref_gray, align)
        g = g.astype(np.float32)
        n += 1
        if n == 1:
            mean[...] = g
        else:
            delta = g - mean
            mean += delta / n
            m2 += delta * (g - mean)

    var = m2 / max(n - 1, 1)
    # 給最小 std=1，避免被極小雜訊放大
    std = np.sqrt(np.maximum(var, 1.0))
    return NormalModel(ref_gray=ref_gray, mean=mean, std=std, n=n)

# ---------------------------- 異常遮罩 ----------------------------

def _zscore_mask(img_gray: np.ndarray, model: NormalModel, z_thresh: float, abs_thr: float) -> np.ndarray:
    g = img_gray.astype(np.float32)
    diff = np.abs(g - model.mean)
    z = diff / (model.std + 1e-6)
    m = ((z >= z_thresh) | (diff >= abs_thr)).astype(np.uint8) * 255
    return m

def _post_process(mask: np.ndarray, min_area: int, open_k: int, close_k: int, dilate_k: int) -> np.ndarray:
    m = mask
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    # 小區塊過濾
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    m = keep
    if dilate_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        m = cv2.dilate(m, k, iterations=1)
    return m

def _overlay(img_gray: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    bgr = _to_bgr(img_gray)
    red = np.zeros_like(bgr); red[:, :, 2] = 255
    out = bgr.copy()
    out[mask > 0] = cv2.addWeighted(bgr[mask > 0], 1 - alpha, red[mask > 0], alpha, 0)
    return out

# ---------------------------- 與你原本相容的 API ----------------------------

def load_reference_images(ref_folder, input_formats):
    """
    仍保留這個函式讓既有呼叫不壞（回傳「平均灰階圖」）。
    內部其實用正常模型建好 mean/std，只回傳 mean（uint8）以維持舊介面。
    """
    exts = _norm_exts(input_formats)
    model = _build_normal_model(Path(ref_folder), exts, recursive=True, align="none")
    return np.clip(model.mean, 0, 255).astype(np.uint8)

def generate_anomaly_mask(ref_img, test_img, threshold=30):
    """
    舊版 API：單純做絕對差 + 閾值 + 固定 5x5 形態學。
    為相容性保留，不建議新程式繼續用；新版走 process_anomaly_detection 的流。
    """
    if ref_img.shape != test_img.shape:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))
    diff = cv2.absdiff(ref_img, test_img)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def process_anomaly_detection(config):
    """
    入口維持同名：讀 A(正常) 建模，對 B 逐張產生異常遮罩（必填鍵與可選鍵如下）。
    必填：
      config['anomaly_detection'] = {
         'reference_folder': str,
         'test_folder': str,
         'output_folder': str,
         'threshold': int,               # 舊鍵，等同 abs_diff_thresh
         'input_formats': list[str],
      }
    可選（有預設值）：
         'z_threshold': float, 'align': 'auto'|'ecc'|'orb'|'none',
         'open_ksize': int, 'close_ksize': int, 'dilate_ksize': int,
         'min_area': int, 'recursive': bool,
         'save_overlay': bool, 'overlay_alpha': float, 'quiet': bool
    """
    params = config["anomaly_detection"]
    cfg = AnomalyConfig(
        reference_folder=Path(params["reference_folder"]),
        test_folder=Path(params["test_folder"]),
        output_folder=Path(params["output_folder"]),
        abs_diff_thresh=int(params.get("threshold", 30)),
        z_thresh=float(params.get("z_threshold", 3.0)),
        align=str(params.get("align", "auto")).lower(),
        open_ksize=int(params.get("open_ksize", 3)),
        close_ksize=int(params.get("close_ksize", 5)),
        dilate_ksize=int(params.get("dilate_ksize", 0)),
        min_area=int(params.get("min_area", 50)),
        input_formats=_norm_exts(params.get("input_formats", IMAGE_EXTS)),
        recursive=bool(params.get("recursive", True)),
        save_overlay=bool(params.get("save_overlay", False)),
        overlay_alpha=float(params.get("overlay_alpha", 0.6)),
        quiet=bool(params.get("quiet", False)),
    )

    _setup_logging(cfg.quiet)
    os.makedirs(cfg.output_folder, exist_ok=True)

    # 1) 建模（mean/std）
    logging.info("建立正常模型...")
    model = _build_normal_model(cfg.reference_folder, cfg.input_formats, cfg.recursive, cfg.align)

    # 2) 列 B
    tests = _list_images(cfg.test_folder, cfg.input_formats, cfg.recursive)
    if not tests:
        raise ValueError("測試圖像資料夾中沒有可用的圖像！")

    # 3) 逐張處理
    for p in tests:
        tg = _imread_gray(p)
        tg = _align_image(tg, model.ref_gray, cfg.align)
        raw = _zscore_mask(tg, model, cfg.z_thresh, cfg.abs_diff_thresh)
        mask = _post_process(raw, cfg.min_area, cfg.open_ksize, cfg.close_ksize, cfg.dilate_ksize)

        rel = p.relative_to(cfg.test_folder)
        out_mask = (cfg.output_folder / rel).with_suffix(".png")
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_mask), mask):
            raise IOError(f"寫入失敗：{out_mask}")
        logging.info(f"已生成 mask：{out_mask}")

        if cfg.save_overlay:
            ov = _overlay(tg, mask, cfg.overlay_alpha)
            out_overlay = (cfg.output_folder / rel).with_suffix(".overlay.jpg")
            cv2.imwrite(str(out_overlay), ov)
