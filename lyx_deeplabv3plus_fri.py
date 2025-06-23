# lyx_deeplabv3plus_fri.py  ───────────────────────────────────────────
import cv2
import numpy as np
import torch
from deeplabv3plus.infer import (
    load_model, preprocess_image, visualize_segmentation
)
from deeplabv3plus.config import experiments, NUM_CLASSES, DEVICE, CLASSES_15, PALETTE_15
from deeplabv3plus.friction_model import FrictionCalculator


class lyx_deeplabv3plus_fri:
    """
    DeepLabV3+ 推断 + 摩擦系数计算 一体化封装
    ─────────────────────────────────────────────
    call(image:BGR np.ndarray) → (seg_vis, seg_cls, friction_map)
    """

    def __init__(self,
                 model_path="deeplabv3plus/last_model_0.43.pth",
                 config=None):
        self.cfg = config if config else experiments[0]
        self.device = DEVICE
        print(f"[DeepLabV3+] loading weight: {model_path}")
        self.model = load_model(model_path, self.cfg, self.device)

        self.friction_calc = FrictionCalculator(
            yaml_path="deeplabv3plus/WildScenes_categories.yaml",
            num_classes=NUM_CLASSES
        )

    # ──────────────────────────────────────────────────────────────
    def __call__(self, image_bgr):
        """
        参数
        ----
        image_bgr : np.ndarray (H,W,3)   OpenCV BGR

        返回
        ----
        seg_vis      : np.uint8 (H,W,3)  彩色语义分割结果
        seg_cls      : np.uint8 (H,W)    类别索引图
        friction_map : np.float32(H,W)   每像素摩擦系数
        """
        # 1. 预处理
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp, orig_size, _ = preprocess_image(rgb_image=rgb)     # NCHW float32

        # 2. 网络前向
        with torch.no_grad():
            logits = self.model(inp.to(self.device))            # 1×C×h×w
            probs  = torch.softmax(logits, dim=1)               # 概率
            seg_cls_small = torch.argmax(probs, 1).squeeze(0).cpu().numpy()

        # 3. resize 回原分辨率
        H, W = orig_size
        seg_cls = cv2.resize(seg_cls_small.astype(np.uint8),
                             (W, H),
                             interpolation=cv2.INTER_NEAREST)

        # 彩色可视化
        seg_vis = visualize_segmentation(seg_cls)

        # ⬇ 把 C×h×w → C×H×W，用双线性插值逐通道 resize
        probs_np = probs.squeeze(0).cpu().numpy()               # C×h×w
        C, h, w = probs_np.shape
        probs_resized = np.empty((C, H, W), dtype=np.float32)
        for c in range(C):
            probs_resized[c] = cv2.resize(
                probs_np[c], (W, H), interpolation=cv2.INTER_LINEAR
            )

        

        # 4. 计算摩擦系数
        friction_map, _, _ = self.friction_calc.assign_friction(probs_resized)

        # 5. 返回
        return seg_vis, seg_cls, friction_map
        # -- 如需 seg_prob 调试，请改为:
        # return seg_vis, seg_cls, friction_map, probs_resized
