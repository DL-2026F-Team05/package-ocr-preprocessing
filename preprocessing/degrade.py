"""정상(C0) 사진에서 나쁜 촬영 조건을 코드로 만들어 내는 함수 모음.

같은 사진에서 정도만 바꿔 만들 수 있어서, "조건이 나빠질수록 OCR이 얼마나
틀리는지"를 깔끔하게 비교할 수 있습니다. 결과는 data/generated/에 저장되고
GitHub에는 올리지 않습니다(.gitignore).
"""
from __future__ import annotations

import cv2
import numpy as np


def low_light(img: np.ndarray, level: int = 2, seed: int = 0) -> np.ndarray:
    """저조도(C1): 밝기를 낮추고, 어두운 사진 특유의 노이즈를 더함."""
    factor = {1: 0.5, 2: 0.25, 3: 0.1}[level]
    sigma = {1: 2, 2: 4, 3: 6}[level]
    noise = np.random.default_rng(seed).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) * factor + noise, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, level: int = 2, angle: float = 0) -> np.ndarray:
    """흔들림(C3): 한 방향으로 번지는 모션 블러. 세기는 사진 폭에 비례."""
    k = max(3, round(img.shape[1] * {1: 0.004, 2: 0.008, 3: 0.016}[level])) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1
    if angle:
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
        kernel = cv2.warpAffine(kernel, rot, (k, k))
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def rotate(img: np.ndarray, degree: float) -> np.ndarray:
    """회전·뒤집힘(C6): 잘리지 않게 캔버스를 넓혀서 회전."""
    if degree % 90 == 0:
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        code = codes.get(int(degree) % 360)
        return img.copy() if code is None else cv2.rotate(img, code)
    h, w = img.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), -degree, 1)
    cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    rot[0, 2] += nw / 2 - w / 2
    rot[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, rot, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def crop_edge(img: np.ndarray, ratio: float, side: str = "right") -> np.ndarray:
    """잘린 화면(C8): 한쪽 가장자리를 ratio만큼 잘라냄."""
    h, w = img.shape[:2]
    cut_w, cut_h = int(w * ratio), int(h * ratio)
    return {
        "right": img[:, : w - cut_w],
        "left": img[:, cut_w:],
        "top": img[cut_h:, :],
        "bottom": img[: h - cut_h, :],
    }[side].copy()


def defocus(img: np.ndarray, level: int = 2) -> np.ndarray:
    """초점 흐림(C9): 전체가 고르게 뿌옇게 번지는 가우시안 블러."""
    sigma = img.shape[1] * {1: 0.001, 2: 0.002, 3: 0.004}[level]
    return cv2.GaussianBlur(img, (0, 0), sigma)


# 조건 ID → [(단계 태그, 만드는 함수)]
SPECS = {
    "c1": [(f"L{i}", lambda im, i=i: low_light(im, i)) for i in (1, 2, 3)],
    "c3": [(f"L{i}", lambda im, i=i: motion_blur(im, i)) for i in (1, 2, 3)],
    "c6": [(f"L{i}", lambda im, d=d: rotate(im, d)) for i, d in enumerate((15, 45, 90, 180), 1)],
    "c8": [(f"L{i}", lambda im, r=r: crop_edge(im, r)) for i, r in enumerate((0.2, 0.4), 1)],
    "c9": [(f"L{i}", lambda im, i=i: defocus(im, i)) for i in (1, 2, 3)],
}


def generate_variants(img: np.ndarray, conditions=("c1", "c3")):
    """정상 사진 한 장에서 조건별·단계별 사진을 만듭니다. (조건ID, 단계, 이미지)를 차례로 돌려줌."""
    for cond in conditions:
        for tag, fn in SPECS[cond]:
            yield cond, tag, fn(img)
