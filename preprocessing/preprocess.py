"""AI 2 이미지 전처리 함수 모음.

모든 함수는 BGR 이미지(cv2.imread 결과, np.ndarray)를 받아 같은 형식으로 돌려줍니다.
다른 모듈에서는 preprocess(image, steps)만 쓰면 됩니다. (docs/interface.md 1번)
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 불러오기
# ---------------------------------------------------------------------------
def load_image(path, max_side: int = 2400) -> np.ndarray:
    """사진을 읽고, 긴 변이 max_side보다 크면 줄입니다. (폰 사진은 너무 커서 OCR이 느림)"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# 전처리 기법
# ---------------------------------------------------------------------------
def gamma(img: np.ndarray, target: float = 0.5, lo: float = 0.3, hi: float = 2.0) -> np.ndarray:
    """감마 보정: 평균 밝기가 target(0~1)이 되도록 밝기 곡선을 조절. 어두운 사진용."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(np.clip(gray.mean() / 255.0, 1e-3, 0.999))
    exponent = float(np.clip(np.log(target) / np.log(mean), lo, hi))
    lut = ((np.arange(256) / 255.0) ** exponent * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def clahe(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE: 사진을 작은 칸으로 나눠 칸마다 대비를 높임. 밝기 정보(L 채널)에만 적용."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def sharpen(img: np.ndarray, amount: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """언샤프 마스킹: 흐린 버전과의 차이를 더해 글자 경계를 또렷하게."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def denoise(img: np.ndarray, strength: int = 5) -> np.ndarray:
    """노이즈 제거: 어두운 사진을 밝힐 때 함께 커지는 자글자글한 점을 줄임."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def reduce_glare(img: np.ndarray, v_min: int = 250, s_max: int = 30,
                 max_ratio: float = 0.25) -> np.ndarray:
    """반사광 제거: 하얗게 날아간 부분을 찾아 주변 색으로 메움(인페인팅).

    주의: 반사광에 완전히 가려진 글자를 되살리지는 못합니다. 번쩍이는 덩어리가
    글자로 잘못 검출되는 것을 줄이는 효과가 주 목적입니다.
    하얀 영역이 max_ratio보다 넓으면 반사광이 아니라 흰 배경으로 보고 그대로 둡니다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).astype(np.uint8) * 255
    ratio = mask.mean() / 255
    if ratio < 0.0005 or ratio > max_ratio:
        return img.copy()
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def upscale(img: np.ndarray, scale: float = 1.5, max_side: int = 4000) -> np.ndarray:
    """확대: 작은 글자를 키움. 너무 커지지 않게 max_side로 제한."""
    h, w = img.shape[:2]
    scale = min(scale, max_side / max(h, w))
    if scale <= 1:
        return img.copy()
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)


STEPS = {
    "gamma": gamma,
    "clahe": clahe,
    "sharpen": sharpen,
    "denoise": denoise,
    "glare": reduce_glare,
    "upscale": upscale,
}

# 1주차: 조건별로 "효과가 있을 것 같은" 조합. 실험 결과를 보고 2주차에 고칩니다.
RECIPES = {
    "c0": [],                               # 정상: 비교 기준이라 그대로
    "c1": ["gamma", "denoise", "clahe"],    # 저조도
    "c2": ["glare", "clahe"],               # 반사광
    "c3": ["sharpen"],                      # 흔들림
    "c4": ["clahe", "sharpen"],             # 구김 (원근 보정은 2주차)
    "c5": ["upscale", "sharpen"],           # 작은 글씨·원거리
    "c6": [],                               # 회전 (회전 보정은 2주차)
    "c7": ["clahe"],                        # 곡면
    "c8": [],                               # 잘린 화면 (전처리로 복구 불가 → 촬영 안내로 해결)
    "c9": ["sharpen"],                      # 초점 흐림
}


# ---------------------------------------------------------------------------
# 품질 측정과 자동 선택
# ---------------------------------------------------------------------------
def measure(img: np.ndarray) -> dict:
    """사진 품질을 숫자로 측정. 조건별로 값이 어떻게 다른지 보는 '품질 문제 분석'에 씁니다.

    - brightness: 평균 밝기 (0~1, 낮을수록 어두움)
    - contrast: 밝기의 표준편차 (0~1, 낮을수록 흐릿하고 밋밋함)
    - sharpness: 라플라시안 분산 (낮을수록 흐림·흔들림)
    - glare_ratio: 하얗게 날아간 픽셀 비율 (높을수록 반사광 많음)
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (1000, max(1, round(h * 1000 / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return {
        "brightness": round(float(gray.mean() / 255), 3),
        "contrast": round(float(gray.std() / 255), 3),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "glare_ratio": round(float(((hsv[..., 2] >= 250) & (hsv[..., 1] <= 30)).mean()), 4),
    }


def auto_steps(img: np.ndarray, metrics: dict | None = None) -> list[str]:
    """품질 측정값을 보고 전처리 조합을 고름. 기준값은 임시이며 실험 후 조정합니다."""
    m = metrics or measure(img)
    steps = []
    if m["glare_ratio"] > 0.002:
        steps.append("glare")
    if m["brightness"] < 0.35:
        steps += ["gamma", "denoise"]
    if m["contrast"] < 0.18 or m["brightness"] < 0.35:
        steps.append("clahe")
    if m["sharpness"] < 100:
        steps.append("sharpen")
    return steps


# ---------------------------------------------------------------------------
# 공개 함수 (다른 모듈은 이것만 사용)
# ---------------------------------------------------------------------------
def preprocess(image: np.ndarray, steps: list[str] | None = None) -> np.ndarray:
    """전처리 기법을 순서대로 적용합니다.

    steps: 기법 이름 목록. 예) ["gamma", "clahe"]
           None이면 사진 품질을 측정해 자동으로 고릅니다(auto_steps).
    """
    if steps is None:
        steps = auto_steps(image)
    out = image
    for name in steps:
        if name not in STEPS:
            raise ValueError(f"모르는 전처리 이름: {name} (가능: {', '.join(STEPS)})")
        out = STEPS[name](out)
    return out
