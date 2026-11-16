"""AI 2 이미지 전처리 함수 모음.

모든 함수는 BGR 이미지(cv2.imread 결과, np.ndarray)를 받아 같은 형식으로 돌려줍니다.
다른 모듈에서는 아래 세 가지만 쓰면 됩니다. (docs/interface.md 1번)

- assess_quality(image)            사진 품질 진단 + 재촬영 안내 문구 + 추천 전처리
- preprocess(image, steps, ocr)    전처리 적용 (steps=None이면 자동 선택)
- pipeline.run_pipeline(...)       진단 → 전처리 → OCR을 한 번에 (pipeline.py)

1주차(v1) 기법은 비교 실험을 위해 그대로 남겨 두었고, 2주차(v2)에 명암·회전·노이즈 보정을 추가했습니다.
"""
from __future__ import annotations

from typing import Callable

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
    return resize_max(img, max_side)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    """긴 변이 max_side보다 크면 비율을 유지해 줄입니다."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _small(img: np.ndarray, width: int = 1000) -> np.ndarray:
    """측정용 축소본. 사진 크기와 상관없이 같은 기준으로 재기 위해 폭을 맞춥니다."""
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# 1주차(v1) 기법 — 비교용으로 유지
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
    """노이즈 제거(고정 세기): 1주차 버전."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def reduce_glare(img: np.ndarray, v_min: int = 250, s_max: int = 30,
                 max_ratio: float = 0.25) -> np.ndarray:
    """반사광 제거: 하얗게 날아간 부분을 찾아 주변 색으로 메움(인페인팅).

    반사광에 가려진 글자를 되살리지는 못합니다(1주차 실험에서 효과 없음 확인).
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


# ---------------------------------------------------------------------------
# 2주차(v2) 노이즈 보정
# ---------------------------------------------------------------------------
def noise_sigma(img: np.ndarray) -> float:
    """노이즈 세기 추정(Immerkær 방식). 글자·무늬 같은 큰 구조는 걸러내고 자글자글한 점만 잽니다.

    값이 클수록 노이즈가 많습니다. 정상 사진 약 1~3, 노이즈를 일부러 넣은 사진 5~18.
    """
    g = cv2.cvtColor(_small(img), cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    conv = cv2.filter2D(g, -1, kernel)[1:-1, 1:-1]
    return float(np.abs(conv).mean() * np.sqrt(np.pi / 2) / 6)


def denoise_auto(img: np.ndarray) -> np.ndarray:
    """노이즈 제거(자동 세기): 추정한 노이즈 세기에 맞춰 Non-local means의 강도를 정합니다.

    1주차에는 세기를 5로 고정했습니다. 2주차 사전 실험(EasyOCR)에서 세게 지울수록 가는 글자 획이 뭉개져
    오히려 나빠져서, 추정 노이즈만큼만 약하게(최대 12) 지웁니다.
    """
    h = float(np.clip(noise_sigma(img) * 1.0, 3, 12))
    return cv2.fastNlMeansDenoisingColored(img, None, h, min(h, 10), 7, 21)


def median3(img: np.ndarray) -> np.ndarray:
    """중간값 필터(3×3): 점 모양 노이즈에 빠르고 가볍게 대응. 비교용."""
    return cv2.medianBlur(img, 3)


# ---------------------------------------------------------------------------
# 2주차(v2) 명암 보정
# ---------------------------------------------------------------------------
def brighten(img: np.ndarray, target: float = 0.5) -> np.ndarray:
    """밝기 보정: 밝기 채널(L)에만 감마를 적용해 어두운 사진을 밝힙니다.

    1주차 gamma는 색(RGB) 채널 전체를 밝혀서 색 노이즈까지 키웠습니다.
    이미 충분히 밝은 사진(중간 밝기 0.42 이상)은 건드리지 않습니다.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32) / 255
    mid = float(np.clip(np.median(L), 1e-3, 0.99))
    if mid >= 0.42:
        return img.copy()
    e = float(np.clip(np.log(target) / np.log(mid), 0.25, 1.0))
    lab[..., 0] = np.clip((L ** e) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def stretch(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    """명암 늘이기: 가장 어두운 1%와 가장 밝은 1%를 기준으로 밝기 범위를 0~255로 넓힙니다."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    a, b = np.percentile(L, lo), np.percentile(L, hi)
    if b - a < 10:
        return img.copy()
    lab[..., 0] = np.clip((L - a) * 255 / (b - a), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_soft(img: np.ndarray) -> np.ndarray:
    """약한 CLAHE(clip 1.5): 1주차 CLAHE(2.0)보다 노이즈를 덜 키웁니다."""
    return clahe(img, clip=1.5)


def illumination(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """조명 평탄화: 큰 그림자·주름 음영을 흐린 배경으로 나눠 지우고, 글자 대비는 남깁니다."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    background = cv2.GaussianBlur(L, (0, 0), max(img.shape[:2]) / 30) + 1
    flat = L / background * float(np.median(L))
    lab[..., 0] = np.clip(strength * flat + (1 - strength) * L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 2주차(v2) 회전 보정
# ---------------------------------------------------------------------------
def rotate_image(img: np.ndarray, degree: float) -> np.ndarray:
    """시계 방향으로 degree만큼 돌립니다. 잘리지 않게 캔버스를 넓히고 가장자리는 흰색으로 채웁니다."""
    degree = float(degree) % 360
    if degree == 0:
        return img.copy()
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if degree in codes:
        return cv2.rotate(img, codes[int(degree)])
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -degree, 1)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _text_mask(img: np.ndarray, width: int = 800) -> np.ndarray:
    """밝은 바탕 위의 어두운 가는 획(=글자 후보)만 남긴 흑백 마스크."""
    g = cv2.cvtColor(_small(img, width), cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    background = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
    return ((blackhat > 40) & (background > 150)).astype(np.uint8) * 255


def estimate_skew(img: np.ndarray) -> tuple[float, float]:
    """글자 줄이 기울어진 각도를 추정합니다. (각도, 확신도)를 돌려줍니다.

    글자 획을 조금 부풀려 '줄 모양 덩어리'로 만든 뒤, 가늘고 긴 덩어리들의 방향을 모아
    가장 많이 나온 방향을 고릅니다. 각도는 시계 방향이 +이고 -90~90 범위입니다.
    바닥 나뭇결 같은 배경을 글자로 착각할 수 있어서, 실제로 돌릴지는 fix_rotation에서 OCR로 확인합니다.
    """
    m = cv2.dilate(_text_mask(img), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    angles, weights = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 40:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
        ev, evec = np.linalg.eigh(cov)
        minor, major = 4 * np.sqrt(max(ev[0], 1e-9)), 4 * np.sqrt(ev[1])
        if minor < 4 or minor > 40 or major / minor < 3:
            continue
        angles.append(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180)
        weights.append(major)
    if len(angles) < 5:
        return 0.0, 0.0
    angles, weights = np.array(angles), np.array(weights)
    hist, _ = np.histogram(angles, bins=180, range=(0, 180), weights=weights)
    smooth = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(7), "valid")
    peak = float(smooth.argmax())
    diff = (angles - peak + 90) % 180 - 90
    near = np.abs(diff) < 4
    angle = (peak + np.average(diff[near], weights=weights[near]) + 90) % 180 - 90
    return round(float(angle), 2), round(float(weights[near].sum() / weights.sum()), 3)


def crop_to_text(img: np.ndarray, margin: float = 0.08) -> np.ndarray:
    """글자가 있는 영역만 남기고 바깥 여백을 잘라냅니다(회전 뒤 생긴 빈 귀퉁이 제거용).

    여백이 크면 OCR이 사진을 줄일 때 글자까지 작아지기 때문에 필요합니다.
    글자 위치는 위·아래 0.5%를 빼고 잡아 튀는 점에 흔들리지 않게 하고, 넉넉히 margin을 둡니다.
    """
    h, w = img.shape[:2]
    mask = _text_mask(img)
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return img
    sx, sy = w / mask.shape[1], h / mask.shape[0]
    x0, x1 = np.percentile(xs, [0.5, 99.5]) * sx
    y0, y1 = np.percentile(ys, [0.5, 99.5]) * sy
    mx, my = (x1 - x0) * margin + 20, (y1 - y0) * margin + 20
    x0, x1 = int(max(0, x0 - mx)), int(min(w, x1 + mx))
    y0, y1 = int(max(0, y0 - my)), int(min(h, y1 + my))
    if (x1 - x0) * (y1 - y0) < 0.15 * w * h:   # 너무 작게 잘리면 잘못 잡은 것으로 보고 그대로 둠
        return img
    return img[y0:y1, x0:x1].copy()


def _rotate_and_trim(img: np.ndarray, degree: float) -> np.ndarray:
    out = rotate_image(img, degree)
    return out if degree % 90 == 0 else crop_to_text(out)


def ocr_score(ocr: Callable, img: np.ndarray, scale: float = 1.0) -> float:
    """OCR이 '확신을 갖고 읽은 글자 수'. 방향이 맞을수록 커집니다.

    후보끼리 공정하게 비교하려고, 사진 크기가 달라도 같은 배율(scale)로 줄여서 읽습니다.
    """
    if scale < 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    result = ocr(img)
    return float(sum(r["score"] * len(r["text"]) for r in result if r["score"] >= 0.5))


def fix_rotation(img: np.ndarray, ocr: Callable | None = None, check_flip: bool = False,
                 info: dict | None = None) -> np.ndarray:
    """회전 보정. 기울기를 추정한 뒤, OCR이 있으면 후보 각도마다 읽혀 보고 가장 잘 읽히는 쪽을 고릅니다.

    - 후보: 그대로(0°), 추정 각도만큼 되돌리기, (check_flip=True면) 각각을 180° 뒤집은 것
    - 기울기 추정 확신도가 0.5 이상이면 되돌린 후보에 15% 가산점 (수평 글자가 더 잘 읽히므로)
    - 각도를 되돌리면 귀퉁이에 빈 여백이 생기므로 글자 영역만 남기고 잘라냅니다(crop_to_text).
    - OCR이 없으면 확신도가 높고 각도가 2~45° 사이일 때만 되돌립니다(보수적).
    - info에 dict를 넘기면 고른 각도와 후보 점수를 적어 줍니다(실험 기록용).
    """
    angle, conf = estimate_skew(img)
    candidates = [0.0]
    if conf >= 0.3 and abs(angle) >= 2:
        candidates.append(-angle)
    if check_flip:
        candidates += [c + 180 for c in candidates]
    if ocr is None:
        chosen = -angle if (conf >= 0.6 and 2 <= abs(angle) <= 45) else 0.0
        scores = {}
    elif len(candidates) == 1:
        chosen, scores = 0.0, {}
    else:
        scale = min(1.0, 960 / max(img.shape[:2]))   # 모든 후보를 같은 배율로 줄여 공정하게 비교
        images = {c: _rotate_and_trim(img, c) for c in candidates}
        scores = {c: ocr_score(ocr, im, scale) for c, im in images.items()}
        if conf >= 0.5:   # 기울기 추정이 확실하면, 수평으로 되돌린 후보에 15% 가산점
            for c in scores:
                if c % 180 != 0:
                    scores[c] *= 1.15
        chosen = max(scores, key=scores.get)
    if info is not None:
        info.update({"skew_angle": angle, "skew_conf": conf, "rotation": round(chosen % 360, 2),
                     "candidates": {round(k % 360, 2): round(v, 1) for k, v in scores.items()}})
    return _rotate_and_trim(img, chosen)


# ---------------------------------------------------------------------------
# 품질 측정과 진단
# ---------------------------------------------------------------------------
def measure(img: np.ndarray) -> dict:
    """사진 품질을 숫자로 측정합니다(1주차 항목 + 2주차 항목).

    - brightness: 평균 밝기 (0~1, 낮을수록 어두움)
    - contrast: 밝기의 표준편차 (0~1)
    - sharpness: 라플라시안 분산 (1주차 선명도, 밝기·대비의 영향을 받음)
    - sharpness_index: 선명도 ÷ 대비² (2주차, 어두운 사진도 공정하게 비교. 0.06 미만이면 흔들림)
    - noise: 노이즈 세기 (2주차)
    - text_height: 폭 1000픽셀 기준 글자 높이의 중간값 (2주차, 5 이하면 너무 멀리서 찍음)
    - glare_ratio: 하얗게 날아간 픽셀 비율 (1주차, 반사광을 구분하지 못해 참고용)
    """
    small = _small(img)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    contrast = float(gray.std() / 255)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": round(float(gray.mean() / 255), 3),
        "contrast": round(contrast, 3),
        "sharpness": round(sharp, 1),
        "sharpness_index": round(sharp / max((contrast * 255) ** 2, 1.0), 3),
        "noise": round(noise_sigma(img), 2),
        "text_height": _text_height(gray),
        "glare_ratio": round(float(((hsv[..., 2] >= 250) & (hsv[..., 1] <= 30)).mean()), 4),
    }


def _text_height(gray: np.ndarray) -> float:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) > 40).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, area = stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_AREA]
    h = h[(area >= 6) & (h >= 3) & (h <= 60)]
    return float(np.median(h)) if len(h) >= 30 else 0.0


# 진단 기준값. 1주차·2주차 사진(정상/저조도/흔들림/원거리)의 측정값을 보고 정했습니다.
THRESHOLDS = {
    "too_dark": 0.08,        # brightness 이 값 미만: 재촬영 권장
    "dark": 0.35,            # brightness 이 값 미만: 밝기 보정
    "blurry": 0.06,          # sharpness_index 이 값 미만: 흔들림 (정상 0.10 이상, 흔들림 0.02~0.04)
    "noisy": 4.0,            # noise 이 값 초과: 노이즈 제거 (정상 1~3)
    "too_far": 5.0,          # text_height 이 값 이하: 너무 멀리서 찍음 (정상 9, 원거리 4)
    "skew": 3.0,             # 추정 기울기(도) 이 값 이상이고 확신도 0.3 이상이면 회전 보정 시도
}

ADVICE = {
    "too_dark": "사진이 너무 어둡습니다. 밝은 곳에서 다시 찍어 주세요.",
    "blurry": "사진이 흔들렸습니다. 휴대폰을 잠시 멈춘 채로 다시 찍어 주세요.",
    "too_far": "글자가 너무 작게 찍혔습니다. 제품에 조금 더 가까이 대고 찍어 주세요.",
}


def assess_quality(img: np.ndarray) -> dict:
    """사진 품질을 진단해서 '무엇이 문제인지, 다시 찍어야 하는지, 어떤 전처리를 할지'를 알려 줍니다.

    돌려주는 값(예):
    {"metrics": {...}, "issues": ["dark"], "retake": False, "advice": "", "steps": ["denoise_auto", "stretch"]}
    - issues: dark, too_dark, blurry, noisy, too_far, skewed 중 해당하는 것
    - retake: 전처리로 해결하기 어려워 다시 찍는 게 나은 경우 True
    - advice: AI 4가 음성으로 읽어 줄 재촬영 안내 문구(retake일 때만)
    - steps: preprocess에 그대로 넘길 추천 전처리 목록
    """
    m = measure(img)
    t = THRESHOLDS
    skew, conf = estimate_skew(img)
    m.update({"skew_angle": skew, "skew_conf": conf})
    issues, steps = [], []
    if m["brightness"] < t["too_dark"]:
        issues.append("too_dark")
    if m["brightness"] < t["dark"]:
        issues.append("dark")
    if m["noise"] > t["noisy"] or m["brightness"] < t["dark"]:
        if m["noise"] > t["noisy"]:
            issues.append("noisy")
        steps.append("denoise_auto")
    if m["brightness"] < t["dark"]:
        steps.append("stretch")
    if m["brightness"] >= 0.15 and m["sharpness_index"] < t["blurry"]:
        issues.append("blurry")
    # 노이즈가 심하면 글자 획이 끊겨 글자 높이가 작게 재지므로 거리 판단을 건너뜀
    if 0 < m["text_height"] <= t["too_far"] and m["noise"] <= t["noisy"]:
        issues.append("too_far")
        steps.append("upscale")
    if abs(skew) >= t["skew"] and conf >= 0.3:
        issues.append("skewed")
        steps.append("rotate")
    retake = [i for i in ("too_dark", "blurry", "too_far") if i in issues]
    return {
        "metrics": m,
        "issues": issues,
        "retake": bool(retake),
        "advice": ADVICE[retake[0]] if retake else "",
        "steps": steps,
    }


def auto_steps(img: np.ndarray, metrics: dict | None = None) -> list[str]:
    """품질 진단 결과에 따라 추천 전처리 목록을 고릅니다(2주차 버전)."""
    return assess_quality(img)["steps"]


# ---------------------------------------------------------------------------
# 단계 이름표와 조건별 조합
# ---------------------------------------------------------------------------
STEPS = {
    # 1주차
    "gamma": gamma,
    "clahe": clahe,
    "sharpen": sharpen,
    "denoise": denoise,
    "glare": reduce_glare,
    "upscale": upscale,
    # 2주차
    "denoise_auto": denoise_auto,
    "median3": median3,
    "brighten": brighten,
    "stretch": stretch,
    "clahe_soft": clahe_soft,
    "illumination": illumination,
    "rotate": fix_rotation,     # OCR을 넘기면 OCR로 방향을 확인합니다
    "rotate_flip": None,        # rotate + 180° 뒤집힘까지 확인 (preprocess에서 처리)
}

# 1주차 조합(비교용). 1주차 실험 결과: 원거리만 좋아지고 나머지는 같거나 나빠짐.
RECIPES_V1 = {
    "c0": [],
    "c1": ["gamma", "denoise", "clahe"],
    "c2": ["glare", "clahe"],
    "c3": ["sharpen"],
    "c4": ["clahe", "sharpen"],
    "c5": ["upscale", "sharpen"],
    "c6": [],
    "c7": ["clahe"],
    "c8": [],
    "c9": ["sharpen"],
    "c10": ["denoise"],
}

# 2주차 조합. 명암·회전·노이즈 보정 후보를 EasyOCR로 먼저 비교해서 골랐습니다(results/ai2_week2 참고).
# - 저조도: 노이즈 제거 → 명암 늘이기가 가장 좋았음 (평균 CER 0.568 → 0.406, 1주차 조합 0.484)
# - 노이즈·구김: 사전 실험에서 뚜렷한 효과가 없어 PaddleOCR로 다시 확인하는 조합
RECIPES = {
    "c0": [],
    "c1": ["denoise_auto", "stretch"],
    "c2": [],                        # 반사광은 전처리로 복구 불가 → 재촬영 안내
    "c3": [],                        # 흔들림은 전처리로 복구 불가 → 재촬영 안내
    "c4": ["illumination"],
    "c5": ["upscale"],
    "c6": ["rotate_flip"],
    "c7": [],
    "c8": [],
    "c9": [],
    "c10": ["denoise_auto"],
}
RECIPES_V2 = RECIPES


# ---------------------------------------------------------------------------
# 공개 함수 (다른 모듈은 이것만 사용)
# ---------------------------------------------------------------------------
def preprocess(image: np.ndarray, steps: list[str] | None = None,
               ocr: Callable | None = None, info: dict | None = None) -> np.ndarray:
    """전처리 기법을 순서대로 적용합니다.

    steps: 기법 이름 목록. 예) ["denoise_auto", "stretch"]
           None이면 assess_quality가 추천한 목록을 씁니다.
    ocr:   회전 보정(rotate)에서 방향을 확인할 OCR 함수(선택). 없으면 보수적으로만 돌립니다.
    info:  dict를 넘기면 회전 보정 결과 등 처리 기록을 적어 줍니다.
    """
    if steps is None:
        steps = assess_quality(image)["steps"]
    out = image
    for name in steps:
        if name not in STEPS:
            raise ValueError(f"모르는 전처리 이름: {name} (가능: {', '.join(STEPS)})")
        if name in ("rotate", "rotate_flip"):
            out = fix_rotation(out, ocr=ocr, check_flip=(name == "rotate_flip"), info=info)
        else:
            out = STEPS[name](out)
    return out
