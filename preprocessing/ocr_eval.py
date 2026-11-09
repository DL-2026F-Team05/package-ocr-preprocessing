"""전처리 효과를 확인하기 위한 OCR 실행기와 오류율(CER) 계산.

OCR 모델 자체는 AI 1 담당이고, 여기서는 기존 OCR을 그대로 불러 비교 실험에만 씁니다.
AI 1의 모델이 나오면 OCREngine에 그 모델을 추가해서 같은 실험을 다시 돌리면 됩니다.
"""
from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np


class OCREngine:
    """OCR 실행기. 결과 형식은 docs/interface.md 2번과 같습니다.

    engine="paddle" : PaddleOCR 한국어 모델 (기본)
    engine="easyocr": EasyOCR (비교용, 설치가 더 쉬움)
    """

    def __init__(self, engine: str = "paddle"):
        self.engine = engine
        if engine == "paddle":
            from paddleocr import PaddleOCR

            self._model = PaddleOCR(
                lang="korean",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        elif engine == "easyocr":
            import easyocr

            try:
                import torch

                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
            self._model = easyocr.Reader(["ko", "en"], gpu=gpu)
        else:
            raise ValueError("engine은 'paddle' 또는 'easyocr'만 가능합니다.")

    def __call__(self, image: np.ndarray) -> list[dict]:
        return self._paddle(image) if self.engine == "paddle" else self._easyocr(image)

    def _paddle(self, image: np.ndarray) -> list[dict]:
        out = []
        for res in self._model.predict(image):
            texts, scores, polys = (_field(res, k) for k in ("rec_texts", "rec_scores", "rec_polys"))
            if len(polys) != len(texts):
                polys = [None] * len(texts)
            for text, score, box in zip(texts, scores, polys):
                out.append({
                    "text": str(text),
                    "box": None if box is None else np.asarray(box).astype(int).tolist(),
                    "score": float(score),
                })
        return out

    def _easyocr(self, image: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return [
            {"text": str(text), "box": np.asarray(box).astype(int).tolist(), "score": float(score)}
            for box, text, score in self._model.readtext(rgb)
        ]


def _field(res, key):
    """PaddleOCR 결과에서 값을 꺼냄 (버전에 따라 꺼내는 방법이 달라서 둘 다 시도)."""
    for getter in (lambda: res[key], lambda: res.json["res"][key]):
        try:
            value = getter()
            return list(value) if value is not None else []
        except Exception:
            continue
    return []


def normalize(text: str) -> str:
    """비교 전에 표기 차이를 없앰: 전각/반각 통일, 공백 제거, 소문자."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def region_cer(ocr_text: str, label: str) -> float:
    """정답 라벨(예: 유통기한 '2026.12.15')이 OCR 결과 어딘가에 얼마나 비슷하게 있는지.

    OCR은 사진 속 글자를 전부 읽기 때문에 전체 결과와 라벨을 그대로 비교하면 안 됩니다.
    그래서 OCR 결과 중 라벨과 가장 비슷한 부분을 찾아, 그 부분과의 글자 오류율을 계산합니다.
    0 = 완벽히 일치, 1 = 전혀 못 읽음.
    """
    pattern, text = normalize(label), normalize(ocr_text)
    m = len(pattern)
    if m == 0:
        return 0.0
    col = list(range(m + 1))
    best = col[m]
    for ch in text:
        new = [0] * (m + 1)
        for i in range(1, m + 1):
            cost = 0 if pattern[i - 1] == ch else 1
            new[i] = min(col[i - 1] + cost, col[i] + 1, new[i - 1] + 1)
        col = new
        best = min(best, col[m])
    return round(min(best / m, 1.0), 4)
