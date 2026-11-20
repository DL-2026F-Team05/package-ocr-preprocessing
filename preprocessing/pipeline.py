"""모듈 연결용 파이프라인: 사진 → 품질 진단 → 전처리 → OCR → JSON.

AI 3(정보 추출)은 출력의 "ocr"을, AI 4(TTS·웹 데모)는 "quality.advice"를 그대로 쓰면 됩니다.
형식은 docs/interface.md를 따릅니다.

명령줄에서 실행:
    python preprocessing/pipeline.py data/raw/p01_c0_1.jpg --engine paddle --out sample.json

코드에서 실행:
    from pipeline import run_pipeline
    result = run_pipeline(image, ocr)           # ocr = OCREngine("paddle")
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import ADVICE, CONFIDENCE, assess_quality, fix_rotation, load_image, preprocess  # noqa: E402


def ocr_confidence(result: list[dict]) -> float:
    """OCR 확신도: 글자 줄별 score의 평균. 글자를 하나도 못 찾으면 0."""
    return float(np.mean([r["score"] for r in result])) if result else 0.0


def run_pipeline(image: np.ndarray, ocr: Callable, steps: list[str] | None = None, *,
                 quality: dict | None = None, first_result: list | None = None,
                 return_image: bool = False) -> dict:
    """사진 한 장을 진단 → 전처리 → OCR까지 처리해 dict로 돌려줍니다.

    1. assess_quality로 사진을 진단하고 추천 전처리(steps)를 적용합니다. steps를 직접 주면 그것을 씁니다.
    2. OCR을 돌립니다.
    3. OCR 확신도가 낮으면(PaddleOCR 기준 0.8 미만) 사진이 돌아가거나 뒤집혔을 수 있으므로,
       회전·180° 뒤집힘 후보를 확인해서 더 잘 읽히는 방향이 있으면 그 방향으로 다시 OCR합니다.
       (확신도가 높은 사진은 이 단계를 건너뛰어서 빠릅니다.)
    4. 그래도 확신도가 낮으면(PaddleOCR 기준 0.7 미만) 재촬영을 안내합니다.
    확신도 기준은 OCR 엔진마다 다릅니다(preprocess.CONFIDENCE).

    quality, first_result: 실험 코드에서 이미 계산한 진단 결과·OCR 결과를 재사용할 때만 씁니다.
    return_image=True면 최종 처리된 사진을 "image"에 담아 줍니다(JSON 저장 전에는 빼야 함).
    """
    start = time.time()
    quality = quality or assess_quality(image)
    th = CONFIDENCE.get(getattr(ocr, "engine", "paddle"), CONFIDENCE["paddle"])   # 엔진별 확신도 기준
    steps = [s for s in (quality["steps"] if steps is None else steps) if s not in ("rotate", "rotate_flip")]
    processed = preprocess(image, steps) if steps else image
    result = first_result if first_result is not None else ocr(processed)
    confidence = ocr_confidence(result)
    final = processed

    info: dict = {}
    retried = confidence < th["retry_rotation"]
    if retried:
        rotated = fix_rotation(processed, ocr=ocr, check_flip=True, info=info)
        if info.get("rotation", 0) != 0:
            new_result = ocr(rotated)
            new_confidence = ocr_confidence(new_result)
            if new_confidence > confidence:
                result, confidence, final = new_result, new_confidence, rotated
            else:
                info["rotation"] = 0

    retake, advice = quality["retake"], quality["advice"]
    if not retake and confidence < th["retake"]:
        retake, advice = True, ADVICE["low_confidence"]
    out = {
        "quality": {
            "retake": retake,
            "advice": advice,
            "issues": quality["issues"],
            "ocr_confidence": round(confidence, 3),
            "metrics": quality["metrics"],
        },
        "preprocess": {"steps": steps, "rotation": info.get("rotation", 0), "orientation_checked": retried,
                       "rotation_scores": {str(k): v for k, v in info.get("candidates", {}).items()}},
        "ocr": result,
        "elapsed_sec": round(time.time() - start, 2),
    }
    if return_image:
        out["image"] = final
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(data: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="사진 → 품질 진단 → 전처리 → OCR → JSON")
    parser.add_argument("images", nargs="+", help="사진 경로 (여러 장 가능)")
    parser.add_argument("--engine", default="paddle", choices=["paddle", "easyocr"])
    parser.add_argument("--out", default="results/sample_output", help="JSON 저장 폴더 또는 .json 파일")
    args = parser.parse_args()

    from ocr_eval import OCREngine
    ocr = OCREngine(args.engine)
    out = Path(args.out)
    for p in args.images:
        data = {"image": Path(p).name, "engine": args.engine, **run_pipeline(load_image(p), ocr)}
        target = out if out.suffix == ".json" and len(args.images) == 1 else out / f"{Path(p).stem}.json"
        save_json(data, target)
        q = data["quality"]
        rot = data["preprocess"]["rotation"]
        print(f"{Path(p).name}: 글자 {len(data['ocr'])}개, 전처리 {data['preprocess']['steps'] or '없음'}"
              f"{f', 회전 {rot}°' if rot else ''}, 확신도 {q['ocr_confidence']}, "
              f"재촬영 {'필요 - ' + q['advice'] if q['retake'] else '불필요'} → {target}")


if __name__ == "__main__":
    main()
