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
from preprocess import assess_quality, load_image, preprocess  # noqa: E402


def run_pipeline(image: np.ndarray, ocr: Callable, steps: list[str] | None = None) -> dict:
    """사진 한 장을 진단 → 전처리 → OCR까지 처리해 dict로 돌려줍니다.

    steps를 주지 않으면 assess_quality가 추천한 전처리를 씁니다.
    """
    start = time.time()
    quality = assess_quality(image)
    steps = quality["steps"] if steps is None else steps
    info: dict = {}
    processed = preprocess(image, steps, ocr=ocr, info=info) if steps else image
    result = ocr(processed)
    return {
        "quality": {
            "retake": quality["retake"],
            "advice": quality["advice"],
            "issues": quality["issues"],
            "metrics": quality["metrics"],
        },
        "preprocess": {"steps": steps, "rotation": info.get("rotation", 0)},
        "ocr": result,
        "elapsed_sec": round(time.time() - start, 2),
    }


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
        print(f"{Path(p).name}: 글자 {len(data['ocr'])}개, 전처리 {data['preprocess']['steps'] or '없음'}, "
              f"재촬영 {'필요 - ' + q['advice'] if q['retake'] else '불필요'} → {target}")


if __name__ == "__main__":
    main()
