"""1주차 실험: 조건별 사진에 전처리를 적용하고, 전/후 OCR 결과를 비교합니다.

결과물 (out_dir 아래)
- compare/*.jpg    : 전처리 전/후 사진을 나란히 붙인 비교 이미지 (= 1주차 "전처리 샘플")
- ocr_results.csv  : 사진별·전처리별 OCR 결과, 품질 측정값, CER
- summary.csv/.md  : 조건별 평균 (품질 문제 분석표의 초안)
"""
from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from degrade import generate_variants
from ocr_eval import region_cer
from preprocess import RECIPES, load_image, measure, preprocess

COND_NAMES = {
    "c0": "정상", "c1": "저조도", "c2": "반사광", "c3": "흔들림", "c4": "구김",
    "c5": "작은 글씨·원거리", "c6": "회전·뒤집힘", "c7": "곡면", "c8": "잘린 화면", "c9": "초점 흐림",
}
NAME_RE = re.compile(r"^(p\d+)_(c\d)_([a-z0-9]+)$")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_name(path) -> tuple[str, str, str] | None:
    """'p01_c2_3.jpg' → ('p01', 'c2', '3'). 규칙에 안 맞으면 None."""
    m = NAME_RE.match(Path(path).stem.lower())
    return m.groups() if m else None


def list_images(folder) -> tuple[list[Path], list[str]]:
    """폴더에서 파일 이름 규칙에 맞는 사진과, 맞지 않아 건너뛴 파일 이름을 돌려줌."""
    images, skipped = [], []
    for p in sorted(Path(folder).glob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in IMG_EXT and parse_name(p):
            images.append(p)
        else:
            skipped.append(p.name)
    return images, skipped


def load_labels(path) -> dict[str, dict[str, str]]:
    """labels.csv → {'p01': {'expiry': '2026.12.15', ...}}"""
    labels: dict[str, dict[str, str]] = {}
    path = Path(path)
    if not path.exists():
        return labels
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pid, region, text = (row.get(k, "").strip() for k in ("product_id", "region", "text"))
            if pid and region and text:
                labels.setdefault(pid.lower(), {})[region] = text
    return labels


def generate_from_clean(raw_dir, gen_dir, conditions=("c1", "c3")) -> list[Path]:
    """제품마다 첫 번째 정상(C0) 사진으로 저조도·흔들림 사진을 만들어 gen_dir에 저장."""
    gen_dir = Path(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for p in list_images(raw_dir)[0]:
        pid, cond, _ = parse_name(p)
        if cond == "c0":
            sources.setdefault(pid, p)
    made = []
    for pid, src in sources.items():
        img = load_image(src)
        for cond, tag, out in generate_variants(img, conditions):
            path = gen_dir / f"{pid}_{cond}_{tag}.jpg"
            cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            made.append(path)
    return made


def save_comparison(panels: list[tuple[str, np.ndarray]], path, height: int = 700) -> None:
    """[(제목, 이미지), ...]를 같은 높이로 맞춰 가로로 붙이고, 위에 제목 띠를 넣어 저장."""
    tiles = []
    for title, img in panels:
        h, w = img.shape[:2]
        body = cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
        bar = np.full((48, body.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, title, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.vstack([bar, body]))
        tiles.append(np.full((height + 48, 12, 3), 255, np.uint8))
    cv2.imwrite(str(path), np.hstack(tiles[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run_experiment(paths, ocr, labels: dict, out_dir, recipes: dict = RECIPES) -> pd.DataFrame:
    """사진마다 '전처리 없음'과 '조건별 전처리'로 OCR을 돌려 결과를 저장합니다."""
    out_dir = Path(out_dir)
    (out_dir / "compare").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, path in enumerate(paths, 1):
        pid, cond, tag = parse_name(path)
        img = load_image(path)
        quality = measure(img)
        steps = list(recipes.get(cond, []))
        variants = [("none", img)]
        if steps:
            variants.append(("+".join(steps), preprocess(img, steps)))

        for name, vimg in variants:
            start = time.time()
            result = ocr(vimg)
            text = " ".join(r["text"] for r in result)
            row = {
                "file": path.name, "product": pid, "condition": cond,
                "condition_name": COND_NAMES.get(cond, cond), "tag": tag,
                "source": path.parent.name, "steps": name,
                "stage": "before" if name == "none" else "after",
                "n_boxes": len(result), "ocr_sec": round(time.time() - start, 2),
                **quality, "ocr_text": text,
            }
            cers = []
            for region, label in labels.get(pid, {}).items():
                row[f"cer_{region}"] = region_cer(text, label)
                cers.append(row[f"cer_{region}"])
            row["cer_mean"] = round(float(np.mean(cers)), 4) if cers else np.nan
            rows.append(row)

        if len(variants) > 1:
            panels = [("before", variants[0][1]), (f"after: {variants[1][0]}", variants[1][1])]
            save_comparison(panels, out_dir / "compare" / f"{path.stem}.jpg")
        print(f"[{n}/{len(paths)}] {path.name}  ({' / '.join(v for v, _ in variants)})")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ocr_results.csv", index=False, encoding="utf-8-sig")
    summary = summarize(df)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_markdown(df, summary, out_dir / "summary.md")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """조건별 평균: 품질 측정값, 전처리 전/후 CER과 검출된 글자 영역 수."""
    before = df[df.stage == "before"]
    after = df[df.stage == "after"]
    out = before.groupby("condition").agg(
        condition_name=("condition_name", "first"),
        images=("file", "nunique"),
        brightness=("brightness", "mean"),
        contrast=("contrast", "mean"),
        sharpness=("sharpness", "mean"),
        glare_ratio=("glare_ratio", "mean"),
        boxes_before=("n_boxes", "mean"),
        cer_before=("cer_mean", "mean"),
    )
    if len(after):
        out = out.join(after.groupby("condition").agg(
            recipe=("steps", "first"),
            boxes_after=("n_boxes", "mean"),
            cer_after=("cer_mean", "mean"),
        ))
    else:
        out["recipe"], out["boxes_after"], out["cer_after"] = "", np.nan, np.nan
    out["cer_change"] = out["cer_after"] - out["cer_before"]
    return out.reset_index().round(3)


def _md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else f"{v:.3f}".rstrip("0").rstrip(".")
        return "-" if v is None or v == "" else str(v).replace("|", "/")
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_markdown(df: pd.DataFrame, summary: pd.DataFrame, path) -> None:
    """보고서에 옮겨 쓰기 쉬운 요약 마크다운. '원인' 칸은 사진을 보고 직접 채웁니다."""
    cols = ["condition", "condition_name", "images", "recipe", "brightness", "sharpness",
            "glare_ratio", "cer_before", "cer_after", "cer_change"]
    text = [
        "# AI 2 1주차 전처리 실험 요약", "",
        "CER: 0 = 정답과 완전히 같음, 1 = 전혀 못 읽음. cer_change가 음수면 전처리로 좋아진 것입니다.", "",
        "## 조건별 평균", "", _md_table(summary[[c for c in cols if c in summary]]), "",
    ]
    labeled = df[(df.stage == "before") & (df.cer_mean > 0)]
    if len(labeled):
        worst = labeled.sort_values("cer_mean", ascending=False).head(10)
        worst = worst.assign(cause="")[["file", "condition_name", "cer_mean", "cause"]]
        text += ["## 오인식이 심한 사진 (원인 칸은 직접 채우기)", "", _md_table(worst), ""]
    Path(path).write_text("\n".join(text), encoding="utf-8")
