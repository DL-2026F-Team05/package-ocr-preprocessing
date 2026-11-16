# 모듈 간 인터페이스 (초안)

각 모듈이 무엇을 받고 무엇을 넘겨주는지 정한 약속입니다. 이 형식만 맞추면 서로의 작업을 기다리지 않고 동시에 개발할 수 있습니다.

> 초안입니다. 바꾸고 싶은 부분은 팀에서 합의한 뒤 이 문서를 먼저 고쳐주세요.

## 전체 흐름

```
사진 → assess_quality() ─┬─ retake=True  → tts()로 재촬영 안내 (advice 문장)
                        └─ preprocess() → ocr() → extract() → tts()
```

## 1. 전처리 (AI 2) — `preprocessing/`

### 1-1. 품질 진단

```python
def assess_quality(image: np.ndarray) -> dict:
    ...
```

출력 예시 (어두운 사진):

```json
{
  "metrics": {"brightness": 0.072, "sharpness_index": 0.148, "noise": 0.71, "text_height": 7.0, "skew_angle": 0.0},
  "issues": ["too_dark", "dark"],
  "retake": true,
  "advice": "사진이 너무 어둡습니다. 밝은 곳에서 다시 찍어 주세요.",
  "steps": ["denoise_auto", "stretch"]
}
```

- `issues`: `dark`(어두움), `too_dark`(너무 어두움), `blurry`(흔들림), `noisy`(노이즈), `too_far`(글자가 너무 작음), `skewed`(기울어짐)
- `retake`: 전처리로 해결하기 어려운 사진이면 `true`. 이때 **AI 4는 `advice`를 음성으로 읽어 주면 됩니다.** OCR은 그래도 시도해서 결과를 같이 쓸 수 있습니다.
- `steps`: 아래 `preprocess`에 그대로 넘길 추천 전처리

### 1-2. 전처리

```python
def preprocess(image: np.ndarray, steps: list[str] | None = None, ocr=None) -> np.ndarray:
    ...
```

- 입력: OpenCV로 읽은 이미지 (BGR, `cv2.imread` 결과 그대로)
- 출력: 같은 형식의 이미지 (BGR). 회전 보정을 하면 가로세로 크기가 바뀔 수 있습니다.
- `steps`: 적용할 기법 이름 목록 (예: `["denoise_auto", "stretch"]`). `None`이면 `assess_quality`가 추천한 조합
- `ocr`: (선택) 2번의 `ocr` 함수. 회전 보정에서 어느 방향이 잘 읽히는지 확인하는 데 씁니다. 없으면 확실한 경우에만 돌립니다.

### 1-3. 한 번에 처리 (모듈 연결용)

```python
from pipeline import run_pipeline
result = run_pipeline(image, ocr)
```

출력 예시 (`results/ai2_week2/easyocr/sample_output/*.json`에 실제 예시가 있습니다. PaddleOCR 예시는 Colab 실행 후 `results/ai2_week2/paddle/sample_output/`에 생깁니다):

```json
{
  "quality": {"retake": false, "advice": "", "issues": [], "metrics": {...}},
  "preprocess": {"steps": [], "rotation": 0},
  "ocr": [{"text": "2027.04.14 까지", "box": [[...]], "score": 0.98}],
  "elapsed_sec": 21.3
}
```

- `ocr`의 `box` 좌표는 **전처리(회전·확대)를 거친 사진 기준**입니다. 원본 사진 위에 표시하려면 `preprocess.rotation`(시계 방향 각도)을 참고하세요.
- **AI 3:** `result["ocr"]`가 2번 OCR 출력과 같은 형식이라 `extract(result["ocr"], user_allergens)`에 그대로 넣으면 됩니다.
- **AI 4:** `result["quality"]["retake"]`가 `true`면 `result["quality"]["advice"]`를 먼저 안내합니다.

## 2. OCR (AI 1) — `ocr/`

```python
def ocr(image: np.ndarray) -> list[dict]:
    ...
```

출력 예시:

```json
[
  {"text": "유통기한 2026.12.15", "box": [[10, 20], [200, 20], [200, 50], [10, 50]], "score": 0.94},
  {"text": "밀, 대두, 우유 함유", "box": [[10, 60], [180, 60], [180, 85], [10, 85]], "score": 0.88}
]
```

- `box`: 글자 영역의 네 꼭짓점 좌표 (왼쪽 위부터 시계 방향)
- `score`: 인식 신뢰도 (0~1)

## 3. 정보 추출 (AI 3) — `extraction/`

```python
def extract(ocr_result: list[dict], user_allergens: list[str]) -> dict:
    ...
```

출력 예시:

```json
{
  "product_name": "새우맛 스낵",
  "expiry": {"raw": "2026.12.15", "date": "2026-12-15", "days_left": 40, "expired": false},
  "allergens": {"found": ["밀", "대두", "우유"], "warning": ["우유"]},
  "cautions": ["같은 제조시설에서 땅콩을 사용한 제품과 같이 제조"],
  "priority": ["allergens", "expiry", "product_name", "cautions"]
}
```

- `warning`: 사용자의 알레르기 목록과 겹치는 성분
- `priority`: 음성으로 읽어줄 순서

## 4. TTS (AI 4) — `tts/`

```python
def tts(info: dict) -> bytes:
    ...
```

- 입력: `extract()`의 출력
- 출력: 음성 파일 데이터 (mp3 또는 wav)
