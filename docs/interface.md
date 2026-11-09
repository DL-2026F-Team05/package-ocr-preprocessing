# 모듈 간 인터페이스 (초안)

각 모듈이 무엇을 받고 무엇을 넘겨주는지 정한 약속입니다. 이 형식만 맞추면 서로의 작업을 기다리지 않고 동시에 개발할 수 있습니다.

> 초안입니다. 바꾸고 싶은 부분은 팀에서 합의한 뒤 이 문서를 먼저 고쳐주세요.

## 전체 흐름

```
사진 → preprocess() → ocr() → extract() → tts()
```

## 1. 전처리 (AI 2) — `preprocessing/`

```python
def preprocess(image: np.ndarray, steps: list[str] | None = None) -> np.ndarray:
    ...
```

- 입력: OpenCV로 읽은 이미지 (BGR, `cv2.imread` 결과 그대로)
- 출력: 같은 형식의 이미지 (BGR)
- `steps`: 적용할 기법 이름 목록 (예: `["clahe", "sharpen"]`), `None`이면 기본 조합

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
