
## 프로젝트 개요

**순수 Python 스크립트**로 Vision AI 파이프라인의 핵심 개념을 빠르게 체득한다.
프레임워크 없이 직접 구현하면서 "어떻게 동작하는지" 설명할 수 있는 수준이 목표.

---

## 기술 스택

- **언어:** Python 3.11+
- **Vision:** OpenCV (`opencv-python-headless`)
- **ML:** PyTorch + torchvision (사전학습 모델만 사용, 학습 없음)
- **수치 연산:** NumPy
- **시각화:** Matplotlib (결과 확인용)
- **의존성 관리:** `requirements.txt` + `venv`

```bash
# 환경 세팅
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install opencv-python-headless torch torchvision numpy matplotlib
```

---

## 프로젝트 구조

```
mithril-vision/
├── CLAUDE.md
├── requirements.txt
│
├── 01_basics/
│   ├── read_video.py          # 영상 읽기 + 프레임 저장
│   ├── draw_shapes.py         # bounding box 그리기 연습
│   └── resize_crop.py         # 전처리 기초
│
├── 02_detection/
│   ├── detect_image.py        # 단일 이미지 객체 탐지
│   ├── detect_video.py        # 영상 실시간 탐지
│   └── model_loader.py        # 모델 로드 유틸 (재사용)
│
├── 03_anomaly/
│   ├── roi_guard.py           # ROI 침입 감지
│   ├── motion_detector.py     # 배경 차분 기반 움직임 감지
│   └── event_logger.py        # 이상 이벤트 로그 저장
│
├── 04_pipeline/
│   └── run_pipeline.py        # 01~03을 연결한 전체 파이프라인
│
└── data/
    ├── sample.mp4             # 테스트 영상 (웹캠 or 유튜브 다운로드)
    └── output/                # 결과 이미지/로그 저장
```

---

## 단계별 구현 계획 (5일)

### Day 1 — OpenCV 기초
```
목표: 영상을 읽고, 프레임을 추출하고, 도형을 그릴 수 있다
```
- [ ] `read_video.py`: `cv2.VideoCapture`로 영상 읽기, 프레임을 이미지로 저장
- [ ] `draw_shapes.py`: 프레임에 rectangle, text, circle 그리기
- [ ] `resize_crop.py`: 해상도 조정, ROI(관심 영역) 크롭

**핵심 함수:**
```python
cap = cv2.VideoCapture("data/sample.mp4")  # 0이면 웹캠
ret, frame = cap.read()                    # 프레임 읽기
cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.putText(frame, "label", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
cv2.imwrite("output/frame.jpg", frame)
```

---

### Day 2 — 객체 탐지 모델 연결
```
목표: torchvision 사전학습 모델로 영상에서 사람을 탐지한다
```
- [ ] `model_loader.py`: Faster R-CNN 모델 로드 + 추론 함수
- [ ] `detect_image.py`: 단일 이미지에서 탐지 결과 시각화
- [ ] `detect_video.py`: 영상 전체를 프레임 단위로 처리 후 결과 영상 저장

**핵심 코드:**
```python
import torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 전처리
from torchvision.transforms import functional as F
tensor = F.to_tensor(frame)  # (H,W,C) BGR → (C,H,W) float

# 추론
with torch.no_grad():
    predictions = model([tensor])[0]
# predictions['boxes'], predictions['labels'], predictions['scores']
# COCO label 1 = person
```

---

### Day 3 — 이상 탐지 로직
```
목표: 감지된 객체 위치 기반으로 위험 상황을 판단하는 규칙 엔진
```
- [ ] `roi_guard.py`: 특정 구역(폴리곤 or 사각형)에 사람이 침입하면 알림
- [ ] `motion_detector.py`: `cv2.BackgroundSubtractorMOG2`로 움직임 감지
- [ ] `event_logger.py`: 이벤트 발생 시 타임스탬프 + 프레임 번호를 CSV로 저장

**핵심 코드:**
```python
# ROI 침입 감지
def is_in_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2  # 중심점
    rx1, ry1, rx2, ry2 = roi
    return rx1 < cx < rx2 and ry1 < cy < ry2

# 배경 차분
fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(frame)
motion_pixels = cv2.countNonZero(fgmask)
if motion_pixels > THRESHOLD:
    print("움직임 감지!")
```

---

### Day 4 — 전체 파이프라인 연결
```
목표: 영상 입력 → 탐지 → 이상 판단 → 로그 저장까지 한 번에 실행
```
- [ ] `run_pipeline.py`: 위 모듈들을 import해서 하나의 루프로 연결
- [ ] 처리 속도(FPS) 측정 및 출력
- [ ] 결과 영상(`output/result.mp4`) 저장

**구조:**
```python
# run_pipeline.py 의사코드
cap = VideoCapture(source)
model = load_model()
logger = EventLogger("output/events.csv")

while cap.isOpened():
    frame = cap.read()
    detections = detect(model, frame)          # Day 2
    events = check_roi(detections, ROI_ZONE)   # Day 3
    if events:
        logger.log(frame_idx, events)          # Day 3
    annotated = draw_results(frame, detections, events)  # Day 1
    writer.write(annotated)
```

---

### Day 5 — 정리 + 면접 스토리
```
목표: 코드를 "내가 설계한 시스템"으로 설명할 수 있게 정리
```
- [ ] `README.md`: 파이프라인 흐름도 + 실행 방법
- [ ] 각 모듈의 설계 결정 이유를 주석으로 보강
- [ ] 성능 측정 결과 기록 (FPS, 탐지 정확도 체감)
- [ ] GitHub 푸시 + 결과 스크린샷 저장

---

## Claude Code에게 시킬 작업 예시

```bash
# 프로젝트 구조 생성
"이 CLAUDE.md 기반으로 폴더 구조와 빈 파일들을 생성해줘"

# Day 1
"01_basics/read_video.py 구현해줘.
 cv2.VideoCapture로 영상을 읽고 매 30프레임마다 data/output/에 저장하는 스크립트"

# Day 2
"02_detection/model_loader.py 구현해줘.
 torchvision fasterrcnn_resnet50_fpn을 로드하고
 numpy BGR 프레임을 받아서 person 탐지 결과만
 [{'bbox':[x1,y1,x2,y2], 'score':0.95}] 형태로 반환하는 detect() 함수"

# Day 3
"03_anomaly/roi_guard.py 구현해줘.
 detect() 결과 리스트와 roi 좌표를 받아서
 침입한 객체 목록을 반환하는 check_intrusion() 함수"

# Day 4
"04_pipeline/run_pipeline.py 구현해줘.
 위 모듈들을 연결해서 영상 전체를 처리하고
 결과 영상과 events.csv를 data/output/에 저장"

# 막힐 때
"detect_video.py에서 cv2.VideoWriter로 결과 영상 저장하는 부분만 구현해줘"
```

---

## 면접 연결 포인트

| 구현 내용 | 면접에서 이렇게 말하기 |
|---|---|
| `model_loader.py` | "torchvision 사전학습 모델로 inference 파이프라인을 직접 구성해봤습니다" |
| `roi_guard.py` | "산업 현장 침입 감지를 ROI 기반 규칙 엔진으로 구현했습니다" |
| `motion_detector.py` | "BackgroundSubtractor로 움직임을 감지하고 탐지 모델과 병렬로 활용했습니다" |
| FPS 측정 | "처리 속도를 직접 측정하며 엣지 환경 제약을 체감했습니다" |
| 모듈 분리 구조 | "각 컴포넌트를 독립적으로 교체 가능하도록 설계했습니다" |

---

## 참고 리소스

- **테스트 영상:** 웹캠(`0`) 또는 유튜브에서 공장/건설 영상 다운로드
- **COCO 클래스 목록:** label 1 = person, 전체 목록은 `torchvision.models.detection` 문서 참고
- **속도가 너무 느리면:** `fasterrcnn_mobilenet_v3_large_fpn` (더 가벼운 모델)로 교체

