# 단일 PCB 이미지 결함 탐지

YOLO26n은 입력 영상에서 **결함 박스 좌표·유형·점수**를 함께 예측합니다. 화면의 `PCB 이미지 검사`는 JPEG·PNG 등의 검사 이미지 한 장을 받아 추론합니다. 탐지 박스는 표시 레이어에 그리며, `탐지 박스`를 끄면 입력 영상을 볼 수 있습니다. CSV·XLSX 일괄 가져오기에도 같은 모델을 사용합니다. 기준 영상은 영상 비교에 선택적으로 첨부합니다.

## 학습 및 평가 데이터

| 분할 | 결함 검사 영상 | 기준 영상 대조군 | 용도 |
| --- | ---: | ---: | --- |
| [Train](../data/splits/train.csv) | 1,000 | 250 | YOLO 가중치 파인튜닝 |
| [Validation](../data/splits/validation.csv) | 112 | 28 | 체크포인트와 검출 점수 임계값 선택 |
| [Test](../data/splits/test.csv) | 220 | 55 | 최종 위치·유형 평가 |

보드 그룹 단위의 기존 분할을 유지합니다. 원본 `x1 y1 x2 y2 유형` 주석을 이미지 크기로 정규화한 YOLO 형식으로 변환합니다. 단선, 단락, 마우스바이트, 돌기, 잉여 구리, 핀홀을 학습합니다. 각 분할에서 4장마다 선택한 기준 영상은 불량 주석이 없는 독립 입력 사례로 사용합니다. YOLO는 각 영상 한 장을 입력으로 받습니다. [원본 데이터와 주석](https://github.com/tangsanli5201/DeepPCB) · [YOLO 학습 데이터 형식](https://docs.ultralytics.com/datasets/detect/)

## 재현 명령

Python 3.11–3.13 가상 환경에서 `python -m pip install ".[detector]"`를 실행하고 [원본 ZIP 받기](PCB_REPRODUCTION.md#실행)에 따라 `data/raw/DeepPCB.zip`을 준비합니다. GPU 학습 환경에서 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe scripts/train_detector.py data/raw/DeepPCB.zip --weights yolo26n.pt
.\.venv\Scripts\python.exe scripts/evaluate_detector.py
.\.venv\Scripts\python.exe run.py
```

YOLO26n 초기 가중치를 로컬에 두었다면 `--weights`에 해당 경로를 지정합니다. 학습은 25 epoch, 640px, 배치 12, 시드 42를 사용합니다. 데이터 변환만 실행할 때는 `--prepare-only`를 붙입니다. 체크포인트는 `outputs/detector/pcb_yolo26n/weights/best.pt`에 저장됩니다. 재학습을 별도로 보관하려면 학습 시 `--name`을 지정하고, 평가 시 `--run`에 해당 폴더를 전달합니다. `evaluate_detector.py`는 검증에서 점수 임계값을 고른 뒤 테스트에서 위치와 유형을 평가합니다. `run.py`는 저장된 임계값과 체크포인트를 로드해 CPU로 추론합니다. [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)

## 결과 해석

RTX 2080 Ti에서 25 epoch를 실행했습니다. 검증에서 체크포인트와 점수 임계값 **0.60**을 정한 뒤, 보드 그룹이 분리된 Test 220장과 기준 영상 대조군 55장을 평가했습니다. 테스트에는 실제 라벨 박스 1,402개가 있습니다.

| 지표 | 결과 |
| --- | ---: |
| 위치 재현율 · IoU 0.33 | 83.7% (1,173/1,402) |
| 위치와 유형 재현율 · IoU 0.33 | 82.7% (1,159/1,402) |
| 위치와 유형 정밀도 / F1 · IoU 0.33 | 94.3% / 0.881 |
| 표준 mAP50 / mAP50–95 | 0.920 / 0.591 |
| 기준 영상 대조군에서 1건 이상 검출 | 3/55장 |

같은 220장과 IoU 0.33 기준의 기존 영상처리·랜덤 포레스트 조합의 위치 재현율은 10.3%였습니다. 유형별로는 단락의 위치·유형 재현율이 가장 낮아 219개 중 139개(63.5%)를 맞혔습니다. 결함 박스는 1대1 대응해 집계했습니다. 템플릿 대조군은 원본 데이터의 기준 이미지입니다. 실제 AOI 도입 전에는 기업의 무표시 정상·불량 영상에서 제품 및 로트를 분리해 미탐과 오탐을 검증해야 합니다.

공개한 [평가 JSON](benchmarks/yolo26n-deeppcb.json)에서 분할 해시, 가중치 SHA-256, 검증 임계값 비교, 유형별 TP·FP·FN을 읽을 수 있습니다. [epoch별 학습 기록](benchmarks/yolo26n-deeppcb-training.csv)도 함께 제공합니다. 재실행 시 평가 JSON은 `outputs/detector/pcb_yolo26n/detector_metrics.json`에 생성됩니다.

## 실제 검사 예시

브라우저에서 640×640 검사 파일 `12100003_test.jpg` 한 장을 업로드했습니다. 모델은 핀홀, 잉여 구리, 마우스바이트, 단락 박스 4개를 출력했습니다. 정답 주석과 비교했을 때 4개는 위치와 유형이 맞았고, 단선 3개는 놓쳤습니다. 테스트 과정에서 담당자 판정에 단선을 추가하고 엑셀의 AI 불량유형과 최종 불량유형 두 열에 결과가 각각 유지되는 것을 확인했습니다. [검사 화면·좌표·엑셀 예시](https://app.notion.com/p/422985e401128180bc17d7214656f402)

DeepPCB 데이터는 연구용이고 본 패키지도 연구·시연용입니다. 기업 적용이나 재배포 범위는 해당 데이터 및 모델 이용 조건을 확인합니다. [DeepPCB 이용 안내](https://github.com/tangsanli5201/DeepPCB#notification) · [Ultralytics 라이선스](https://docs.ultralytics.com/help/license/)
