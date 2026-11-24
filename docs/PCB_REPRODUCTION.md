# DeepPCB 모델 학습·평가 재현

DeepPCB의 기준·검사 이미지에서 추출한 특징으로 로지스틱 회귀와 랜덤 포레스트를 직접 학습했습니다. 원본 데이터는 ZIP 한 개로 유지하고, [분할 CSV](../data/splits/)에 각 이미지 쌍의 소속과 상대 경로를 기록합니다. 시연 앱은 학습에서 선택한 모델을 읽어 추론합니다.

## 데이터 구성

각 쌍은 `*_test.jpg` 검사 영상, `*_temp.jpg` 기준 영상, `*.txt` 불량 유형·위치 주석으로 구성됩니다. 분할 CSV는 샘플 ID, 보드 그룹, ZIP 내부의 이미지·주석 경로, 업스트림 목록 이름을 기록합니다. [`PCBData/`](https://github.com/tangsanli5201/DeepPCB/tree/08e98c4db5922613fb97176eb3d6497d48260cb1/PCBData) 기준 상대 경로입니다.

| 분할 | 영상 쌍 · 그룹 | 사용처 |
| --- | --- | --- |
| [Train](../data/splits/train.csv) | 1,000쌍 · 7개 | 후보 특징으로 두 분류기 학습 |
| [Validation](../data/splits/validation.csv) | 112쌍 · `group12000`, `group12300` | 유형·위치 F1로 모델 및 임계값 선택 |
| [Test](../data/splits/test.csv) | 220쌍 · `group12100`, `group90100` | 선택 확정 후 불량 재현율·유형별 TP/FP/FN 평가 |

원본 `trainval.txt`는 모두 학습에 씁니다. 원본 `test.txt`에서 학습에 등장한 보드 그룹의 168쌍을 제외하고, 나머지 332쌍을 검증과 테스트에 보드 그룹 단위로 나눕니다. 재현 스크립트는 중복 샘플 ID와 분할 간 보드 그룹 겹침을 검사합니다. 검증·테스트에는 기준 영상을 자기 자신과 비교한 대조군도 각 영상 쌍당 한 건씩 추가합니다. 독립적으로 촬영한 정상 제품의 오탐률을 구하려면 추가 데이터가 필요합니다.

```text
data/
  raw/DeepPCB.zip            공식 원본 ZIP · 로컬에서 내려받기
  splits/train.csv           학습에 쓰는 정확한 샘플 목록
  splits/validation.csv      모델과 임계값을 선택하는 목록
  splits/test.csv            최종 평가 목록
scripts/
  pcb_splits.py              분할 생성·검사
  prepare_pcb_demo.py        분할, 학습, 평가 실행
outputs/reproduction/        로컬 산출물
  training.json              학습 설정, 환경, 선정 모델과 검증 점수
  model.joblib               시연 앱에서 읽는 선정 모델
  candidate_models.joblib    비교한 두 모델과 검증 단계의 선택 기록
  benchmark.json             분리된 테스트 결과와 추론 시간
```

## 실행

[README 빠른 시작](../README.md#빠른-시작)으로 Python 환경을 설치한 뒤, 저장소 루트에서 실행합니다. 학습과 평가는 같은 ZIP과 분할 CSV를 사용해야 합니다.

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
curl.exe -fL https://codeload.github.com/tangsanli5201/DeepPCB/zip/08e98c4db5922613fb97176eb3d6497d48260cb1 -o data/raw/DeepPCB.zip
Get-FileHash data/raw/DeepPCB.zip -Algorithm SHA256

# 1. 공식 목록에서 분할 CSV를 재생성해 저장소의 data/splits와 비교
.\.venv\Scripts\python.exe scripts/prepare_pcb_demo.py data/raw/DeepPCB.zip --stage split

# 2. 커밋된 Train·Validation CSV로 모델과 임계값을 확정
.\.venv\Scripts\python.exe scripts/prepare_pcb_demo.py data/raw/DeepPCB.zip --stage train

# 3. 저장된 모델과 임계값으로 Test CSV의 결과를 평가
.\.venv\Scripts\python.exe scripts/prepare_pcb_demo.py data/raw/DeepPCB.zip --stage evaluate
```

macOS·Linux에서는 `mkdir -p data/raw`, `curl`, `.venv/bin/python`을 사용합니다. 고정 커밋에서 검증한 ZIP의 SHA-256은 `e2380725de34ad8e897a51d83a6ced863c95f1c0d2ed25ac9e810903840c9cd4`입니다. v0.1.0은 같은 커밋을 `master` 이름으로 내려받은 ZIP으로 학습했으며, 압축 파일의 경로 접두사가 달라 ZIP 해시가 다릅니다.

`--stage all`로 학습과 평가를 연속 실행할 수 있습니다. 방금 만든 CSV를 직접 쓰려면 두 단계에 모두 `--splits-dir outputs/reproduction/splits`를 붙입니다. `training.json`의 ZIP·분할 해시와 저장된 모델 선택 기록이 일치할 때 평가를 진행합니다. 정렬 영상의 연구용 시연 데이터도 로컬에서 재생성하려면 평가 단계에 `--demo`를 붙인 뒤 `run.py --assets outputs/reproduction`으로 실행합니다.

## 학습 방식과 비교 기준

1. 정렬된 640×640 기준·검사 영상을 각각 이진화합니다. 차이가 연결된 후보 영역마다 크기, 형태, 차이의 방향, 주변 패턴 밀도 등 **15개 특징**을 계산합니다.
2. 후보를 원본 주석에 IoU 0.19 이상으로 대응시켜 불량 유형을 붙이고, 나머지 후보에는 배경 라벨을 붙입니다. Train에서 추출된 후보는 배경 100,502개와 불량 3,475개입니다.
3. 표준화 후 다항 로지스틱 회귀와 랜덤 포레스트를 학습합니다. 랜덤 포레스트는 트리 36개, 최대 깊이 11, 리프 최소 표본 7개이며 시드는 42입니다.
4. Validation에서 후보 임계값 `0.45, 0.60, 0.75, 0.90`을 비교합니다. 유형·위치 F1이 더 높은 랜덤 포레스트를 선택해 임계값 `0.45`를 확정했습니다. Test에서는 이 값을 사용해 이미지 단위 탐지와 유형·위치 매칭을 평가합니다.

랜덤 포레스트의 검증 유형·위치 F1은 **0.382**, 테스트 유형·위치 F1은 **0.432**, 테스트 불량 영상 재현율은 **204/220(92.7%)**입니다. 유형·위치 F1은 같은 종류의 불량을 IoU 0.19 이상으로 1대1 매칭한 값입니다. 이미지 단위 재현율은 불량이 있는 영상에서 후보가 하나라도 표시된 비율입니다. [수치와 평가 한계](PCB_BENCHMARK.md)

고정 ZIP으로 재실행해 두 모델의 검증·테스트 지표가 v0.1.0과 같은 것을 확인했습니다. 개발 PC 환경은 Python 3.13.5, OpenCV 4.12.0, NumPy 2.2.6, scikit-learn 1.6.1, joblib 1.4.2, Pillow 11.1.0입니다. 재생성된 `model.joblib`은 배포 모델과 바이트 단위로 일치했고, SHA-256은 `4438714196b329b4f04880b220e0d95ba087a956a2d0ec483c762c3e5b34e93b`입니다. CPU 시간은 실행 환경에 따라 달라집니다.

원본 저자는 DeepPCB의 사용 범위를 연구용으로 명시합니다. 원본은 공식 저장소에서 받아 로컬에서 학습하며, 새로 생성한 원본 이미지 복사본과 평가 산출물도 로컬에 저장됩니다. [DeepPCB 사용 조건](https://github.com/tangsanli5201/DeepPCB/blob/08e98c4db5922613fb97176eb3d6497d48260cb1/README.md#notification)
