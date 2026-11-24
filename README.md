# AIMPACT PCB 품질검사 MVP

**PCB 검사 이미지 한 장에서 결함 위치와 유형을 찾아 표시하고, 담당자 최종판정을 기록하는 AOI 작업대입니다.**

## 시연 이미지

| 시연에 넣을 이미지 | YOLO 검사 결과 |
| --- | --- |
| [입력 원본 JPG · 12100003_test.jpg](data/pcb_demo/images/12100003_test.jpg) | [박스·유형이 표시된 화면과 엑셀 예시 · Notion](https://app.notion.com/p/422985e401128180bc17d7214656f402) |

**PCB 이미지 검사**에서 위 JPG를 선택하면 같은 사례를 재현할 수 있습니다. [모델 학습·평가 결과](docs/PCB_DETECTOR.md)

## 빠른 시작

Python 3.11–3.13이 필요합니다. 이 소스를 클론한 경우 [YOLO 학습·평가 안내](docs/PCB_DETECTOR.md)에 따라 파인튜닝 가중치를 준비합니다. 연구용 로컬 실행 패키지에는 학습된 가중치가 포함되어 있습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ".[detector]"
.\.venv\Scripts\python.exe run.py
```

macOS·Linux에서는 `.\.venv\Scripts\python.exe` 대신 `.venv/bin/python`을 사용합니다. 브라우저에서 **http://127.0.0.1:8765**를 열면 DeepPCB 예제 12건이 바로 로드됩니다. 앱은 CPU로 동작하고 기록은 로컬 SQLite에 저장합니다.

## 90초 시연

1. **PCB 이미지 검사**에서 위 시연 JPG를 엽니다. YOLO가 출력한 박스와 결함 유형을 확인하고 박스를 클릭해 확대합니다.
2. **판정 검토**에서 유형이나 정상·불량 판정을 수정하고 저장합니다. AI 초안과 담당자 판정 차수가 기록에 남습니다.
3. **품질 현황**에서 반복 불량과 AI·담당자 일치 건수를 확인합니다. **검사 이력**에서 제품·기간·유형으로 조회하고 엑셀로 내려받습니다.

업로드를 시연하려면 **데이터 가져오기 → 업로드 예제 ZIP**을 사용하세요. CSV 또는 XLSX의 열을 제품코드·검사일시·검사 영상 파일명에 연결하고, 협력사별 매핑을 저장해 다음 업로드에서 재사용할 수 있습니다.

## 모델 학습·평가 재현

DeepPCB의 원본 불량 박스와 6종 유형 주석으로 **YOLO26n을 직접 파인튜닝**했습니다. [학습 영상 1,000장](data/splits/train.csv), [검증 영상 112장](data/splits/validation.csv), [테스트 영상 220장](data/splits/test.csv)은 서로 다른 보드 그룹에서 왔습니다. 각 분할에 템플릿 대조군을 추가했습니다. [YOLO 데이터 구조·학습·평가 재현](docs/PCB_DETECTOR.md)

## 시연 데이터와 모델

DeepPCB 사례와 학습 모델은 연구용입니다. YOLO는 검사 영상 한 장에서 결함 위치, 유형과 점수를 함께 출력합니다. 초기 담당자 판정 4건은 시연용 기록입니다. 기준 영상 비교 방식은 `python run.py --detector paired`로 재현할 수 있습니다.

[YOLO 실행·학습·평가](docs/PCB_DETECTOR.md) · [기존 모델 학습 재현](docs/PCB_REPRODUCTION.md) · [데이터 출처](data/pcb_demo/SOURCE.md)
