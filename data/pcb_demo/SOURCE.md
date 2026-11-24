# DeepPCB 연구용 시연 자료

이 폴더의 JPEG는 Tang 등의 DeepPCB에서 가져온 640×640 검사 영상과 기준 영상입니다. [DeepPCB 저장소](https://github.com/tangsanli5201/DeepPCB) · [논문](https://arxiv.org/abs/1902.06197)

- 원본 Git 커밋 `08e98c4db5922613fb97176eb3d6497d48260cb1`
- 시연 원본 이미지 그룹 `group12100`, `group90100`
- 원본 데이터 설명에서 정한 사용 범위는 연구용입니다. [업스트림 라이선스](UPSTREAM_LICENSE.txt)
- `samples.json`의 협력사·로트·검사일은 시연용 메타데이터입니다. `-CTRL` 레코드는 기준 영상을 자기 자신과 비교하는 정상 대조군입니다.
- `model.joblib`은 원본 연구용 데이터를 사용해 이번 MVP에서 학습한 모델입니다. `benchmark.json`에는 분할 그룹, 원본 ZIP 해시와 평가 결과를 기록했습니다.
