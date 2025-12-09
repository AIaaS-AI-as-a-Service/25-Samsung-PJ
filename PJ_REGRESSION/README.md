# XJTU-SY 베어링 RUL 회귀 실습 프로젝트

이 저장소는 XJTU-SY 베어링(run-to-failure) 데이터셋을 이용하여
**Remaining Useful Life (RUL)** 회귀 문제를 다루기 위한 교육용 예제 프로젝트입니다.
다변량 시계열(진동 신호에서 추출한 특징)을 입력으로 받아 LSTM 기반 모델로 RUL을 예측하는
end-to-end 파이프라인을 제공합니다.

## 1. 데이터셋 개요 (XJTU-SY)

- 제공 기관: Xi'an Jiaotong University (XJTU) & Sumyoung Technology (SY)
- 내용: 3가지 운전 조건 하에서 측정된 15개 베어링의 **run-to-failure 진동 시계열**
- 형식:
  - 각 베어링(run)에 대해 여러 개의 CSV 파일이 존재
  - 각 CSV 파일:
    - 1열: 수평(horizontal) 진동
    - 2열: 수직(vertical) 진동
  - 샘플링 간격: 약 1분마다 1.28초 구간의 진동 신호를 측정

---

## 2. 디렉토리 구조

이 프로젝트는 다음과 같은 최소 구조를 가정합니다.

```text
samsung-rul-xjtu-sy/
├─ README.md
├─ requirements.txt
├─ data/
│  └─ raw/
│       └─ XJTU-SY_Bearing_Datasets/
├─ src/
│  ├─ __init__.py
│  ├─ data_utils.py      # 데이터 로딩, 특징 추출, RUL 타깃 생성, 슬라이딩 윈도우
│  └─ models.py          # LSTM 기반 RUL 회귀 모델
└─ notebooks/
   ├─ 01_rul_regression_sample.ipynb   # 실습용 (주석/힌트 포함)
   └─ 01_rul_regression_answer.ipynb   # 정답용 (완전 구현)
```

- `data/raw/XJTU-SY_Bearing_Datasets` 아래에는 압축을 해제한 원본 데이터셋 전체를 그대로 두는 것을 권장합니다.
- `src/`에는 파이프라인에서 재사용되는 공통 코드(데이터 유틸, 모델 정의)가 들어갑니다.
- `notebooks/`에는 교육생이 직접 실행/수정할 주피터 노트북이 들어갑니다.

---

## 3. RUL 정의 및 전처리 개념

1. **특징 추출 (feature extraction)**
   - 각 CSV 파일의 진동 신호(2채널)에 대해 간단한 시간 영역 특징을 계산합니다.
     - 예: RMS, Peak-to-Peak, Kurtosis 등
   - 결과적으로 각 CSV는 길이가 `F`인 특징 벡터로 요약되고,
     각 베어링(run)은 `(N, F)` 형태의 시계열(특징 시퀀스)로 표현됩니다.

2. **RUL 타깃 생성**
   - 한 run에 CSV가 `N`개 있을 때, 시점 `t`(0부터 시작)에서의 RUL은
     - `RUL(t) = N - t` 로 정의합니다. (초기에는 큰 값, 고장에 가까울수록 1에 수렴)
   - 너무 초기에 RUL이 과도하게 큰 값을 갖지 않도록, `max_rul` 상한으로 **클리핑(capping)** 할 수 있습니다.
     - 예: `max_rul = 125` → `RUL(t) = min(N - t, 125)`

3. **슬라이딩 윈도우**
   - LSTM 입력은 일정 길이의 시퀀스가 필요하므로,
     - 길이 `window_size`의 슬라이딩 윈도우를 `(N, F)` 시계열에 적용합니다.
   - 한 윈도우가 `[t, t+1, ..., t+window_size-1]` 구간을 포함한다면,
     - 이 윈도우의 라벨은 마지막 시점의 RUL: `RUL(t + window_size - 1)` 로 둡니다.

이 모든 로직은 `src/data_utils.py`에 구현되어 있으며,
노트북에서는 해당 함수들을 호출하여 바로 학습 데이터를 만들 수 있습니다.

---

## 4. 노트북 실행 흐름

1. `notebooks/01_rul_regression_sample.ipynb`
   - 데이터셋 로딩 → 슬라이딩 윈도우 생성 → LSTM 모델 정의 → 학습/평가 순으로 진행됩니다.
   - 중요한 단계마다 주석과 힌트를 추가해 두었습니다.
2. `notebooks/01_rul_regression_answer.ipynb`
   - sample 노트북과 동일한 구조이지만,
     모든 코드가 완전히 채워진 상태의 “참고용/정답” 버전입니다.

두 노트북 모두 기본적으로 CPU에서 동작하도록 작성되어 있으며,
가능한 경우 자동으로 GPU(CUDA)를 사용하게 됩니다.