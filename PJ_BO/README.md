# SECOM 반도체 불량 예측 · Bayesian Optimization 기반 딥러닝 최적화 프로젝트

---

## 프로젝트 개요

본 프로젝트는 SECOM 반도체 공정 센서 데이터를 활용하여  
제품의 **불량 여부(Fail / Pass)** 를 예측하는 딥러닝 모델을 구현하고,  
Optuna 기반 **Bayesian Optimization**으로 하이퍼파라미터를 자동 탐색하여  
모델 성능을 최적화하는 실습 프로젝트입니다.

단순 분류 모델 구현에 그치지 않고,

- 고차원 제조 데이터 처리 (원본 590차원, 전처리 후 562차원)
- 극심한 클래스 불균형 대응
- PyTorch 기반 딥러닝 모델 설계
- Optuna를 이용한 하이퍼파라미터 자동 최적화
- Threshold 조정을 통한 운영 성능 개선

까지 포함하는 **실전형 머신러닝 파이프라인 구축**이 목표입니다.

---

## 문제 정의 (Problem Definition)

### 목표

- 센서 데이터를 기반으로 장비 상태(양품 / 불량) 분류
- 불량(Fail) 검출 성능(F1, Recall) 극대화
- 수작업 튜닝이 아닌 Bayesian Optimization 기반 자동 최적화 수행
- 수율 관리 개선을 위한 데이터 기반 의사결정 지원

### 주요 과제

- 590차원 고차원 센서 데이터 (전처리 후 562차원) 처리
- 컬럼별 결측치 다수 존재
- 불량 비율 약 6.6% 수준의 극심한 클래스 불균형
- 하이퍼파라미터 설정에 따라 성능 변동 폭이 큼
- 고정 임계값(threshold = 0.5)의 비효율성

---

## 데이터셋 정보

### SECOM (SEmiCOnductor Manufacturing)

- 샘플 수: 1,567
- 원본 Feature 수: 590
- 전처리 후 사용 Feature 수: 562
- 타겟 컬럼: Pass / Fail
  - -1 → Pass → 0
  -  1 → Fail → 1
- 불량 비율: 약 6.64%

### 데이터 특성

- 고차원 센서 데이터
- 다수 결측치 존재
- 클래스 불균형 문제 (Fail ≪ Pass)

### 전처리 파이프라인 (Answer.ipynb 기준)

1. `Time`, `Pass/Fail` 컬럼 분리
2. 결측치 비율 50% 이상인 컬럼 제거
3. `SimpleImputer(strategy="mean")` 로 나머지 결측치 평균 대치
4. `StandardScaler` 로 모든 입력 피처 정규화
5. `train/val/test` 분할 (Stratified split)
   - Train: 64%
   - Validation: 16%
   - Test: 20%

전처리 후 Feature 수는 562개이며,  
Target은 0(Pass) / 1(Fail) 이진 분류 형태로 사용됩니다.

---

## 접근 방식 (Approach)

---

### 1. 데이터 전처리

- Time 컬럼 제거 (모델 입력에서 제외)
- 결측치 비율 ≥ 50% 컬럼 삭제
- 나머지 결측치는 SimpleImputer(mean)으로 대치
- StandardScaler 로 모든 피처 정규화
- train/val/test = 64/16/20 비율로 stratified 분할

---

### 2. 모델 아키텍처 : Flexible MLP (PyTorch)

Answer.ipynb 에서 구현된 `FlexibleMLP` 기반 분류 모델을 사용합니다.

#### 입력 및 흐름

- 입력 형태: (batch_size, num_features) = (batch_size, 562)

모델 구조 개념:

- 입력 데이터 (562차원)
  - ↓
- [Linear → BatchNorm1d → ReLU → Dropout] × N
  - (은닉층 개수 N과 각 레이어 차원은 Optuna로 탐색)
  - ↓
- Linear(마지막 dim → 1)
  - ↓
- Sigmoid(추론 시) → Fail / Pass 확률

#### 주요 특징

- 은닉층 개수(`n_layers`)와 각 레이어 뉴런 수(`hidden_dim_i`)를 Optuna로 자동 탐색
- `dropout_rate` 또한 하이퍼파라미터로 최적화
- 손실 함수: `BCEWithLogitsLoss(pos_weight=...)`
  - `pos_weight = (#Negative / #Positive)` 로 설정하여 Fail 클래스에 가중치 부여
- Optimizer: Adam

---

### 3. 학습 전략

- Validation F1-score 기준 Early Stopping 적용
  - Optuna Objective 안에서 epochs=100, patience=15
- 클래스 불균형을 고려한 `pos_weight` 설정
- 최종 평가는 test set에서만 수행
- Optuna 최적화 이후에는 Train + Validation 전체 데이터로 최종 모델 재학습

---

## Bayesian Optimization (Optuna)

### 탐색 대상 하이퍼파라미터 (Answer.ipynb 기준)

- 네트워크 구조
  - `n_layers`: 2 ~ 5
  - `hidden_dim_i`: 32 ~ 512 (step = 32)
- 정규화 및 일반화
  - `dropout_rate`: 0.1 ~ 0.5
  - (BatchNorm은 기본적으로 사용)
- 학습 관련
  - `lr` (learning rate): 1e-5 ~ 1e-2 (log scale)
  - `batch_size`: {16, 32, 64, 128}

### 최적화 목표

- Validation F1-score (특히 Fail 클래스 성능) 최대화

### Optuna 설정

- Sampler: `TPESampler(seed=42)`
- Study:
  - `direction="maximize"`
  - `study_name="secom_dnn_optimization"`
- Trial 횟수: `n_trials = 50`

---

## Baseline vs Optimized Model

### Baseline 모델 (Optuna 적용 전)

코드 상 기본 설정:

- Hidden layers: [256, 128, 64]
- Dropout: 0.3
- Batch size: 64
- Learning rate: 0.001
- Epoch: 150
- Loss: BCEWithLogitsLoss(pos_weight=...)
- Optimizer: Adam
- Train + Validation 전체 데이터(X_train_full)로 학습 후 Test 평가

### Bayesian Optimization 최적 모델

- `study.best_params` 기반으로 은닉층 개수, 각 레이어 크기, 드롭아웃, 학습률, 배치 크기 자동 구성
- Train + Validation 전체 데이터(X_train_full)로 최종 재학습
- 동일 Test Set에서 Baseline과 성능 비교

### 평가 지표

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

Answer.ipynb에서는 Baseline/최적화 모델의 각 지표를 비교하고,  
성능 향상률(%)까지 출력 및 시각화합니다.

---

## Threshold Tuning

고정된 threshold(0.5)가 항상 최적은 아니기 때문에,  
다양한 threshold를 탐색하여 F1-score 기준 최적 임계값을 찾는 실험을 수행합니다.

### 실험 내용 (Answer.ipynb 기준)

- Threshold 범위: 0.1 ~ 0.9
- Step: 0.05
- 각 임계값마다 다음 지표 계산:
  - F1-score
  - Precision
  - Recall
- F1-score가 최대가 되는 threshold를 선택

### 성과

- Fail 클래스에 대한 Recall/F1 개선
- Precision–Recall–F1 간 trade-off 시각적 분석
- 운영 환경(불량을 놓치면 안 되는 상황 등)에 맞는 threshold 선택 전략 실습

---

## 실습 노트북 설명

### Answer.ipynb

구성 흐름:

1. 데이터 로딩 (uci-secom.csv)
2. 데이터 탐색 및 기본 통계 출력
3. 결측치 및 스케일링을 포함한 전처리 파이프라인 구성
4. `FlexibleMLP` 모델 정의
5. Baseline 모델 학습 및 평가
6. Optuna 기반 Bayesian Optimization 수행
7. Best 하이퍼파라미터로 최종 모델 재학습
8. Threshold 튜닝 및 지표/그래프 시각화
9. Baseline vs Optimized 모델 비교 및 요약 출력

---

## What You Can Learn

이 프로젝트를 통해 다음 내용을 실습하고 이해할 수 있습니다.

- 고차원 센서 데이터 전처리 및 분석 방법
- 불균형 데이터에 대한 대응 전략 (pos_weight, threshold tuning 등)
- PyTorch 기반 MLP 모델 설계 및 학습 루프 구현
- Optuna를 활용한 딥러닝 하이퍼파라미터 최적화 실습
- Validation F1-score 기준 모델 선택 전략
- Threshold 기반 성능 튜닝 및 운영 관점의 평가 방법
- Baseline vs Optimized 모델 비교/분석 리포트 작성

---

## 참고 자료 (Reference)

### GitHub 레포지토리 (참고 예시)

- Optuna 공식 예제  
  https://github.com/optuna/optuna-examples

- PyTorch MLP + Optuna 하이퍼파라미터 튜닝 예제  
  https://github.com/nikitaprasad21/LLM-Cheat-Code/blob/main/PyTorch/Hyperparameter_Tuning_the_ANN_using_Optuna.ipynb

- SECOM 데이터 분석 및 불량 탐지 예제  
  https://github.com/sharmaroshan/SECOM-Detecting-Defected-Items

- UCI SECOM 제조 공정 예측 프로젝트 예시  
  https://github.com/Eason0227/Semiconductor-Manufacturing-Procees-Prediction

위 레포지토리들은 전처리 아이디어, 모델 구조, 불균형 처리, Optuna 활용 방식 등을 참고하는 용도로 활용할 수 있습니다.

---

## 확장 아이디어

- XGBoost / LightGBM / Random Forest 등 다른 모델과 성능 비교
- Feature Selection 또는 차원 축소(PCA 등) 적용 후 성능 변화 분석
- AutoML 도구와 연계하여 전체 파이프라인 자동화
- Explainable AI(XAI) 도입: SHAP / LIME으로 중요한 센서 피처 해석
- 모델 서빙 (REST API 또는 Batch Inference 파이프라인) 구현
- 실제 공정 로그(시간 정보 포함)를 반영한 시계열 모델 (1D-CNN / LSTM 등)로 확장

---
