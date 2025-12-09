# PHM 데이터셋을 활용한 장비 고장 분류 프로젝트

## 프로젝트 개요

본 프로젝트는 PHM (Prognostics and Health Management) 데이터셋을 활용하여 장비에서 발생하는 고장 유형을 분류하는 딥러닝 모델을 구현합니다. CNN-LSTM 하이브리드 아키텍처를 사용하여 시계열 센서 데이터로부터 공간적 및 시간적 특징을 추출하고 고장 유형을 예측합니다.

## 문제 정의

### 목표
- 장비의 센서 데이터를 분석하여 발생 가능한 고장 유형을 사전에 분류
- 예측 정확도를 높여 사전 예방적 유지보수(Predictive Maintenance) 가능

### 주요 과제
1. 다변량 시계열 센서 데이터 처리
2. 불균형한 고장 유형 데이터 처리
3. 시간적 패턴과 공간적 특징의 동시 학습

## 데이터셋 정보

### PHM 데이터셋 (Prognostics and Health Management)

PHM 데이터셋은 NASA나 IEEE PHM Conference에서 제공하는 산업 장비의 센서 데이터로, 주로 다음과 같은 특징을 가집니다:

#### 데이터 구조
- **센서 종류**: 온도, 압력, 진동, 회전속도 등 다수의 센서
- **시계열 데이터**: 시간에 따라 측정된 연속적인 센서 값
- **고장 유형**: Normal, Fault Type 1, Fault Type 2, ... 등 여러 종류의 고장 라벨

#### 일반적인 데이터 형태
```
timestamp, sensor_1, sensor_2, sensor_3, ..., sensor_n, fault_label
0,         23.5,     1.2,      45.3,     ..., 0.89,     0 (Normal)
1,         23.7,     1.3,      45.1,     ..., 0.91,     0 (Normal)
...
1000,      28.2,     2.1,      52.3,     ..., 1.45,     1 (Fault Type 1)
```

### PHM 데이터셋 소스

**PHM Society Data Challenge**
- 연례 데이터 챌린지에서 제공하는 산업 장비 센서 데이터
- 다양한 고장 유형과 정상 상태 데이터 포함
- 시계열 다변량 센서 데이터 (온도, 압력, 진동, 회전속도 등)
- URL: https://phmsociety.org/phm_competition/

## 접근 방식

### 1. 데이터 전처리

#### 타겟 변수 처리
- One-hot encoding을 통한 고장 유형 라벨 변환
- 예: `[0, 1, 0, 0]` → Fault Type 2

#### 입력 데이터 정규화
- Min-Max Scaling 또는 Standard Scaling
- 센서 간 스케일 차이를 줄여 학습 효율성 향상

#### 슬라이딩 윈도우 (Sliding Window)
- 시계열 데이터를 고정 길이 윈도우로 분할
- 예: 100 타임스텝의 센서 데이터를 하나의 샘플로 사용
- 윈도우 크기와 stride 설정으로 데이터 증강 효과

```python
# 예시: 윈도우 크기 100, stride 50
# 원본: [0, 1, 2, 3, ..., 999]
# 샘플 1: [0:100]
# 샘플 2: [50:150]
# 샘플 3: [100:200]
```

### 2. 모델 아키텍처: CNN-LSTM

#### CNN (Convolutional Neural Network)
- **역할**: 공간적 특징 추출
- 각 타임스텝에서 여러 센서 간의 상관관계 파악
- 1D Conv 레이어를 통해 지역적 패턴 인식

#### LSTM (Long Short-Term Memory)
- **역할**: 시간적 특징 추출
- 시계열 데이터의 장기 의존성(Long-term dependency) 학습
- 과거 패턴이 미래 고장에 미치는 영향 모델링

#### 통합 아키텍처
```
입력 데이터 (batch, timesteps, features)
    ↓
1D CNN 레이어 (공간적 특징 추출)
    ↓
LSTM 레이어 (시간적 특징 추출)
    ↓
Dense 레이어 (분류)
    ↓
Softmax 출력 (고장 유형 확률)
```

### 3. 학습 전략

- **손실 함수**: Categorical Crossentropy
- **최적화**: Adam Optimizer
- **평가 지표**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **클래스 불균형 처리**: Class weights, SMOTE, 또는 Focal Loss 고려

## 프로젝트 구조

```
samsung_project/
├── README.md                          # 프로젝트 설명서
├── requirements.txt                   # Python 패키지 의존성
├── data/                              # 데이터 디렉토리
│   └── README.md                      # 데이터 다운로드 안내
├── notebooks/                         # Jupyter 노트북
│   ├── example_fault_classification.ipynb   # 실습용 예시 코드
│   └── solution_fault_classification.ipynb  # 완전한 답안 코드
└── models/                            # 학습된 모델 저장 디렉토리
```

## 환경 설정

### Conda 환경 생성

```bash
# Conda 환경 생성
conda create -n pj_fc python=3.9 -y

# 환경 활성화
conda activate pj_fc

# 필요 패키지 설치
pip install -r requirements.txt
```

### Jupyter Notebook 실행

```bash
# Jupyter notebook 설치 및 실행
pip install jupyter notebook
python -m ipykernel install --user --name pj_fc --display-name pj_fc
```

## 참고 자료

### GitHub 레포지토리

1. **LSTM for Predictive Maintenance (Microsoft Azure)**
   - https://github.com/Azure/lstms_for_predictive_maintenance
   - Azure의 LSTM 기반 예측 유지보수 예제
   - PHM 데이터셋 활용 사례

2. **Time Series Classification with Deep Learning**
   - https://github.com/hfawaz/dl-4-tsc
   - 시계열 분류를 위한 다양한 딥러닝 아키텍처 구현
   - CNN, LSTM, ResNet 등 포함

3. **Predictive Maintenance using Deep Learning**
   - https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM
   - LSTM을 활용한 예측 유지보수 튜토리얼
   - 데이터 전처리부터 모델 배포까지

### 학습 자료

- **Keras Time Series Tutorial**: https://www.tensorflow.org/tutorials/structured_data/time_series
- **LSTM for Time Series**: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
- **CNN-LSTM Architecture**: https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

## 실습 가이드

### 실습 노트북 설명

1. **example_fault_classification.ipynb** (실습용)
   - 핵심 코드 구조와 함께 주석으로 힌트 제공
   - 빈칸을 채우는 방식이 아닌, 스스로 구현할 수 있도록 가이드
   - 각 단계별 설명과 기대 결과 포함

2. **solution_fault_classification.ipynb** (답안)
   - 완전히 구현된 코드
   - 상세한 주석과 설명
   - 결과 분석 및 시각화 포함

### 학습 순서

1. README.md를 통해 문제와 데이터 이해
2. example 노트북으로 직접 구현 시도
3. 막히는 부분은 힌트 참고
4. solution 노트북으로 비교 및 학습

## 평가 기준

- 데이터 전처리의 적절성
- 모델 아키텍처의 이해도
- 학습 과정의 효율성
- 결과 분석 및 해석 능력

## 확장 가능한 개선 사항

1. **Attention Mechanism** 추가로 중요 시점 강조
2. **Transfer Learning** 사전 학습된 모델 활용
3. **Ensemble Methods** 여러 모델 조합
4. **Real-time Inference** 실시간 예측 시스템 구축
5. **Explainable AI** 모델 해석 가능성 향상 (SHAP, GradCAM)
