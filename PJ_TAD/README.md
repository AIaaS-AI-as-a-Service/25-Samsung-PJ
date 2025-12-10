# SWaT 데이터셋 기반 시계열 이상 탐지 프로젝트 (Time Series Anomaly Detection)

---

## 프로젝트 개요

본 프로젝트는 SWaT (Secure Water Treatment) 데이터셋을 활용하여 산업 제어 시스템(ICS)의 **사이버 공격을 탐지**하는 딥러닝 기반 이상 탐지 실습 프로젝트입니다.

두 가지 시계열 딥러닝 아키텍처를 비교 실습합니다:

- **TCN-AE** (Temporal Convolutional Network Autoencoder)
- **LSTM-AE** (Long Short-Term Memory Autoencoder)

- 다변량 시계열 센서 데이터 전처리 (51개 센서)
- 비지도 학습 기반 이상 탐지 (정상 데이터만으로 학습)
- Reconstruction Error 기반 Threshold 설정
- PyTorch 기반 시계열 딥러닝 모델 설계
- TCN vs LSTM 아키텍처 비교 분석

---

## 문제 정의 (Problem Definition)

### 목표

- 정상 운영 데이터로 학습하여 비정상(공격) 패턴 탐지
- Reconstruction Error 기반 이상 점수(Anomaly Score) 산출
- Precision, Recall, F1-Score, AUC-ROC 기반 탐지 성능 평가
- TCN-AE와 LSTM-AE 아키텍처 성능 비교

### 주요 과제

- 51개 센서의 다변량 시계열 데이터 처리
- 1초 단위 고빈도 데이터의 효율적 다운샘플링
- 결측치 및 저분산/고상관 컬럼 처리
- 슬라이딩 윈도우 기반 시퀀스 생성
- 정상 데이터 분포 기반 최적 Threshold 결정

---

## 데이터셋 정보

### SWaT (Secure Water Treatment) 데이터셋

본 프로젝트에서는 싱가포르 iTrust 연구소에서 제공하는 **SWaT** 데이터셋을 사용합니다.

#### 데이터셋 특징

| 항목 | 내용 |
|------|------|
| **소스** | Singapore University of Technology and Design (SUTD) iTrust |
| **도메인** | 수처리 시설 (Water Treatment Testbed) |
| **센서 수** | 51개 (유량계, 수위계, 펌프, 밸브 등) |
| **정상 데이터** | 7일간 정상 운영 기록 (~495,000 샘플) |
| **공격 데이터** | 4일간 36개 공격 시나리오 포함 (~449,919 샘플) |
| **샘플링 주기** | 1초 |

#### 센서 유형

| 접두어 | 센서 유형 | 설명 |
|--------|----------|------|
| **FIT** | Flow Indicator Transmitter | 유량계 |
| **LIT** | Level Indicator Transmitter | 수위계 |
| **AIT** | Analyzer Indicator Transmitter | 분석기 |
| **PIT** | Pressure Indicator Transmitter | 압력계 |
| **DPIT** | Differential Pressure Indicator Transmitter | 차압계 |
| **MV** | Motorized Valve | 전동 밸브 |
| **P** | Pump | 펌프 |
| **UV** | UV Dechlorinator | UV 탈염소기 |

#### 데이터 구조

```
Timestamp, FIT101, LIT101, MV101, P101, P102, AIT201, ..., Normal/Attack
28/12/2015 10:00:00 AM, 2.427, 522.84, 2, 2, 1, 262.01, ..., Normal
28/12/2015 10:00:01 AM, 2.446, 522.88, 2, 2, 1, 262.01, ..., Normal
...
2/1/2016 1:41:09 PM, 2.502, 524.22, 2, 2, 1, 168.89, ..., Attack
```

#### 레이블 분포

- **Normal**: 정상 운영 상태
- **Attack**: 36가지 공격 시나리오 (단일/다중 포인트 공격)

---

## 접근 방식 (Approach)

### 1. 데이터 전처리

#### 전처리 파이프라인

1. **Timestamp 컬럼 제거**: 모델 입력에서 제외
2. **다운샘플링**: 1초 → 10초 간격으로 축소 (1/10)
3. **결측치 처리**: Forward/Backward Fill + 평균값 대체
4. **저분산 컬럼 제거**: 분산 < 0.01인 컬럼 삭제
5. **고상관 컬럼 제거**: 상관계수 > 0.95인 중복 컬럼 삭제
6. **정규화**: MinMax Scaling (0-1 범위)
7. **시퀀스 생성**: Sliding Window (window_size=100, stride=10)

#### 데이터 분할

| 용도 | 데이터 소스 | 레이블 | 비율 |
|------|-------------|--------|------|
| **Train** | normal.csv | 정상만 | 80% |
| **Validation** | normal.csv | 정상만 | 20% |
| **Test** | attack.csv | 정상+공격 혼합 | 전체 |

### 2. 모델 아키텍처

#### TCN-AE (Temporal Convolutional Network Autoencoder)

**특징:**
- Causal Convolution: 미래 정보를 사용하지 않는 인과적 컨볼루션
- Dilated Convolution: 지수적으로 증가하는 dilation (1, 2, 4, ...)
- Residual Connection: 깊은 네트워크의 그래디언트 흐름 개선
- 병렬 처리 가능: CNN 기반으로 학습 속도 빠름

```
입력 데이터 (batch, seq_len, features)
    ↓
Encoder: TCN Blocks (dilation: 1→2→4)
    ↓
Latent Space (batch, latent_dim, seq_len)
    ↓
Decoder: TCN Blocks (dilation: 4→2→1)
    ↓
출력 (batch, seq_len, features) - Reconstructed
```

#### LSTM-AE (Long Short-Term Memory Autoencoder)

**특징:**
- Gate Mechanism: Forget/Input/Output 게이트로 장기 의존성 학습
- Hidden State: 시퀀스의 시간적 패턴 압축
- RepeatVector: 잠재 벡터를 시퀀스 길이만큼 반복하여 디코딩
- 순차 처리: 시계열 특성에 적합하나 학습 속도 느림

```
입력 데이터 (batch, seq_len, features)
    ↓
Encoder LSTM (features → hidden_dim → latent_dim)
    ↓
Latent Vector (batch, latent_dim)
    ↓
RepeatVector (batch, seq_len, latent_dim)
    ↓
Decoder LSTM (latent_dim → hidden_dim → features)
    ↓
출력 (batch, seq_len, features) - Reconstructed
```

### 3. 이상 탐지 원리

1. **정상 데이터로만 학습**: 모델이 "정상 패턴"을 재구성하는 능력 학습
2. **Reconstruction Error 계산**: 입력과 재구성 출력 간의 MSE
3. **Threshold 결정**: 검증 데이터(정상)의 99 percentile
4. **이상 판정**: Reconstruction Error ≥ Threshold → 이상(Attack)

### 4. 학습 전략

- **손실 함수**: MSE (Mean Squared Error)
- **최적화**: Adam Optimizer (lr=1e-3, weight_decay=1e-5)
- **Early Stopping**: Validation Loss 기준 patience=10
- **Gradient Clipping**: max_norm=1.0 (LSTM 그래디언트 폭발 방지)
- **평가 지표**: Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix

---

## TCN-AE vs LSTM-AE 비교

| 특성 | TCN-AE | LSTM-AE |
|------|--------|---------|
| **아키텍처** | CNN 기반 | RNN 기반 |
| **병렬화** | 높음 | 낮음 (순차 처리) |
| **학습 속도** | 빠름 | 느림 |
| **장기 의존성** | Dilation으로 확장 | Gate 메커니즘 |
| **메모리 사용** | 적음 | 많음 |
| **그래디언트 흐름** | 안정적 | 소실/폭발 가능 |
| **해석 가능성** | Receptive Field 명확 | Hidden State 분석 가능 |

---

## 프로젝트 구조

```
TAD/
├── notebooks/                       # Jupyter 노트북
│   ├── tcn_ae_practice.ipynb        # TCN-AE 실습용 코드
│   ├── tcn_ae_answer.ipynb          # TCN-AE 완전한 답안 코드
│   ├── lstm_ae_practice.ipynb       # LSTM-AE 실습용 코드
│   └── lstm_ae_answer.ipynb         # LSTM-AE 완전한 답안 코드
```

---
## 환경 설정

### Conda 환경 생성

```bash
# Conda 환경 생성
conda create -n tad python=3.10 -y

# 환경 활성화
conda activate tad

# 필요 패키지 설치
pip install -r requirements.txt
```

### Jupyter Notebook 실행

```bash
# Jupyter kernel 등록
python -m ipykernel install --user --name tad --display-name "TAD"

# Jupyter notebook 실행
jupyter notebook
```

---

## 실습 노트북 설명

### 1. TCN-AE 노트북

#### tcn_ae_practice.ipynb (실습용)

- 핵심 코드 구조와 함께 주석으로 힌트 제공
- 빈칸을 채우며 직접 구현하는 방식
- 각 단계별 설명과 기대 결과 포함

#### tcn_ae_answer.ipynb (답안)

- 완전히 구현된 코드
- 상세한 주석과 설명
- 학습 결과 및 시각화 포함

### 2. LSTM-AE 노트북

#### lstm_ae_practice.ipynb (실습용)

- TCN-AE와 동일한 구조로 LSTM-AE 구현 실습
- 모델 아키텍처 부분만 다르게 구현

#### lstm_ae_answer.ipynb (답안)

- LSTM Encoder/Decoder 구현


### 학습 순서

1. README.md를 통해 문제와 데이터 이해
2. tcn_ae_practice.ipynb로 TCN-AE 직접 구현 시도
3. 막히는 부분은 answer 노트북 참고
4. lstm_ae_practice.ipynb로 LSTM-AE 구현
5. 두 모델의 성능 비교 및 분석

---

## 하이퍼파라미터

### 공통 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| downsample_rate | 10 | 다운샘플링 비율 |
| window_size | 100 | 시퀀스 길이 |
| stride | 10 | 윈도우 이동 간격 |
| batch_size | 64 | 배치 크기 |
| learning_rate | 1e-3 | 학습률 |
| epochs | 100 | 최대 에폭 |
| patience | 10 | Early Stopping |
| threshold_percentile | 99 | 이상 판정 임계값 |

### TCN-AE 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| encoder_dims | [64, 128, 256] | 인코더 채널 수 |
| decoder_dims | [256, 128, 64] | 디코더 채널 수 |
| kernel_size | 3 | 컨볼루션 커널 크기 |
| dropout | 0.2 | 드롭아웃 비율 |

### LSTM-AE 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| hidden_dim | 128 | LSTM 은닉층 차원 |
| latent_dim | 64 | 잠재 공간 차원 |
| num_layers | 2 | LSTM 레이어 수 |
| dropout | 0.2 | 드롭아웃 비율 |
| bidirectional | False | 양방향 LSTM 여부 |

---

## 평가 기준

- 데이터 전처리의 적절성
- 모델 아키텍처의 이해도
- 학습 과정의 효율성 (Early Stopping, Gradient Clipping)
- Threshold 설정의 합리성
- 결과 분석 및 해석 능력
- TCN vs LSTM 비교 분석

---

## 학습 목표

- 다변량 시계열 센서 데이터 전처리 방법
- Sliding Window 기반 시퀀스 생성
- PyTorch 기반 Autoencoder 모델 설계
- TCN (Temporal Convolutional Network) 아키텍처 이해
- LSTM Autoencoder 아키텍처 이해
- 비지도 학습 기반 이상 탐지 원리
- Reconstruction Error 기반 Threshold 설정
- 시계열 모델 성능 평가 및 시각화
- CNN vs RNN 기반 시계열 모델 비교

---

## 참고 자료 (Reference)

### GitHub 레포지토리

1. **SWaT Dataset Official**
   - https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

2. **TCN Autoencoder Implementation**
   - https://github.com/MarkusThill/bioma-tcn-ae

3. **LSTM Autoencoder Implementation**
   - https://github.com/dltkddn0525/LSTM-autoencoder
  

### 학습 자료

- **TCN Autoencoder Paper**:  https://www.gm.fh-koeln.de/ciopwebpub/Thill20a.d/bioma2020-tcn.pdf
- **LSTM Autoencoder Paper**: https://arxiv.org/pdf/1607.00148

---

## 확장 아이디어

1. **Variational Autoencoder (VAE)** 적용
   - LSTM-VAE, TCN-VAE로 확률적 이상 탐지

2. **Transformer-based AE** 구현
   - Self-Attention 기반 시계열 모델링

3. **GAN 기반 이상 탐지**
   - Discriminator의 판별 능력 활용

4. **Ensemble Methods**
   - TCN-AE + LSTM-AE 앙상블
