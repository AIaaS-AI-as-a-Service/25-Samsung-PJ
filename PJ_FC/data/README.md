# 데이터 디렉토리

## 데이터 다운로드 안내

이 프로젝트에서는 **PHM (Prognostics and Health Management) 데이터셋**을 사용합니다.

### NASA Turbofan Engine Degradation (C-MAPSS) 데이터셋 ✓ 다운로드 완료

#### 데이터셋 정보
- **이름**: C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **소스**: NASA Prognostics Center of Excellence
- **다운로드**: https://phm-datasets.s3.amazonaws.com/NASA/
- **특징**: 항공기 터보팬 엔진의 Run-to-Failure 시뮬레이션 데이터
- **데이터 형태**: 시계열 다변량 센서 데이터 (21개 센서)
- **운영 설정**: 3개 (고도, 속도, 스로틀 등)
- **고장 유형**:
  - 0: Normal (RUL > 100 cycles)
  - 1: Early Degradation (50 < RUL ≤ 100)
  - 2: Advanced Degradation (20 < RUL ≤ 50)
  - 3: Critical (RUL ≤ 20)
- **파일 형식**: TXT (공백 구분)

#### 포함된 데이터셋
- **FD001**: 1개 운영 조건, 1개 고장 모드 (HPC Degradation) - **현재 사용 중**
- **FD002**: 6개 운영 조건, 1개 고장 모드
- **FD003**: 1개 운영 조건, 2개 고장 모드
- **FD004**: 6개 운영 조건, 2개 고장 모드

#### 데이터 통계
- **학습 데이터**: 20,631개 샘플, 100개 엔진
- **테스트 데이터**: 13,096개 샘플, 100개 엔진
- **센서 개수**: 21개
- **운영 설정**: 3개

### 데이터 저장 위치

```
data/
├── raw/                                           # 원본 데이터
│   └── 6. Turbofan Engine Degradation Simulation Data Set/
│       ├── train_FD001.txt                        # 학습 데이터 (원본)
│       ├── test_FD001.txt                         # 테스트 데이터 (원본)
│       └── RUL_FD001.txt                          # Remaining Useful Life
├── processed/                                      # 전처리된 데이터
│   ├── train_FD001_processed.csv                  # 학습 데이터 (전처리 완료)
│   └── test_FD001_processed.csv                   # 테스트 데이터 (전처리 완료)
├── prepare_phm_data.py                            # 데이터 전처리 스크립트
└── README.md                                      # 이 파일
```

### 데이터 사용 방법

#### 방법 1: 전처리된 데이터 사용 (추천)

```python
import pandas as pd

# 전처리된 데이터 로드
df = pd.read_csv('../data/processed/train_FD001_processed.csv')

print(df.head())
print(f"Shape: {df.shape}")
print(f"\n고장 유형 분포:\n{df['fault_type'].value_counts().sort_index()}")
```

#### 방법 2: 원본 데이터부터 직접 전처리

```bash
cd data
python prepare_phm_data.py
```

## 데이터 구조

### 전처리된 CSV 파일 구조

| 컬럼명 | 설명 | 예시 값 |
|--------|------|---------|
| unit_id | 엔진 번호 (1~100) | 1 |
| time_cycles | 운영 사이클 (시간) | 1, 2, 3, ... |
| op_setting_1 | 운영 설정 1 (고도 등) | -0.0007 |
| op_setting_2 | 운영 설정 2 (속도 등) | -0.0004 |
| op_setting_3 | 운영 설정 3 (스로틀 등) | 100.0 |
| sensor_1 ~ sensor_21 | 센서 측정값 (온도, 압력, 진동 등) | 518.67, 641.82, ... |
| RUL | Remaining Useful Life (남은 수명) | 191 |
| fault_type | 고장 유형 (0: Normal, 1: Early, 2: Advanced, 3: Critical) | 0 |

### 데이터 예시

```csv
unit_id,time_cycles,op_setting_1,op_setting_2,op_setting_3,sensor_1,sensor_2,...,sensor_21,RUL,fault_type
1,1,-0.0007,-0.0004,100.0,518.67,641.82,...,23.4190,191,0
1,2,0.0019,-0.0003,100.0,518.67,642.15,...,23.4236,190,0
1,3,-0.0043,0.0003,100.0,518.67,642.35,...,23.3442,189,0
...
1,190,0.0023,-0.0001,100.0,518.67,643.21,...,23.5123,1,3
1,191,-0.0011,0.0002,100.0,518.67,644.02,...,23.4987,0,3
```
