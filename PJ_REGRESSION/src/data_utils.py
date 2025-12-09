import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import math


def list_runs(raw_root: str) -> Dict[str, List[str]]:
    """
    XJTU-SY 베어링 데이터셋의 run(조건+베어링)을 탐색하여
    각 run에 속한 CSV 파일 경로 리스트를 반환합니다.

    Parameters
    ----------
    raw_root : str
        XJTU-SY_Bearing_Datasets 디렉토리 경로.

    Returns
    -------
    Dict[str, List[str]]
        key: run_id (예: "35Hz_12kN/Bearing1_1")
        value: 해당 run에 속한 CSV 파일 경로 리스트 (시간 순서대로 정렬).
    """
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"raw_root가 존재하지 않습니다: {raw_root}")

    runs: Dict[str, List[str]] = {}
    # 조건 디렉토리 (예: 35Hz_12kN, 37.5Hz_11kN, 40Hz_10kN)
    for cond_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        # 베어링(run) 디렉토리 (예: Bearing1_1, Bearing1_2, ...)
        for bearing_dir in sorted([p for p in cond_dir.iterdir() if p.is_dir()]):
            run_id = f"{cond_dir.name}/{bearing_dir.name}"
            csv_paths = sorted(str(p) for p in bearing_dir.glob("*.csv"))
            if csv_paths:
                runs[run_id] = csv_paths

    if not runs:
        raise RuntimeError(f"{raw_root} 아래에서 CSV 파일을 찾지 못했습니다.")

    return runs


def extract_features_from_signal(sig_2ch: np.ndarray) -> np.ndarray:
    """
    2채널 진동 신호로부터 간단한 시간 영역 특징을 추출합니다.

    입력 배열은 (L, 2) 형태를 가정하며,
    각 채널에 대해 다음 특징을 계산합니다.

    - RMS (root mean square)
    - Peak-to-Peak (최댓값 - 최솟값)
    - Kurtosis (첨도, Fisher 정의)

    최종 출력 벡터 길이는 6 (채널당 3개 × 2채널)입니다.

    Parameters
    ----------
    sig_2ch : np.ndarray
        shape (L, C) 의 배열. 최소 2개 채널이 있다고 가정합니다.

    Returns
    -------
    np.ndarray
        shape (6,) 의 특징 벡터 (float32).
    """
    if sig_2ch.ndim != 2 or sig_2ch.shape[1] < 2:
        raise ValueError(f"입력 신호 shape가 예상과 다릅니다: {sig_2ch.shape}")

    feats = []
    num_channels = 2  # XJTU-SY: horizontal, vertical
    for ch in range(num_channels):
        x = sig_2ch[:, ch].astype(np.float32)
        rms = float(np.sqrt(np.mean(x ** 2)))
        p2p = float(np.max(x) - np.min(x))

        # kurtosis는 상수 배열에서 NaN을 반환할 수 있으므로 방어 코드 추가
        k_raw = kurtosis(x, fisher=True, bias=False)
        k = float(k_raw)
        if not math.isfinite(k):  # NaN 또는 ±inf이면
            k = 0.0

        feats.extend([rms, p2p, k])

    return np.asarray(feats, dtype=np.float32)


def load_csv_as_features(csv_path: str) -> np.ndarray:
    """
    단일 CSV 파일을 읽어 2채널 진동 신호를 로드하고, 특징 벡터로 변환합니다.

    - CSV 첫 줄에 헤더(예: 'Horizontal_vibration_signals', 'Vertical_vibration_signals')
      이 있다고 가정하고, pandas에 헤더를 맡깁니다.
    - 숫자형 컬럼만 선택하여 2채널 신호로 사용합니다.

    Parameters
    ----------
    csv_path : str
        CSV 파일 경로.

    Returns
    -------
    np.ndarray
        shape (F,) 의 특징 벡터.
    """
    # 헤더를 자동 인식하도록 header=None 제거
    df = pd.read_csv(csv_path)

    # 숫자형 컬럼만 선택 (문자열 헤더/설명 컬럼 제거)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        raise ValueError(
            f"숫자형 컬럼이 2개 미만입니다. 파일 형식을 확인하세요: {csv_path}\n"
            f"컬럼들: {list(df.columns)}"
        )

    # 앞의 2개 숫자 컬럼만 사용 (수평/수직 진동)
    data = numeric_df.iloc[:, :2].to_numpy(dtype=np.float32)

    return extract_features_from_signal(data)


def load_run_features(csv_paths: List[str]) -> np.ndarray:
    """
    하나의 run에 속한 모든 CSV 파일에 대해 특징을 계산하여
    (N, F) 형태의 시계열 특징 배열을 생성합니다.

    Parameters
    ----------
    csv_paths : List[str]
        run을 구성하는 CSV 파일 경로 리스트 (시간 순서).

    Returns
    -------
    np.ndarray
        shape (N, F) 의 특징 시계열.
    """
    feats = [load_csv_as_features(p) for p in csv_paths]
    return np.stack(feats, axis=0)


def generate_rul_target(length: int, max_rul: Optional[float] = None) -> np.ndarray:
    """
    run 길이가 length일 때, 시점 t(0부터 시작)의 RUL을 N - t로 정의합니다.

    예를 들어 length=5이면,
    RUL 시퀀스는 [5, 4, 3, 2, 1] 이 됩니다.

    max_rul이 주어지면, 해당 값을 초과하는 구간은 max_rul로 클리핑합니다.

    Parameters
    ----------
    length : int
        run 내 시계열 길이 N.
    max_rul : float, optional
        RUL 상한값. None이면 클리핑을 하지 않습니다.

    Returns
    -------
    np.ndarray
        shape (N,) 의 RUL 벡터 (float32).
    """
    rul = np.arange(length, 0, -1, dtype=np.float32)  # [N, N-1, ..., 1]
    if max_rul is not None:
        rul = np.minimum(rul, float(max_rul))
    return rul


def create_sliding_windows(
    X_run: np.ndarray,
    rul_run: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    run 단위 특징 시계열 (N, F)와 RUL 벡터(N,)에서 슬라이딩 윈도우를 생성합니다.

    한 윈도우는 길이 window_size의 구간 [t, t+1, ..., t+window_size-1]를 포함하며,
    이 윈도우의 라벨은 마지막 시점의 RUL, 즉 RUL(t + window_size - 1)을 사용합니다.

    Parameters
    ----------
    X_run : np.ndarray
        shape (N, F) 특징 시계열.
    rul_run : np.ndarray
        shape (N,) RUL 시퀀스.
    window_size : int
        윈도우 길이.
    stride : int
        윈도우 시작 인덱스 간격.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X_windows: shape (M, window_size, F)
        y_windows: shape (M,)
    """
    N, F = X_run.shape
    if len(rul_run) != N:
        raise ValueError("X_run과 rul_run의 길이가 일치하지 않습니다.")

    if N < window_size:
        return (
            np.empty((0, window_size, F), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    windows = []
    labels = []
    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        windows.append(X_run[start:end, :])
        labels.append(rul_run[end - 1])

    Xw = np.stack(windows, axis=0).astype(np.float32)
    yw = np.asarray(labels, dtype=np.float32)
    return Xw, yw


def _default_split(run_ids: List[str]) -> Dict[str, List[str]]:
    """
    run_id 리스트를 간단히 train/val/test로 나누는 기본 규칙입니다.

    - 전체의 60%: train
    - 20%: val
    - 20%: test

    교육용 샘플이므로 매우 단순한 기준을 사용합니다.
    필요한 경우 사용자가 직접 split_config를 만들어 전달하면 됩니다.
    """
    run_ids = sorted(run_ids)
    n = len(run_ids)
    if n < 3:
        # 너무 적은 경우에는 그냥 앞에서부터 train, 나머지 test로 둠
        return {
            "train": run_ids,
            "val": [],
            "test": [],
        }

    n_train = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    n_test = max(1, n - n_train - n_val)

    train_ids = run_ids[:n_train]
    val_ids = run_ids[n_train:n_train + n_val]
    test_ids = run_ids[n_train + n_val:n_train + n_val + n_test]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def build_dataset(
    raw_root: str,
    window_size: int = 20,
    stride: int = 1,
    max_rul: Optional[float] = 125.0,
    split_config: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray]]:
    """
    XJTU-SY 원시 데이터 디렉토리에서 바로 학습/검증/테스트용 윈도우 데이터를 생성합니다.

    전체 파이프라인:
    1) run 탐색
    2) run별 특징 시계열 (N, F) 생성
    3) run별 RUL 타깃 시퀀스 (N,) 생성
    4) 슬라이딩 윈도우 적용 → (M, window_size, F), (M,)

    Parameters
    ----------
    raw_root : str
        XJTU-SY_Bearing_Datasets 폴더 경로.
    window_size : int, optional
        슬라이딩 윈도우 길이.
    stride : int, optional
        윈도우 stride.
    max_rul : float, optional
        RUL 상한값. None이면 클리핑 없음.
    split_config : dict, optional
        사용자가 직접 정의한 run 분할 정보.
        예:
        {
            "train": ["35Hz_12kN/Bearing1_1", ...],
            "val": ["35Hz_12kN/Bearing1_5"],
            "test": ["40Hz_10kN/Bearing3_2", ...],
        }
        None이면 내부의 기본 규칙(_default_split)을 사용합니다.

    Returns
    -------
    (train_X, train_y), (val_X, val_y), (test_X, test_y)
        각 X: shape (M_split, window_size, F)
        각 y: shape (M_split,)
    """
    runs = list_runs(raw_root)
    all_run_ids = sorted(runs.keys())

    if split_config is None:
        split = _default_split(all_run_ids)
    else:
        split = split_config

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for split_name, run_list in split.items():
        if not run_list:
            results[split_name] = (
                np.empty((0, window_size, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
            continue

        X_parts = []
        y_parts = []

        for run_id in run_list:
            if run_id not in runs:
                raise KeyError(f"split_config에 정의된 run_id를 찾을 수 없습니다: {run_id}")
            csv_paths = runs[run_id]
            X_run = load_run_features(csv_paths)
            rul_run = generate_rul_target(len(X_run), max_rul=max_rul)
            Xw, yw = create_sliding_windows(X_run, rul_run, window_size, stride)
            if Xw.size == 0:
                continue
            X_parts.append(Xw)
            y_parts.append(yw)

        if not X_parts:
            # 해당 split에서 유효한 윈도우를 하나도 만들지 못한 경우
            results[split_name] = (
                np.empty((0, window_size, 0), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        else:
            X_all = np.concatenate(X_parts, axis=0)
            y_all = np.concatenate(y_parts, axis=0)
            results[split_name] = (X_all, y_all)

    train = results.get("train", (np.empty((0, window_size, 0), dtype=np.float32),
                                   np.empty((0,), dtype=np.float32)))
    val = results.get("val", (np.empty((0, window_size, 0), dtype=np.float32),
                               np.empty((0,), dtype=np.float32)))
    test = results.get("test", (np.empty((0, window_size, 0), dtype=np.float32),
                                 np.empty((0,), dtype=np.float32)))

    return train, val, test
