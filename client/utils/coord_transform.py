"""
座標変換ユーティリティ
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """カメラ内部パラメータ"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
