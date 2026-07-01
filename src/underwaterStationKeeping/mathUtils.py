from __future__ import annotations

import math

import numpy as np


def wrapAngle(angleRad: float) -> float:
    return (angleRad + math.pi) % (2.0 * math.pi) - math.pi


def rotationMatrix(yawRad: float) -> np.ndarray:
    cosYaw = math.cos(yawRad)
    sinYaw = math.sin(yawRad)
    return np.array([[cosYaw, -sinYaw], [sinYaw, cosYaw]], dtype=float)


def clampVector(vectorValue: np.ndarray, vectorLimit: float) -> np.ndarray:
    vectorNorm = float(np.linalg.norm(vectorValue))
    if vectorNorm <= vectorLimit or vectorNorm <= 1e-12:
        return vectorValue
    return vectorValue * (vectorLimit / vectorNorm)


def clampScalar(value: float, lowerValue: float, upperValue: float) -> float:
    return min(max(value, lowerValue), upperValue)
