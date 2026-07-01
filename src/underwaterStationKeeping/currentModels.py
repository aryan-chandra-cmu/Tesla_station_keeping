from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CurrentModel:
    baseCurrentX: float = 0.2
    baseCurrentY: float = -0.1
    gustAmplitudeX: float = 0.12
    gustAmplitudeY: float = 0.08
    gustFrequencyX: float = 0.12
    gustFrequencyY: float = 0.08
    shearGain: float = 0.015

    def value(self, timeSec: float, positionX: float, positionY: float) -> np.ndarray:
        currentX = (
            self.baseCurrentX
            + self.gustAmplitudeX * np.sin(self.gustFrequencyX * timeSec)
            + self.shearGain * positionY
        )
        currentY = (
            self.baseCurrentY
            + self.gustAmplitudeY * np.cos(self.gustFrequencyY * timeSec)
            - self.shearGain * positionX
        )
        return np.array([currentX, currentY], dtype=float)


def makeCurrentModel(scenarioName: str) -> CurrentModel:
    if scenarioName == "mild":
        return CurrentModel(
            baseCurrentX=0.12,
            baseCurrentY=-0.05,
            gustAmplitudeX=0.04,
            gustAmplitudeY=0.03,
            shearGain=0.006,
        )
    if scenarioName == "strong":
        return CurrentModel(
            baseCurrentX=0.35,
            baseCurrentY=-0.18,
            gustAmplitudeX=0.18,
            gustAmplitudeY=0.14,
            shearGain=0.02,
        )
    if scenarioName == "reversing":
        return CurrentModel(
            baseCurrentX=0.0,
            baseCurrentY=0.0,
            gustAmplitudeX=0.35,
            gustAmplitudeY=0.28,
            gustFrequencyX=0.06,
            gustFrequencyY=0.045,
            shearGain=0.01,
        )
    raise ValueError(f"unknown scenarioName: {scenarioName}")
