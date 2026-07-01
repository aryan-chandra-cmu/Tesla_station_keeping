from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class BoxStationParams:
    massKg: float = 24.0
    linearDamping: float = 18.0
    maxForceN: float = 55.0
    maxForceRateNPerS: float = 110.0
    maxEpisodeSteps: int = 600
    timeStep: float = 0.05
    workspaceLimitM: float = 8.0
    stationHalfSizeX: float = 0.25
    stationHalfSizeY: float = 0.25
    stationHalfSizeZ: float = 0.20
    maxInitialOffsetM: float = 3.0
    maxInitialSpeedMps: float = 0.2
    maxCurrentMps: float = 0.45
    baseCurrentX: float = 0.18
    baseCurrentY: float = -0.10
    baseCurrentZ: float = 0.0
    gustAmplitudeX: float = 0.18
    gustAmplitudeY: float = 0.14
    gustAmplitudeZ: float = 0.05
    currentFrequencyX: float = 0.13
    currentFrequencyY: float = 0.09
    currentFrequencyZ: float = 0.17

    @classmethod
    def fromDict(cls, data: dict[str, Any] | None) -> "BoxStationParams":
        if data is None:
            return cls()
        validData = {
            fieldName: data[fieldName]
            for fieldName in cls.__dataclass_fields__
            if fieldName in data
        }
        return cls(**validData)


class UnderwaterBoxStationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params: BoxStationParams | None = None) -> None:
        super().__init__()
        self.params = params or BoxStationParams()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )
        self.targetPosition = np.zeros(3, dtype=float)
        self.position = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.currentVelocity = np.zeros(3, dtype=float)
        self.previousAction = np.zeros(3, dtype=float)
        self.elapsedSteps = 0
        self.currentPhase = np.zeros(3, dtype=float)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.elapsedSteps = 0
        self.targetPosition[:] = 0.0
        self.position = self.np_random.uniform(
            -self.params.maxInitialOffsetM,
            self.params.maxInitialOffsetM,
            size=3,
        )
        self.velocity = self.np_random.uniform(
            -self.params.maxInitialSpeedMps,
            self.params.maxInitialSpeedMps,
            size=3,
        )
        self.previousAction[:] = 0.0
        self.currentPhase = self.np_random.uniform(0.0, 2.0 * np.pi, size=3)
        self.currentVelocity = self._computeCurrent(0.0)
        return self._getObservation(), self._getInfo()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)

        maxActionDelta = (
            self.params.maxForceRateNPerS
            * self.params.timeStep
            / max(self.params.maxForceN, 1e-6)
        )
        actionDelta = np.clip(
            action - self.previousAction,
            -maxActionDelta,
            maxActionDelta,
        )
        limitedAction = self.previousAction + actionDelta
        forceWorld = limitedAction * self.params.maxForceN

        timeSec = self.elapsedSteps * self.params.timeStep
        self.currentVelocity = self._computeCurrent(timeSec)
        relativeVelocity = self.velocity - self.currentVelocity
        acceleration = (
            forceWorld - self.params.linearDamping * relativeVelocity
        ) / self.params.massKg

        self.velocity = self.velocity + acceleration * self.params.timeStep
        self.position = self.position + self.velocity * self.params.timeStep
        self.previousAction = limitedAction
        self.elapsedSteps += 1

        reward = self._computeReward(limitedAction, actionDelta)
        terminated = bool(
            np.linalg.norm(self.position - self.targetPosition)
            > self.params.workspaceLimitM
            or not np.all(np.isfinite(self.position))
            or not np.all(np.isfinite(self.velocity))
        )
        truncated = self.elapsedSteps >= self.params.maxEpisodeSteps
        return self._getObservation(), reward, terminated, truncated, self._getInfo()

    def _computeCurrent(self, timeSec: float) -> np.ndarray:
        currentX = self.params.baseCurrentX + self.params.gustAmplitudeX * np.sin(
            self.params.currentFrequencyX * timeSec + self.currentPhase[0]
        )
        currentY = self.params.baseCurrentY + self.params.gustAmplitudeY * np.cos(
            self.params.currentFrequencyY * timeSec + self.currentPhase[1]
        )
        currentZ = self.params.baseCurrentZ + self.params.gustAmplitudeZ * np.sin(
            self.params.currentFrequencyZ * timeSec + self.currentPhase[2]
        )
        currentVector = np.array([currentX, currentY, currentZ], dtype=float)
        return np.clip(
            currentVector,
            -self.params.maxCurrentMps,
            self.params.maxCurrentMps,
        )

    def _getObservation(self) -> np.ndarray:
        positionError = self.position - self.targetPosition
        boxHalfSize = self._boxHalfSize()
        normalizedPositionError = positionError / self.params.workspaceLimitM
        normalizedBoxDistance = positionError / boxHalfSize
        normalizedVelocity = self.velocity / 2.0
        normalizedCurrent = self.currentVelocity / self.params.maxCurrentMps
        observation = np.concatenate(
            [
                normalizedPositionError,
                normalizedVelocity,
                normalizedCurrent,
                self.previousAction,
            ]
        )
        observation[:3] = np.clip(normalizedPositionError, -2.0, 2.0)
        observation[3:6] = np.clip(normalizedVelocity, -2.0, 2.0)
        observation[6:9] = np.clip(normalizedCurrent, -2.0, 2.0)
        observation[9:12] = np.clip(self.previousAction, -1.0, 1.0)
        return observation.astype(np.float32)

    def _computeReward(self, action: np.ndarray, actionDelta: np.ndarray) -> float:
        positionError = self.position - self.targetPosition
        boxHalfSize = self._boxHalfSize()
        normalizedBoxDistance = positionError / boxHalfSize
        outsideDistance = np.maximum(np.abs(normalizedBoxDistance) - 1.0, 0.0)
        positionCost = float(np.linalg.norm(normalizedBoxDistance))
        outsideCost = float(np.linalg.norm(outsideDistance))
        velocityCost = float(np.linalg.norm(self.velocity))
        actionCost = float(np.sum(np.square(action)))
        actionRateCost = float(np.sum(np.square(actionDelta)))
        insideBox = bool(np.all(np.abs(positionError) <= boxHalfSize))
        slowEnough = velocityCost < 0.08

        reward = (
            -0.35 * positionCost
            -1.5 * outsideCost
            -0.25 * velocityCost
            -0.015 * actionCost
            -0.02 * actionRateCost
        )
        if insideBox:
            reward += 2.0
        if insideBox and slowEnough:
            reward += 3.0
        return float(reward)

    def _getInfo(self) -> dict[str, float | bool]:
        positionError = self.position - self.targetPosition
        boxHalfSize = self._boxHalfSize()
        insideBox = bool(np.all(np.abs(positionError) <= boxHalfSize))
        return {
            "positionErrorM": float(np.linalg.norm(positionError)),
            "speedMps": float(np.linalg.norm(self.velocity)),
            "currentSpeedMps": float(np.linalg.norm(self.currentVelocity)),
            "insideBox": insideBox,
            "positionX": float(self.position[0]),
            "positionY": float(self.position[1]),
            "positionZ": float(self.position[2]),
            "velocityX": float(self.velocity[0]),
            "velocityY": float(self.velocity[1]),
            "velocityZ": float(self.velocity[2]),
            "currentX": float(self.currentVelocity[0]),
            "currentY": float(self.currentVelocity[1]),
            "currentZ": float(self.currentVelocity[2]),
            "thrustActionX": float(self.previousAction[0]),
            "thrustActionY": float(self.previousAction[1]),
            "thrustActionZ": float(self.previousAction[2]),
            "forceX": float(self.previousAction[0] * self.params.maxForceN),
            "forceY": float(self.previousAction[1] * self.params.maxForceN),
            "forceZ": float(self.previousAction[2] * self.params.maxForceN),
        }

    def _boxHalfSize(self) -> np.ndarray:
        return np.array(
            [
                self.params.stationHalfSizeX,
                self.params.stationHalfSizeY,
                self.params.stationHalfSizeZ,
            ],
            dtype=float,
        )
