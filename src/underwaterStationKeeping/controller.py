from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from underwaterStationKeeping.mathUtils import rotationMatrix, wrapAngle
from underwaterStationKeeping.vehicleModel import ThrusterCommand, VehicleState


@dataclass
class StationTarget:
    positionX: float = 0.0
    positionY: float = 0.0
    yawRad: float = 0.0


@dataclass
class ControllerGains:
    positionKp: float = 42.0
    positionKi: float = 2.0
    positionKd: float = 55.0
    yawKp: float = 28.0
    yawKi: float = 1.0
    yawKd: float = 18.0
    positionIntegralLimit: float = 2.0
    yawIntegralLimit: float = 0.8


class StationKeepingController:
    def __init__(self, gains: ControllerGains, target: StationTarget) -> None:
        self.gains = gains
        self.target = target
        self.positionIntegral = np.zeros(2, dtype=float)
        self.yawIntegral = 0.0

    def reset(self) -> None:
        self.positionIntegral[:] = 0.0
        self.yawIntegral = 0.0

    def computeCommand(
        self,
        state: VehicleState,
        currentWorld: np.ndarray,
        timeStep: float,
    ) -> ThrusterCommand:
        positionErrorWorld = np.array(
            [
                self.target.positionX - state.positionX,
                self.target.positionY - state.positionY,
            ],
            dtype=float,
        )
        bodyVelocityWorld = rotationMatrix(state.yawRad) @ np.array(
            [state.surgeVelocity, state.swayVelocity],
            dtype=float,
        )
        positionRateErrorWorld = -(bodyVelocityWorld + currentWorld)

        self.positionIntegral += positionErrorWorld * timeStep
        self.positionIntegral = np.clip(
            self.positionIntegral,
            -self.gains.positionIntegralLimit,
            self.gains.positionIntegralLimit,
        )

        forceWorld = (
            self.gains.positionKp * positionErrorWorld
            + self.gains.positionKi * self.positionIntegral
            + self.gains.positionKd * positionRateErrorWorld
        )
        forceBody = rotationMatrix(state.yawRad).T @ forceWorld

        yawError = wrapAngle(self.target.yawRad - state.yawRad)
        yawRateError = -state.yawRate
        self.yawIntegral += yawError * timeStep
        self.yawIntegral = float(
            np.clip(
                self.yawIntegral,
                -self.gains.yawIntegralLimit,
                self.gains.yawIntegralLimit,
            )
        )
        momentYaw = (
            self.gains.yawKp * yawError
            + self.gains.yawKi * self.yawIntegral
            + self.gains.yawKd * yawRateError
        )

        return ThrusterCommand(
            forceXBody=float(forceBody[0]),
            forceYBody=float(forceBody[1]),
            momentYaw=float(momentYaw),
        )
