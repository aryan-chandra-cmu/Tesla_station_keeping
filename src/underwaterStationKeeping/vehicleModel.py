from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from underwaterStationKeeping.mathUtils import rotationMatrix, wrapAngle


@dataclass
class VehicleParams:
    massKg: float = 38.0
    yawInertiaKgM2: float = 7.5
    surgeDamping: float = 18.0
    swayDamping: float = 24.0
    yawDamping: float = 8.0
    maxForceN: float = 85.0
    maxMomentNm: float = 35.0
    maxForceRateNPerS: float = 120.0
    maxMomentRateNmPerS: float = 80.0


@dataclass
class VehicleState:
    positionX: float = 0.0
    positionY: float = 0.0
    yawRad: float = 0.0
    surgeVelocity: float = 0.0
    swayVelocity: float = 0.0
    yawRate: float = 0.0

    def asArray(self) -> np.ndarray:
        return np.array(
            [
                self.positionX,
                self.positionY,
                self.yawRad,
                self.surgeVelocity,
                self.swayVelocity,
                self.yawRate,
            ],
            dtype=float,
        )

    @classmethod
    def fromArray(cls, stateArray: np.ndarray) -> "VehicleState":
        return cls(
            positionX=float(stateArray[0]),
            positionY=float(stateArray[1]),
            yawRad=float(stateArray[2]),
            surgeVelocity=float(stateArray[3]),
            swayVelocity=float(stateArray[4]),
            yawRate=float(stateArray[5]),
        )


@dataclass
class ThrusterCommand:
    forceXBody: float = 0.0
    forceYBody: float = 0.0
    momentYaw: float = 0.0

    def forceVector(self) -> np.ndarray:
        return np.array([self.forceXBody, self.forceYBody], dtype=float)


def limitThrusterCommand(
    requestedCommand: ThrusterCommand,
    previousCommand: ThrusterCommand,
    timeStep: float,
    params: VehicleParams,
) -> ThrusterCommand:
    requestedForce = requestedCommand.forceVector()
    previousForce = previousCommand.forceVector()

    forceNorm = float(np.linalg.norm(requestedForce))
    if forceNorm > params.maxForceN:
        requestedForce = requestedForce * (params.maxForceN / forceNorm)

    maxForceDelta = params.maxForceRateNPerS * timeStep
    forceDelta = requestedForce - previousForce
    forceDeltaNorm = float(np.linalg.norm(forceDelta))
    if forceDeltaNorm > maxForceDelta and forceDeltaNorm > 1e-12:
        requestedForce = previousForce + forceDelta * (maxForceDelta / forceDeltaNorm)

    requestedMoment = float(
        np.clip(requestedCommand.momentYaw, -params.maxMomentNm, params.maxMomentNm)
    )
    maxMomentDelta = params.maxMomentRateNmPerS * timeStep
    momentDelta = np.clip(
        requestedMoment - previousCommand.momentYaw,
        -maxMomentDelta,
        maxMomentDelta,
    )
    limitedMoment = previousCommand.momentYaw + float(momentDelta)

    return ThrusterCommand(
        forceXBody=float(requestedForce[0]),
        forceYBody=float(requestedForce[1]),
        momentYaw=limitedMoment,
    )


def integrateVehicle(
    state: VehicleState,
    command: ThrusterCommand,
    currentWorld: np.ndarray,
    timeStep: float,
    params: VehicleParams,
) -> VehicleState:
    stateArray = state.asArray()

    def stateDerivative(inputArray: np.ndarray) -> np.ndarray:
        inputState = VehicleState.fromArray(inputArray)
        bodyVelocity = np.array([inputState.surgeVelocity, inputState.swayVelocity])
        worldVelocity = rotationMatrix(inputState.yawRad) @ bodyVelocity + currentWorld

        surgeAccel = (
            command.forceXBody - params.surgeDamping * inputState.surgeVelocity
        ) / params.massKg
        swayAccel = (
            command.forceYBody - params.swayDamping * inputState.swayVelocity
        ) / params.massKg
        yawAccel = (
            command.momentYaw - params.yawDamping * inputState.yawRate
        ) / params.yawInertiaKgM2

        return np.array(
            [
                worldVelocity[0],
                worldVelocity[1],
                inputState.yawRate,
                surgeAccel,
                swayAccel,
                yawAccel,
            ],
            dtype=float,
        )

    # this function uses fourth-order Runge-Kutta integration.
    k1 = stateDerivative(stateArray)
    k2 = stateDerivative(stateArray + 0.5 * timeStep * k1)
    k3 = stateDerivative(stateArray + 0.5 * timeStep * k2)
    k4 = stateDerivative(stateArray + timeStep * k3)
    nextArray = stateArray + (timeStep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    nextArray[2] = wrapAngle(float(nextArray[2]))
    return VehicleState.fromArray(nextArray)
