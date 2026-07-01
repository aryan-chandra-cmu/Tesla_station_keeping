from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from underwaterStationKeeping.controller import (
    ControllerGains,
    StationKeepingController,
    StationTarget,
)
from underwaterStationKeeping.currentModels import makeCurrentModel
from underwaterStationKeeping.mathUtils import wrapAngle
from underwaterStationKeeping.vehicleModel import (
    ThrusterCommand,
    VehicleParams,
    VehicleState,
    integrateVehicle,
    limitThrusterCommand,
)


@dataclass
class SimulationConfig:
    scenarioName: str = "mild"
    durationSec: float = 120.0
    timeStep: float = 0.05
    initialPositionX: float = 3.0
    initialPositionY: float = -2.0
    initialYawRad: float = 0.8
    targetPositionX: float = 0.0
    targetPositionY: float = 0.0
    targetYawRad: float = 0.0


def runSimulation(config: SimulationConfig) -> np.ndarray:
    params = VehicleParams()
    target = StationTarget(
        positionX=config.targetPositionX,
        positionY=config.targetPositionY,
        yawRad=config.targetYawRad,
    )
    controller = StationKeepingController(ControllerGains(), target)
    currentModel = makeCurrentModel(config.scenarioName)
    state = VehicleState(
        positionX=config.initialPositionX,
        positionY=config.initialPositionY,
        yawRad=config.initialYawRad,
    )
    previousCommand = ThrusterCommand()

    sampleCount = int(config.durationSec / config.timeStep) + 1
    logData = np.zeros((sampleCount, 17), dtype=float)

    for sampleIndex in range(sampleCount):
        timeSec = sampleIndex * config.timeStep
        currentWorld = currentModel.value(timeSec, state.positionX, state.positionY)
        requestedCommand = controller.computeCommand(state, currentWorld, config.timeStep)
        limitedCommand = limitThrusterCommand(
            requestedCommand,
            previousCommand,
            config.timeStep,
            params,
        )

        positionError = np.array(
            [target.positionX - state.positionX, target.positionY - state.positionY],
            dtype=float,
        )
        yawError = wrapAngle(target.yawRad - state.yawRad)
        forceNorm = float(np.linalg.norm(limitedCommand.forceVector()))

        logData[sampleIndex] = np.array(
            [
                timeSec,
                state.positionX,
                state.positionY,
                state.yawRad,
                state.surgeVelocity,
                state.swayVelocity,
                state.yawRate,
                currentWorld[0],
                currentWorld[1],
                limitedCommand.forceXBody,
                limitedCommand.forceYBody,
                limitedCommand.momentYaw,
                float(np.linalg.norm(positionError)),
                yawError,
                forceNorm,
                requestedCommand.forceXBody,
                requestedCommand.forceYBody,
            ],
            dtype=float,
        )

        state = integrateVehicle(state, limitedCommand, currentWorld, config.timeStep, params)
        previousCommand = limitedCommand

    return logData


def saveLog(logData: np.ndarray, outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    headerText = (
        "timeSec,positionX,positionY,yawRad,surgeVelocity,swayVelocity,yawRate,"
        "currentX,currentY,forceXBody,forceYBody,momentYaw,positionErrorNorm,"
        "yawError,forceNorm,requestedForceXBody,requestedForceYBody"
    )
    np.savetxt(outputPath, logData, delimiter=",", header=headerText, comments="")


def summarizeLog(logData: np.ndarray) -> dict[str, float]:
    finalRow = logData[-1]
    positionError = logData[:, 12]
    yawErrorAbs = np.abs(logData[:, 13])
    forceNorm = logData[:, 14]
    return {
        "finalPositionErrorM": float(finalRow[12]),
        "finalYawErrorRad": float(abs(finalRow[13])),
        "meanPositionErrorM": float(np.mean(positionError)),
        "maxPositionErrorM": float(np.max(positionError)),
        "meanYawErrorRad": float(np.mean(yawErrorAbs)),
        "maxForceN": float(np.max(forceNorm)),
    }
