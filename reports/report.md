# Underwater Vehicle Station-Keeping Report

## Problem Formulation and Modeling Choices

The goal is to hold an underwater vehicle at a desired station while ocean currents push it away. In physical terms, this is a disturbance-rejection problem: the vehicle must create enough force to cancel the effect of moving water, while also removing any position and heading error that already exists.

The non-RL controller uses a planar 3-DOF model:

```text
state = x, y, yaw, surge velocity, sway velocity, yaw rate
input = body-frame x force, body-frame y force, yaw moment
```

The vehicle position is expressed in the world frame. Linear velocities are expressed in the body frame. The ocean current is modeled as a world-frame velocity disturbance. The key modeling idea is that drag should depend on the vehicle's velocity relative to the water, not just relative to the ground. A vehicle drifting perfectly with the water should not experience the same damping force as a vehicle moving quickly through still water.

The simplified dynamics are:

```text
world velocity = R(yaw) * body velocity + current velocity
linear acceleration = (commanded force - damping * body velocity) / mass
yaw acceleration = (commanded moment - yaw damping * yaw rate) / yaw inertia
```

Thruster limits are included because station keeping is only meaningful if the controller cannot command unlimited force. The simulation limits maximum force, maximum yaw moment, and how quickly thrust can change. These limits make the problem closer to a real controls task: the controller must not only point in the right direction, but also work within actuator authority.

The repository also includes an optional RL version. The RL environment models the station as a 3D box around the target. The observation contains proprioceptive state and disturbance information:

```text
observation = position error, vehicle velocity, current velocity, previous thrust
action = thrust in x, y, z
```

This RL setup is intentionally simple and control-focused. It avoids camera input, mapping, or perception so that the policy is evaluated on the actual control problem: reaching and holding a station under current disturbance.

## Assumptions Made and Why

The non-RL model assumes direct control over body-frame surge force, sway force, and yaw moment. This avoids the extra problem of mapping individual thrusters to net forces and moments. That actuator-allocation layer is important on a real vehicle, but it is separate from the station-keeping control question.

The model ignores depth, roll, and pitch for the classical controller. This was chosen because the assignment asks for a working control solution, and the central behavior can be demonstrated clearly in the horizontal plane. A planar model also makes the results easy to inspect: position error, heading error, current velocity, and command effort are all visible in a compact set of plots.

Hydrodynamic effects are approximated with linear damping. Real underwater vehicles often have nonlinear drag, added mass, thruster lag, and coupling between axes. The simplified model is still useful because it preserves the first-order control challenge: current creates a persistent velocity disturbance, and the controller must hold position without exceeding actuator limits.

The current field is treated as measured in the non-RL controller. This is a favorable assumption, but it is reasonable for a first implementation because many underwater vehicles estimate water-relative motion using onboard sensors. If current were not measured, the integral term would still reject steady drift, but response would be slower.

## Control Approach and Why It Was Chosen

### Classical Controller

The primary controller is a PID station-keeping controller. The position loop behaves like a virtual spring-damper system:

```text
force = Kp * position error + Ki * accumulated error + Kd * velocity error
```

The proportional term pulls the vehicle back toward the station. The derivative term damps motion so the vehicle does not overshoot and oscillate. The integral term removes steady bias from persistent current. Without the integral term, a constant current can leave the vehicle with a small permanent offset. With too much integral action, the controller can wind up and overshoot, so the integral state is clipped.

The position command is computed in the world frame and then rotated into the vehicle body frame. This matters because the target is defined in world coordinates, while thrusters act in the vehicle's local frame. A separate yaw loop regulates heading using the same PID idea.

The command is then passed through force, moment, and slew-rate limits. This last step is important analytically: if the controller only works by requesting unrealistic impulses, it is not a useful controller. The plotted control histories show when the controller reaches the available authority and how it behaves after the initial transient.

This approach was chosen because it is transparent and easy to debug. Each term has a physical interpretation, and the failure modes are readable from plots: too little damping causes oscillation, too little integral action leaves drift, and insufficient force authority prevents station keeping under strong current.

### RL Control Pipeline

The RL controller is included as an optional comparison. It uses the same basic station-keeping idea, but instead of manually specifying a PID law, the policy learns a mapping:

```text
state and current information -> thrust command
```

The RL environment uses a 3D station box. A policy receives position error, velocity, current velocity, and previous action. The previous action is included because thrust rate limits create short-term memory in the actuator: the available command at the next step depends on the current command.

The reward is shaped around the task definition:

```text
penalize distance from the station box
penalize being outside the box
penalize high speed
penalize unnecessary thrust
reward being inside the box
reward being inside the box while moving slowly
```

The initial direct RL run did not learn reliably. It survived full episodes, but evaluation reward stayed strongly negative and the entropy coefficient became small early in training. That is a common sign that the policy has settled into a weak deterministic behavior before discovering the useful control strategy.

For that reason, the RL version uses a curriculum:

| Stage | Purpose | Environment |
|---|---|---|
| Stage 1 | Learn to hold station | Large box, small initial offset, weak current |
| Stage 2 | Learn recovery behavior | Medium box, larger offsets, moderate current |
| Stage 3 | Solve final task | Tight box, full offset range, full current |

The curriculum follows the structure of the control problem. Holding position near the station is easier than recovering from a large offset. Recovering from a medium offset is easier than handling the full current and tight station box from the beginning. Training in this order gives the policy a useful control behavior before it faces the hardest version of the task.

## Results

### Classical Controller Results

Three current conditions were tested: mild, strong, and reversing. Each scenario starts with a large initial position offset and nonzero heading error. Success is measured by final position error, heading error, settling time, and whether the controller respects actuator limits.

| Scenario | Final position error | Mean position error | Final yaw error | Settling time under 0.25 m | Max force norm |
|---|---:|---:|---:|---:|---:|
| Mild current | 0.018 m | 0.093 m | 0.022 deg | 4.2 s | 85.0 N |
| Strong current | 0.157 m | 0.167 m | 0.022 deg | 4.9 s | 85.0 N |
| Reversing current | 0.165 m | 0.165 m | 0.022 deg | 4.2 s | 84.6 N |

The main observation is that the controller converges quickly despite different current profiles. The early force spike is expected: the vehicle begins several meters away from the station, so the controller initially uses most of the available thrust to remove the large error. After the transient, the commands settle into smaller values that counteract the time-varying current.

![Mild current station-keeping result](figures/mild_run.png)

![Strong current station-keeping result](figures/strong_run.png)

![Reversing current station-keeping result](figures/reversing_run.png)

The mild case settles to nearly zero position error. The strong and reversing cases retain a small residual error because the current disturbance is larger and time-varying, but both remain within the 0.25 m station-keeping tolerance after the initial transient. The yaw loop is much easier than the translational loop in this model because the current disturbance does not directly apply a yaw moment.

### RL Controller Results

The curriculum-trained RL policy was evaluated at each stage and on the final Stage 3 environment. The policy learned the main task behavior: drive toward the station, enter the box, reduce speed, and use smaller corrective thrusts once it is near the target.

| Evaluation | Final position error | Mean position error | Final speed | Inside-box fraction |
|---|---:|---:|---:|---:|
| Stage 1 representative rollout | 0.331 m | 0.315 m | 0.042 m/s | 0.98 |
| Stage 2 representative rollout | 0.184 m | 0.290 m | 0.051 m/s | 0.93 |
| Stage 3 representative rollout | 0.167 m | 0.308 m | 0.050 m/s | 0.90 |
| Stage 3 ten-episode evaluation | 0.146 m final sample | 0.250 m | 0.065 m/s final sample | 0.873 |

The Stage 1 result shows the policy learning the easiest version of the problem: staying inside a large station box with weak current. Stage 2 increases both initial error and current strength, and the policy still finishes inside the box. The Stage 3 rollout is the most important result because it uses the final tight station box and full disturbance range. It ends with 0.167 m position error and 0.050 m/s speed, which is inside the station box.

![RL Stage 1 policy rollout](figures/rl_policy_path_stage1.png)

![RL Stage 2 policy rollout](figures/rl_policy_path_stage2.png)

![RL Stage 3 policy rollout](figures/rl_policy_path.png)

The maximum error in the RL plots is dominated by the starting offset, not by steady-state drift. The policy initially uses large thrust to remove the distance error, then drops to smaller commands once it reaches the station region. Compared with the classical PID controller, the RL controller is less smooth and uses more visibly discrete corrective actions, but it succeeds on the simplified 3D station-box task.

## Limitations

The classical controller works well when the required station-keeping force is within actuator limits. It will break if the ocean current is strong enough that the required steady force exceeds the maximum thrust. It will also struggle if the current changes faster than the thrust slew-rate limit allows, because the controller cannot instantly produce the force needed to cancel the new disturbance.

The planar model omits depth dynamics, roll, pitch, nonlinear hydrodynamics, added mass, sensor noise, and actuator allocation. These omissions make the simulation easier to inspect, but they also mean the result should be interpreted as a controls prototype rather than a full underwater vehicle simulator.

The non-RL controller assumes current velocity is measured. If current measurement is unavailable or noisy, the derivative and feed-forward behavior would be less accurate. Integral action could still reject steady drift, but with slower convergence and more overshoot.

The RL controller succeeds in the simplified 3D box task, but it is less interpretable than the classical controller. Reward design and curriculum structure strongly affect whether the policy learns station keeping or finds a poor local behavior. The trained policy is also tied to the observation scaling, current range, station-box size, and direct world-frame thrust model used during training. A more realistic version would need thruster allocation, sensor noise, actuator lag, and validation across many randomized current fields.

## Honesty and External Tools

The solution uses Python, NumPy, Matplotlib, Gymnasium, and Stable-Baselines3. AI assistance was used to help structure the repository, boilerplate RL code - tuning and reward/ curriculum design etc were done by me, and prepare this report. The controller design, modeling assumptions, plotted metrics, and limitations are stated explicitly so the result can be evaluated on engineering judgment rather than hidden implementation details.
