import { Tensor } from 'onnxruntime-web'

import { OnnxModel } from './runner.js'

export class LocomotionRunner {
  constructor(config, model = new OnnxModel(config.onnx)) {
    this.config = config
    this.numActions = config.policy_joint_names.length
    this.defaultJointPos = Float32Array.from(config.default_joint_pos)
    this.actionScale = config.action_scale
    this.controlDt = config.control_dt ?? 0.02
    this.gaitPeriod = config.gait_period ?? 0.8
    this.model = model
    this.command = { forward: 0, lateral: 0, yaw: 0 }
    this.previousAction = new Float32Array(this.numActions)
    this.elapsed = 0
  }

  async init() {
    await this.model.init?.()
  }

  reset() {
    this.previousAction.fill(0)
    this.elapsed = 0
  }

  setCommand(command = {}) {
    const limits = this.config.command_limits
    this.command = {
      forward: clamp(command.forward ?? 0, limits.forward),
      lateral: clamp(command.lateral ?? 0, limits.lateral),
      yaw: clamp(command.yaw ?? 0, limits.yaw),
    }
  }

  async step(state) {
    this.elapsed += this.controlDt
    const phase = (this.elapsed % this.gaitPeriod) / this.gaitPeriod
    const observation = buildLocomotionObservation(
      state,
      this.command,
      this.previousAction,
      phase,
      this.config,
    )
    const [output] = await this.model.run({
      policy: new Tensor('float32', observation, [1, observation.length]),
    })
    const action = output.action?.data
    if (action?.length !== this.numActions || Array.from(action).some((value) => !Number.isFinite(value))) {
      throw new Error('Invalid locomotion policy output')
    }
    this.previousAction.set(action)
    return Float32Array.from(action, (value, index) => (
      this.defaultJointPos[index] + this.actionScale * value
    ))
  }

  dispose() {
    return this.model.dispose?.()
  }
}

export function buildLocomotionObservation(state, command, previousAction, phase, config) {
  const count = config.policy_joint_names.length
  const output = new Float32Array(9 + count * 3 + 2)
  const scales = config.observation_scale
  const [qw, qx, qy, qz] = state.rootQuat

  output.set(Array.from(state.rootAngVel, (value) => value * scales.angularVelocity), 0)
  output.set([
    2 * (-qz * qx + qw * qy),
    -2 * (qz * qy + qw * qx),
    1 - 2 * (qw * qw + qz * qz),
  ], 3)
  output.set([
    command.forward * config.command_scale[0],
    command.lateral * config.command_scale[1],
    command.yaw * config.command_scale[2],
  ], 6)
  output.set(Array.from(state.jointPos, (value, index) => (
    (value - config.default_joint_pos[index]) * scales.jointPosition
  )), 9)
  output.set(Array.from(state.jointVel, (value) => value * scales.jointVelocity), 9 + count)
  output.set(previousAction, 9 + count * 2)
  output.set([Math.sin(2 * Math.PI * phase), Math.cos(2 * Math.PI * phase)], 9 + count * 3)
  return output
}

function clamp(value, [minimum, maximum]) {
  return Math.max(minimum, Math.min(maximum, Number(value) || 0))
}
