import { InferenceSession, Tensor } from 'onnxruntime-web'
import { Quaternion, Vector3 } from 'three'

import { MotionTracker } from './tracking.js'

export class PolicyRunner {
  constructor(config) {
    this.config = config
    this.policyJointNames = config.policy_joint_names?.slice() ?? []
    if (!this.policyJointNames.length) throw new Error('PolicyRunner requires policy_joint_names')

    this.numActions = this.policyJointNames.length
    this.actionScale = vector(config.action_scale, this.numActions, 1)
    this.defaultJointPos = vector(config.default_joint_pos, this.numActions, 0)
    this.actionClip = typeof config.action_clip === 'number' ? config.action_clip : 10
    this.lastActions = new Float32Array(this.numActions)
    this.tracking = config.tracking
      ? new MotionTracker({ ...config.tracking, policy_joint_names: this.policyJointNames })
      : null
    this.modules = buildModules(this, config.obs_config?.policy ?? [])
    this.numObservations = this.modules.reduce((sum, module) => sum + module.size, 0)
    this.model = new OnnxModel(config.onnx)
    this.input = {}
    this.running = false
  }

  async init() {
    await this.model.init()
    this.reset()
  }

  reset(state = null) {
    this.input = this.model.initialInput()
    this.lastActions.fill(0)
    this.tracking?.reset(state)
    this.modules.forEach((module) => module.reset?.(state))
  }

  async step(state) {
    if (this.running) return null
    if (!state) throw new Error('PolicyRunner.step requires a state object')
    this.running = true
    try {
      this.tracking?.advance()
      const observation = new Float32Array(this.numObservations)
      let offset = 0
      for (const module of this.modules) {
        module.update?.(state)
        const values = module.compute(state)
        observation.set(values, offset)
        offset += values.length
      }
      this.input.policy = new Tensor('float32', observation, [1, observation.length])
      const [output, recurrent] = await this.model.run(this.input)
      this.input = { ...this.input, ...recurrent }
      const action = output.action?.data
      if (!action || action.length !== this.numActions) throw new Error('Invalid policy action output')
      for (let index = 0; index < this.numActions; index++) {
        this.lastActions[index] = Math.max(-this.actionClip, Math.min(this.actionClip, action[index]))
      }
      return Float32Array.from(this.lastActions, (value, index) => (
        this.defaultJointPos[index] + this.actionScale[index] * value
      ))
    } finally {
      this.running = false
    }
  }
}

export class OnnxModel {
  constructor(config) {
    this.path = config.path
    this.meta = config.meta
    this.recurrent = config.meta.in_keys.includes('adapt_hx')
  }

  async init() {
    const bytes = await fetch(this.path).then((response) => response.arrayBuffer())
    this.session = await InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  }

  initialInput() {
    return this.recurrent
      ? { is_init: new Tensor('bool', [true], [1]), adapt_hx: new Tensor('float32', new Float32Array(128), [1, 128]) }
      : {}
  }

  async run(input) {
    const feeds = Object.fromEntries(this.meta.in_keys.map((key, index) => [this.session.inputNames[index], input[key]]))
    const raw = await this.session.run(feeds)
    const output = Object.fromEntries(this.meta.out_keys.map((key, index) => [key, raw[this.session.outputNames[index]]]))
    const recurrent = this.recurrent
      ? { is_init: new Tensor('bool', [false], [1]), adapt_hx: output['next,adapt_hx'] }
      : {}
    return [output, recurrent]
  }
}

function buildModules(policy, specs) {
  return specs.map(({ name, ...options }) => {
    switch (name) {
      case 'BootIndicator': return { size: 1, compute: () => new Float32Array(1) }
      case 'RootAngVelB': return { size: 3, compute: (state) => Float32Array.from(state.rootAngVel) }
      case 'ProjectedGravityB': return projectedGravity()
      case 'JointPos': return jointHistory(policy.numActions, options.pos_steps)
      case 'PrevActions': return actionHistory(policy, options.history_steps)
      case 'TrackingCommandObsRaw': return trackingCommand(policy, options.future_steps)
      case 'TargetRootZObs': return targetRootZ(policy, options.future_steps)
      case 'TargetJointPosObs': return targetJointPos(policy, options.future_steps)
      case 'TargetProjectedGravityBObs': return targetGravity(policy, options.future_steps)
      default: throw new Error(`Unknown observation type: ${name}`)
    }
  })
}

function projectedGravity() {
  return { size: 3, compute: ({ rootQuat: [w, x, y, z] }) => {
    const value = new Vector3(0, 0, -1).applyQuaternion(new Quaternion(x, y, z, w).invert())
    return Float32Array.from(value)
  } }
}

function jointHistory(count, steps = [0, 1, 2, 3, 4, 8]) {
  const history = Array.from({ length: Math.max(...steps) + 1 }, () => new Float32Array(count))
  return {
    size: steps.length * count,
    reset: (state) => history.forEach((frame) => frame.set(state?.jointPos ?? new Float32Array(count))),
    update: (state) => { for (let i = history.length - 1; i > 0; i--) history[i].set(history[i - 1]); history[0].set(state.jointPos) },
    compute: () => { const output = new Float32Array(steps.length * count); steps.forEach((step, i) => output.set(history[Math.min(step, history.length - 1)], i * count)); return output },
  }
}

function actionHistory(policy, steps = 4) {
  const history = Array.from({ length: Math.max(1, Math.floor(steps)) }, () => new Float32Array(policy.numActions))
  return {
    size: history.length * policy.numActions,
    reset: () => history.forEach((frame) => frame.fill(0)),
    update: () => { for (let i = history.length - 1; i > 0; i--) history[i].set(history[i - 1]); history[0].set(policy.lastActions) },
    compute: () => { const output = new Float32Array(history.length * policy.numActions); history.forEach((frame, i) => output.set(frame, i * policy.numActions)); return output },
  }
}

function trackingCommand(policy, steps = [0, 2, 4, 8, 16]) {
  return trackingModule(policy, steps, (tracker, state, indices) => {
    const origin = tracker.refRootPos[indices[0]]
    const reference = tracker.refRootQuat[indices[0]]
    const positions = indices.slice(1).flatMap((index) => rotateInverse(reference, tracker.refRootPos[index].map((v, i) => v - origin[i])))
    const inverse = quaternionInverse(state.rootQuat)
    const rotations = indices.flatMap((index) => rotation6(quaternionMultiply(inverse, tracker.refRootQuat[index])))
    return Float32Array.from([...positions, ...rotations])
  }, (steps.length - 1) * 3 + steps.length * 6)
}

function targetRootZ(policy, steps = [0, 2, 4, 8, 16]) {
  return trackingModule(policy, steps, (tracker, state, indices) => Float32Array.from(indices, (index) => tracker.refRootPos[index][2] + 0.035), steps.length)
}

function targetJointPos(policy, steps = [0, 2, 4, 8, 16]) {
  return trackingModule(policy, steps, (tracker, state, indices) => {
    const output = new Float32Array(indices.length * tracker.nJoints)
    indices.forEach((index, offset) => output.set(tracker.refJointPos[index], offset * tracker.nJoints))
    return output
  }, steps.length * policy.tracking.nJoints)
}

function targetGravity(policy, steps = [0, 2, 4, 8, 16]) {
  return trackingModule(policy, steps, (tracker, state, indices) => Float32Array.from(indices.flatMap((index) => rotateInverse(tracker.refRootQuat[index], [0, 0, -1]))), steps.length * 3)
}

function trackingModule(policy, steps, compute, size) {
  return { size, compute: (state) => {
    const tracker = policy.tracking
    if (!tracker?.isReady()) return new Float32Array(size)
    const indices = steps.map((step) => Math.max(0, Math.min(tracker.refIdx + step, tracker.refLen - 1)))
    return compute(tracker, state, indices)
  } }
}

function vector(value, length, fallback) {
  const output = new Float32Array(length)
  for (let index = 0; index < length; index++) output[index] = value?.[index] ?? (typeof value === 'number' ? value : fallback)
  return output
}

function normalizeQuaternion(value) {
  const length = Math.hypot(...value)
  return length < 1e-9 ? [1, 0, 0, 0] : Array.from(value, (item) => item / length)
}

function quaternionInverse(value) {
  const [w, x, y, z] = normalizeQuaternion(value)
  return [w, -x, -y, -z]
}

function quaternionMultiply([w, x, y, z], [a, b, c, d]) {
  return [w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c, w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a]
}

function rotateInverse(quaternion, vectorValue) {
  const [w, x, y, z] = normalizeQuaternion(quaternion)
  return new Vector3(...vectorValue).applyQuaternion(new Quaternion(x, y, z, w).invert()).toArray()
}

function rotation6(value) {
  const [w, x, y, z] = normalizeQuaternion(value)
  return [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
}
