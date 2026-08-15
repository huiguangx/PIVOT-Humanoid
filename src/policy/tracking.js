import { Quaternion, Vector3 } from 'three'

import { lerpFrames, normalizeMotion, slerpFrames } from './motion.js'

export class MotionTracker {
  constructor(config) {
    this.transitionSteps = config.transition_steps ?? 100
    this.datasetJointNames = config.dataset_joint_names ?? []
    this.policyJointNames = config.policy_joint_names ?? []
    this.nJoints = this.datasetJointNames.length || this.policyJointNames.length
    this.policyToDataset = this.#buildPolicyToDatasetMap()
    this.motions = {}

    for (const [name, raw] of Object.entries(config.motions ?? {})) {
      const motion = normalizeMotion(raw, this.nJoints)
      if (motion) this.motions[name] = motion
    }
    if (!this.motions.default) throw new Error('MotionTracker requires a "default" motion')

    this.refJointPos = []
    this.refRootQuat = []
    this.refRootPos = []
    this.refIdx = 0
    this.refLen = 0
    this.transitionLen = 0
    this.motionLen = 0
    this.currentName = 'default'
    this.currentDone = true
  }

  availableMotions() {
    return Object.keys(this.motions)
  }

  addMotions(records, { overwrite = false } = {}) {
    const result = { added: [], skipped: [], invalid: [] }
    if (!records || typeof records !== 'object') return result

    for (const [name, raw] of Object.entries(records)) {
      if (!name) {
        result.invalid.push(name)
      } else if (!overwrite && this.motions[name]) {
        result.skipped.push(name)
      } else {
        const motion = normalizeMotion(raw, this.nJoints)
        if (!motion) result.invalid.push(name)
        else {
          this.motions[name] = motion
          result.added.push(name)
        }
      }
    }
    return result
  }

  reset(state) {
    this.currentDone = true
    this.refIdx = 0
    this.refLen = 0
    this.transitionLen = 0
    this.motionLen = 0
    this.refJointPos = []
    this.refRootQuat = []
    this.refRootPos = []
    this.currentName = 'default'
    this.requestMotion('default', state)
  }

  requestMotion(name, state) {
    if (!this.motions[name]) return false
    this.#startMotion(name, state)
    return true
  }

  isReady() {
    return this.refLen > 0
  }

  playbackState() {
    const refIdx = Math.max(0, Math.min(this.refIdx, Math.max(this.refLen - 1, 0)))
    return {
      available: this.refLen > 0,
      currentName: this.currentName,
      currentDone: this.currentDone,
      refIdx,
      refLen: this.refLen,
      transitionLen: this.transitionLen,
      motionLen: this.motionLen,
      inTransition: this.transitionLen > 0 && refIdx < this.transitionLen,
      isDefault: this.currentName === 'default',
    }
  }

  advance() {
    if (this.refLen && this.refIdx < this.refLen - 1) {
      this.refIdx += 1
      if (this.refIdx === this.refLen - 1) this.currentDone = true
    }
  }

  getFrame(index) {
    const frame = Math.max(0, Math.min(index, this.refLen - 1))
    return {
      jointPos: this.refJointPos[frame],
      rootQuat: this.refRootQuat[frame],
      rootPos: this.refRootPos[frame],
    }
  }

  convertMotionJointPosPolicyToDataset(frames) {
    if (!frames || !Array.isArray(frames)) return frames
    return frames.map((frame) => this.#mapPolicyJointPosToDataset(frame))
  }

  #startMotion(name, state) {
    const current = this.#readCurrentState(state)
    if (state && this.policyToDataset) current.jointPos = this.#mapPolicyJointPosToDataset(current.jointPos)

    const motion = this.#alignMotionToCurrent(this.motions[name], current)
    const count = Math.max(0, Math.floor(this.transitionSteps))
    const transition = {
      jointPos: lerpFrames(current.jointPos, motion.jointPos[0], count),
      rootPos: lerpFrames(current.rootPos, motion.rootPos[0], count),
      rootQuat: slerpFrames(current.rootQuat, motion.rootQuat[0], count),
    }

    this.refJointPos = [...transition.jointPos, ...motion.jointPos]
    this.refRootQuat = [...transition.rootQuat, ...motion.rootQuat]
    this.refRootPos = [...transition.rootPos, ...motion.rootPos]
    this.transitionLen = transition.jointPos.length
    this.motionLen = motion.jointPos.length
    this.refIdx = 0
    this.refLen = this.refJointPos.length
    this.currentName = name
    this.currentDone = this.refLen <= 1
  }

  #readCurrentState(state) {
    if (state) {
      return {
        jointPos: Array.from(state.jointPos),
        rootPos: Array.from(state.rootPos),
        rootQuat: Array.from(state.rootQuat),
      }
    }
    const fallback = this.motions.default
    return {
      jointPos: Array.from(fallback.jointPos[0] ?? new Float32Array(this.nJoints)),
      rootPos: Array.from(fallback.rootPos[0] ?? [0, 0, 0.78]),
      rootQuat: Array.from(fallback.rootQuat[0] ?? [1, 0, 0, 0]),
    }
  }

  #alignMotionToCurrent(motion, current) {
    const origin = new Vector3(...motion.rootPos[0])
    const currentRoot = new Vector3(...current.rootPos)
    const rotation = yawQuaternion(current.rootQuat).multiply(yawQuaternion(motion.rootQuat[0]).invert())
    const start = new Vector3(currentRoot.x, currentRoot.y, origin.z)

    return {
      jointPos: motion.jointPos.map((frame) => Float32Array.from(frame)),
      rootPos: motion.rootPos.map((frame) => {
        const value = new Vector3(...frame).sub(origin).applyQuaternion(rotation).add(start)
        return Float32Array.from(value)
      }),
      rootQuat: motion.rootQuat.map(([w, x, y, z]) => {
        const value = rotation.clone().multiply(new Quaternion(x, y, z, w))
        return Float32Array.from([value.w, value.x, value.y, value.z])
      }),
    }
  }

  #buildPolicyToDatasetMap() {
    if (!this.datasetJointNames.length || !this.policyJointNames.length) return null
    const indices = new Map(this.datasetJointNames.map((name, index) => [name, index]))
    return this.policyJointNames.map((name) => indices.get(name) ?? -1)
  }

  #mapPolicyJointPosToDataset(values) {
    if (!this.policyToDataset || !this.datasetJointNames.length) return Float32Array.from(values)
    const mapped = new Float32Array(this.datasetJointNames.length)
    this.policyToDataset.forEach((target, source) => {
      if (target >= 0) mapped[target] = values[source] ?? 0
    })
    return mapped
  }
}

function yawQuaternion([w, x, y, z]) {
  const yaw = 0.5 * Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return new Quaternion(0, 0, Math.sin(yaw), Math.cos(yaw))
}
