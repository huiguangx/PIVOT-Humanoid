import createMujoco from '../vendor/mujoco.js'
import { Scene } from 'three'

import { LocomotionRunner } from '../policy/locomotion.js'
import { PolicyRunner } from '../policy/runner.js'
import { DEFAULT_ROBOT_ID, getRobotProfile } from '../robots/registry.js'
import { loadMotions, loadPolicyConfig, populateMujocoFilesystem } from './assets.js'
import { buildScene, createRenderer, DragInteraction, updateBodies } from './scene.js'

export class SimulationController {
  static async create(container, robotId = DEFAULT_ROBOT_ID) {
    const mujoco = await createMujoco()
    if (!mujoco.FS.analyzePath('/working').exists) mujoco.FS.mkdir('/working')
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working')
    const controller = new SimulationController(mujoco, container)
    await controller.init(robotId)
    return controller
  }

  constructor(mujoco, container) {
    this.mujoco = mujoco
    this.container = container
    Object.assign(this, createRenderer(container))
    this.params = { paused: true, current_motion: 'default' }
    this.renderScale = innerWidth < 500 || innerHeight < 700 ? 1 : 2
    this.followEnabled = false
    this.drag = new DragInteraction(this.scene, this.camera, this.renderer.domElement, this.controls)
    this.zeroTorque = new Float64Array(3)
    this.simStepHz = 0
    this.alive = false
    this.runtime = null
    this.switchVersion = 0
    this.loadedAssetRoots = new Set()
    this.resize = () => this.onResize()
    addEventListener('resize', this.resize)
    this.renderer.setAnimationLoop(() => this.render())
  }

  async init(robotId = DEFAULT_ROBOT_ID) {
    await this.switchRobot(robotId)
    this.alive = true
  }

  async switchRobot(robotId) {
    const profile = getRobotProfile(robotId)
    const version = ++this.switchVersion
    let candidate
    try {
      candidate = await this.prepareRobotRuntime(profile)
      if (version !== this.switchVersion) {
        this.disposeRobotRuntime(candidate)
        return false
      }
      this.commitRobotRuntime(candidate)
      return true
    } catch (cause) {
      if (candidate) this.disposeRobotRuntime(candidate)
      throw cause
    }
  }

  async prepareRobotRuntime(profile) {
    let model, data, runner, root
    try {
      const assetKey = `${profile.assetRoot}:${profile.assetDestination}`
      if (!this.loadedAssetRoots?.has(assetKey)) {
        await populateMujocoFilesystem(this.mujoco, profile.assetRoot, profile.assetDestination)
        this.loadedAssetRoots?.add(assetKey)
      }
      const config = await loadPolicyConfig(profile.policy)
      if (config.tracking?.motions_path && !config.tracking.motions) {
        config.tracking.motions = await loadMotions(new URL(config.tracking.motions_path, location.href))
      }

      model = this.mujoco.MjModel.loadFromXML(`/working/${profile.scene}`)
      data = new this.mujoco.MjData(model)
      const jointNames = decodeNames(model.names, model.name_jntadr, model.njnt)
      const mapping = mapJoints(this.mujoco, model, jointNames, config.policy_joint_names)
      runner = profile.driver === 'tracking' ? new PolicyRunner(config) : new LocomotionRunner(config)
      await runner.init()

      const runtime = {
        profile,
        model,
        data,
        runner,
        jointNames,
        policyJointNames: config.policy_joint_names.slice(),
        ...mapping,
        kp: Float32Array.from(config.stiffness),
        kd: Float32Array.from(config.damping),
        controlType: config.control_type ?? 'joint_position',
        timestep: model.opt.timestep,
        decimation: Math.max(1, Math.round((config.control_dt ?? 0.02) / model.opt.timestep)),
        currentPolicyPath: profile.policy,
      }
      const state = readRuntimeState(runtime)
      runner.reset(state)
      const target = await runner.step(state)
      if (target?.length !== runtime.policyJointNames.length || Array.from(target).some((value) => !Number.isFinite(value))) {
        throw new Error(`Invalid ${profile.id} policy target`)
      }
      runner.reset(state)

      const stagingScene = new Scene()
      const visual = buildScene(this.mujoco, model, data, stagingScene)
      ;({ root } = visual)
      return { ...runtime, ...visual }
    } catch (cause) {
      root?.removeFromParent()
      await runner?.dispose?.()
      data?.delete?.()
      model?.delete?.()
      throw cause
    }
  }

  commitRobotRuntime(runtime) {
    const previous = this.runtime
    this.drag?.end()
    runtime.root.removeFromParent()
    previous?.root?.removeFromParent()
    this.scene?.add(runtime.root)
    this.runtime = runtime
    this.robotId = runtime.profile.id
    Object.assign(this, {
      model: runtime.model,
      data: runtime.data,
      policyRunner: runtime.runner,
      mujocoRoot: runtime.root,
      bodies: runtime.bodies,
      lights: runtime.lights,
      jointNames: runtime.jointNames,
      policyJointNames: runtime.policyJointNames,
      ctrlAddresses: runtime.ctrlAddresses,
      qposAddresses: runtime.qposAddresses,
      qvelAddresses: runtime.qvelAddresses,
      kp: runtime.kp,
      kd: runtime.kd,
      controlType: runtime.controlType,
      timestep: runtime.timestep,
      decimation: runtime.decimation,
      currentPolicyPath: runtime.currentPolicyPath,
    })
    this.params.current_motion = 'default'
    if (previous) this.disposeRobotRuntime(previous)
  }

  disposeRobotRuntime(runtime) {
    runtime?.root?.removeFromParent()
    runtime?.runner?.dispose?.()
    runtime?.data?.delete?.()
    runtime?.model?.delete?.()
  }

  start(onError) {
    this.params.paused = false
    this.loopPromise = this.loop().catch((cause) => {
      this.params.paused = true
      onError?.(cause)
    })
    return this.loopPromise
  }

  async loop() {
    let frames = 0
    let sampleStart = performance.now()
    while (this.alive) {
      const started = performance.now()
      const runtime = this.runtime
      const runner = runtime?.runner ?? this.policyRunner
      if (!this.params.paused && runner) {
        const target = await runner.step(runtime ? readRuntimeState(runtime) : this.readPolicyState())
        if (runtime && !isCurrentRuntime(runtime, this.runtime)) continue
        const active = runtime ?? this
        for (let step = 0; step < active.decimation; step++) {
          active.data.qfrc_applied.fill(0)
          if (target && active.controlType === 'joint_position') this.#applyControl(target, active)
          this.#applyDragForce(active)
          this.mujoco.mj_step(active.model, active.data)
        }
        updateBodies(active.model, active.data, active.bodies, active.lights)
        frames++
        const elapsed = performance.now() - sampleStart
        if (elapsed >= 500) { this.simStepHz = frames * 1000 / elapsed; frames = 0; sampleStart = performance.now() }
      }
      const wait = Math.max(0, this.timestep * this.decimation * 1000 - (performance.now() - started))
      await new Promise((resolve) => setTimeout(resolve, wait))
    }
  }

  #applyControl(target, runtime) {
    for (let index = 0; index < runtime.policyJointNames.length; index++) {
      const actuator = runtime.ctrlAddresses[index]
      const range = actuatorControlRange(runtime.model, actuator)
      runtime.data.ctrl[actuator] = pdTorque(
        target[index], runtime.data.qpos[runtime.qposAddresses[index]], runtime.data.qvel[runtime.qvelAddresses[index]],
        runtime.kp[index], runtime.kd[index], range,
      )
    }
  }

  #applyDragForce(runtime) {
    const object = this.drag.physicsObject
    if (!object?.bodyID) return
    updateBodies(runtime.model, runtime.data, runtime.bodies)
    this.drag.update()
    this.mujoco.mj_applyFT(
      runtime.model,
      runtime.data,
      dragForce(this.drag.currentWorld.clone().sub(this.drag.worldHit)),
      this.zeroTorque,
      threeToMujoco(this.drag.worldHit),
      object.bodyID,
      runtime.data.qfrc_applied,
    )
  }

  readPolicyState() {
    return readRuntimeState(this.runtime)
  }

  resetSimulation() {
    this.mujoco.mj_resetData(this.model, this.data)
    this.mujoco.mj_forward(this.model, this.data)
    this.policyRunner.reset(this.readPolicyState())
    this.params.current_motion = 'default'
  }

  isUpright(options) { return isUprightState(this.readPolicyState(), options) }
  getRobotId() { return this.runtime?.profile.id ?? this.robotId ?? null }
  getCapabilities() { return this.runtime?.profile.capabilities ?? [] }
  setLocomotionCommand(command) {
    if (!this.policyRunner?.setCommand) return false
    this.policyRunner.setCommand(command)
    return true
  }
  getSimStepHz() { return this.simStepHz }
  setFollowEnabled(value) { this.followEnabled = Boolean(value) }
  setRenderScale(value) { this.renderScale = Math.max(0.5, Math.min(2, value)); this.renderer.setPixelRatio(this.renderScale) }

  onResize() {
    this.camera.aspect = innerWidth / innerHeight
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(innerWidth, innerHeight)
  }

  render() {
    if (this.followEnabled && this.bodies?.[1]) this.controls.target.lerp(this.bodies[1].position, 0.05)
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  dispose() {
    this.alive = false
    removeEventListener('resize', this.resize)
    this.renderer.setAnimationLoop(null)
    this.controls.dispose()
    this.drag.dispose()
    this.renderer.dispose()
    this.disposeRobotRuntime(this.runtime)
    this.runtime = null
  }
}

export function pdTorque(target, position, velocity, stiffness, damping, [minimum, maximum]) {
  return Math.min(Math.max(stiffness * (target - position) - damping * velocity, minimum), maximum)
}

export function actuatorControlRange(model, actuator) {
  return model.actuator_ctrllimited[actuator]
    ? Array.from(model.actuator_ctrlrange.slice(actuator * 2, actuator * 2 + 2))
    : [-Infinity, Infinity]
}

export function isCurrentRuntime(expected, active) {
  return expected === active
}

export function threeToMujoco({ x, y, z }) {
  return new Float64Array([x, -z || 0, y])
}

export function dragForce(offset, strength = 60, maximum = 30) {
  const force = threeToMujoco({ x: offset.x * strength, y: offset.y * strength, z: offset.z * strength })
  const length = Math.hypot(...force)
  if (length > maximum) force.forEach((value, index) => { force[index] = value * maximum / length })
  return force
}

export function isUprightState(state, { thresholdDeg = 15, kneeThresholdRad = 0.5 } = {}) {
  const [w, x, y, z] = state.rootQuat
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))))
  const threshold = thresholdDeg * Math.PI / 180
  return Math.abs(roll) < threshold && Math.abs(pitch) < threshold
    && Math.abs(state.jointPos[9] ?? 0) < kneeThresholdRad
    && Math.abs(state.jointPos[10] ?? 0) < kneeThresholdRad
}

function decodeNames(rawNames, addresses, count) {
  const bytes = new Uint8Array(rawNames)
  const decoder = new TextDecoder()
  return Array.from({ length: count }, (_, index) => {
    const start = addresses[index]
    let end = start
    while (end < bytes.length && bytes[end]) end++
    return decoder.decode(bytes.subarray(start, end))
  })
}

function mapJoints(mujoco, model, jointNames, names) {
  const jointTransmission = mujoco.mjtTrn.mjTRN_JOINT.value
  const actuatorJoints = Array.from({ length: model.nu }, (_, index) => {
    if (model.actuator_trntype[index] !== jointTransmission) throw new Error(`Unsupported actuator ${index}`)
    return model.actuator_trnid[index * 2]
  })
  const ctrlAddresses = []
  const qposAddresses = []
  const qvelAddresses = []
  for (const name of names) {
    const joint = jointNames.indexOf(name)
    const actuator = actuatorJoints.indexOf(joint)
    if (joint < 0 || actuator < 0) throw new Error(`Joint not mapped: ${name}`)
    ctrlAddresses.push(actuator)
    qposAddresses.push(model.jnt_qposadr[joint])
    qvelAddresses.push(model.jnt_dofadr[joint])
  }
  return { ctrlAddresses, qposAddresses, qvelAddresses }
}

function readRuntimeState(runtime) {
  return {
    jointPos: Float32Array.from(runtime.qposAddresses, (address) => runtime.data.qpos[address]),
    jointVel: Float32Array.from(runtime.qvelAddresses, (address) => runtime.data.qvel[address]),
    rootPos: Float32Array.from(runtime.data.qpos.slice(0, 3)),
    rootQuat: Float32Array.from(runtime.data.qpos.slice(3, 7)),
    rootAngVel: Float32Array.from(runtime.data.qvel.slice(3, 6)),
  }
}
