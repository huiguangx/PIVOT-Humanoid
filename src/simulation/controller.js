import createMujoco from '../vendor/mujoco.js'
import { PolicyRunner } from '../policy/runner.js'
import { loadMotions, loadPolicyConfig, populateMujocoFilesystem } from './assets.js'
import { buildScene, createRenderer, DragInteraction, updateBodies } from './scene.js'

const DEFAULT_POLICY = '/examples/checkpoints/g1/tracking_policy_amass.json'

export class SimulationController {
  static async create(container) {
    const mujoco = await createMujoco()
    if (!mujoco.FS.analyzePath('/working').exists) mujoco.FS.mkdir('/working')
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working')
    await populateMujocoFilesystem(mujoco)
    const controller = new SimulationController(mujoco, container)
    await controller.init()
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
    this.resize = () => this.onResize()
    addEventListener('resize', this.resize)
    this.renderer.setAnimationLoop(() => this.render())
  }

  async init() {
    await this.loadScene('g1/g1.xml')
    await this.reloadPolicy(DEFAULT_POLICY)
    this.alive = true
  }

  async loadScene(path) {
    this.drag?.end()
    this.model?.delete()
    this.data?.delete()
    this.scene.getObjectByName('MuJoCo Root')?.removeFromParent()
    this.model = this.mujoco.MjModel.loadFromXML(`/working/${path}`)
    this.data = new this.mujoco.MjData(this.model)
    ;({ root: this.mujocoRoot, bodies: this.bodies, lights: this.lights } = buildScene(this.mujoco, this.model, this.data, this.scene))
    this.jointNames = decodeNames(this.model.names, this.model.name_jntadr, this.model.njnt)
    this.timestep = this.model.opt.timestep
    this.decimation = Math.max(1, Math.round(0.02 / this.timestep))
  }

  async reloadPolicy(path = DEFAULT_POLICY, { onnxPath } = {}) {
    const config = await loadPolicyConfig(path)
    if (onnxPath) config.onnx = { ...config.onnx, path: onnxPath }
    if (config.tracking?.motions_path && !config.tracking.motions) {
      config.tracking.motions = await loadMotions(new URL(config.tracking.motions_path, location.href))
    }
    this.#mapJoints(config.policy_joint_names)
    this.kp = Float32Array.from(config.stiffness)
    this.kd = Float32Array.from(config.damping)
    this.controlType = config.control_type ?? 'joint_position'
    this.policyRunner = new PolicyRunner(config)
    await this.policyRunner.init()
    this.policyRunner.reset(this.readPolicyState())
    this.currentPolicyPath = path
    this.params.current_motion = 'default'
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
      if (!this.params.paused && this.policyRunner) {
        const target = await this.policyRunner.step(this.readPolicyState())
        for (let step = 0; step < this.decimation; step++) {
          this.data.qfrc_applied.fill(0)
          if (target && this.controlType === 'joint_position') this.#applyControl(target)
          this.#applyDragForce()
          this.mujoco.mj_step(this.model, this.data)
        }
        updateBodies(this.model, this.data, this.bodies, this.lights)
        frames++
        const elapsed = performance.now() - sampleStart
        if (elapsed >= 500) { this.simStepHz = frames * 1000 / elapsed; frames = 0; sampleStart = performance.now() }
      }
      const wait = Math.max(0, this.timestep * this.decimation * 1000 - (performance.now() - started))
      await new Promise((resolve) => setTimeout(resolve, wait))
    }
  }

  #applyControl(target) {
    for (let index = 0; index < this.policyJointNames.length; index++) {
      const actuator = this.ctrlAddresses[index]
      const range = this.model.actuator_ctrlrange.slice(actuator * 2, actuator * 2 + 2)
      this.data.ctrl[actuator] = pdTorque(
        target[index], this.data.qpos[this.qposAddresses[index]], this.data.qvel[this.qvelAddresses[index]],
        this.kp[index], this.kd[index], range,
      )
    }
  }

  #applyDragForce() {
    const object = this.drag.physicsObject
    if (!object?.bodyID) return
    updateBodies(this.model, this.data, this.bodies)
    this.drag.update()
    this.mujoco.mj_applyFT(
      this.model,
      this.data,
      dragForce(this.drag.currentWorld.clone().sub(this.drag.worldHit)),
      this.zeroTorque,
      threeToMujoco(this.drag.worldHit),
      object.bodyID,
      this.data.qfrc_applied,
    )
  }

  #mapJoints(names) {
    const jointTransmission = this.mujoco.mjtTrn.mjTRN_JOINT.value
    const actuatorJoints = Array.from({ length: this.model.nu }, (_, index) => {
      if (this.model.actuator_trntype[index] !== jointTransmission) throw new Error(`Unsupported actuator ${index}`)
      return this.model.actuator_trnid[index * 2]
    })
    this.policyJointNames = names.slice()
    this.ctrlAddresses = []
    this.qposAddresses = []
    this.qvelAddresses = []
    for (const name of names) {
      const joint = this.jointNames.indexOf(name)
      const actuator = actuatorJoints.indexOf(joint)
      if (joint < 0 || actuator < 0) throw new Error(`Joint not mapped: ${name}`)
      this.ctrlAddresses.push(actuator)
      this.qposAddresses.push(this.model.jnt_qposadr[joint])
      this.qvelAddresses.push(this.model.jnt_dofadr[joint])
    }
  }

  readPolicyState() {
    return {
      jointPos: Float32Array.from(this.qposAddresses, (address) => this.data.qpos[address]),
      jointVel: Float32Array.from(this.qvelAddresses, (address) => this.data.qvel[address]),
      rootPos: Float32Array.from(this.data.qpos.slice(0, 3)),
      rootQuat: Float32Array.from(this.data.qpos.slice(3, 7)),
      rootAngVel: Float32Array.from(this.data.qvel.slice(3, 6)),
    }
  }

  resetSimulation() {
    this.mujoco.mj_resetData(this.model, this.data)
    this.mujoco.mj_forward(this.model, this.data)
    this.policyRunner.reset(this.readPolicyState())
    this.params.current_motion = 'default'
  }

  isUpright(options) { return isUprightState(this.readPolicyState(), options) }
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
    this.model?.delete()
    this.data?.delete()
  }
}

export function pdTorque(target, position, velocity, stiffness, damping, [minimum, maximum]) {
  return Math.min(Math.max(stiffness * (target - position) - damping * velocity, minimum), maximum)
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
