import assert from 'node:assert/strict'
import { test } from 'node:test'

import * as controllerModule from '../src/simulation/controller.js'

const { SimulationController, actuatorControlRange, isCurrentRuntime, isUprightState, pdTorque } = controllerModule

test('PD torque is clipped to the actuator range', () => {
  assert.equal(pdTorque(2, 0, 0, 100, 2, [-50, 50]), 50)
  assert.equal(pdTorque(-2, 0, 0, 100, 2, [-50, 50]), -50)
})

test('unlimited torque motors are not clamped to MuJoCo default zero range', () => {
  const model = {
    actuator_ctrllimited: [0, 1],
    actuator_ctrlrange: [0, 0, -20, 20],
  }

  assert.deepEqual(actuatorControlRange(model, 0), [-Infinity, Infinity])
  assert.deepEqual(actuatorControlRange(model, 1), [-20, 20])
})

test('upright detection checks torso tilt and both knees', () => {
  const upright = { rootQuat: [1, 0, 0, 0], jointPos: Array(29).fill(0) }
  assert.equal(isUprightState(upright), true)
  upright.jointPos[9] = 1
  assert.equal(isUprightState(upright), false)
})

test('simulation loop failures are reported and pause the controller', async () => {
  const controller = Object.create(SimulationController.prototype)
  Object.assign(controller, {
    alive: true,
    params: { paused: false },
    policyRunner: { step: async () => { throw new Error('policy failed') } },
    readPolicyState: () => ({}),
  })

  let failure
  await controller.start((cause) => { failure = cause })
  assert.equal(failure?.message, 'policy failed')
  assert.equal(controller.params.paused, true)
})

test('drag force uses MuJoCo axes and is capped at 30N', () => {
  assert.equal(typeof controllerModule.dragForce, 'function')
  assert.deepEqual(Array.from(controllerModule.threeToMujoco({ x: 1, y: 2, z: 3 })), [1, -3, 2])
  assert.deepEqual(Array.from(controllerModule.dragForce({ x: 0, y: 0.1, z: 0 })), [0, 0, 6])
  assert.deepEqual(Array.from(controllerModule.dragForce({ x: 1, y: 0, z: 0 })), [30, 0, 0])
})

test('failed robot preparation leaves the active runtime unchanged', async () => {
  const active = { profile: { id: 'g1' } }
  const controller = Object.create(SimulationController.prototype)
  Object.assign(controller, {
    runtime: active,
    robotId: 'g1',
    switchVersion: 0,
    prepareRobotRuntime: async () => { throw new Error('bad H1 policy') },
  })

  await assert.rejects(() => controller.switchRobot('h1'), /bad H1 policy/)
  assert.equal(controller.runtime, active)
  assert.equal(controller.getRobotId(), 'g1')
})

test('a stale robot switch cannot replace a newer selection', async () => {
  let resolveH1
  const h1 = new Promise((resolve) => { resolveH1 = resolve })
  const committed = []
  const disposed = []
  const controller = Object.create(SimulationController.prototype)
  Object.assign(controller, {
    switchVersion: 0,
    prepareRobotRuntime: ({ id }) => id === 'h1' ? h1 : Promise.resolve({ profile: { id } }),
    commitRobotRuntime: (runtime) => committed.push(runtime.profile.id),
    disposeRobotRuntime: (runtime) => disposed.push(runtime.profile.id),
  })

  const older = controller.switchRobot('h1')
  const newer = controller.switchRobot('g1')
  resolveH1({ profile: { id: 'h1' } })

  assert.equal(await newer, true)
  assert.equal(await older, false)
  assert.deepEqual(committed, ['g1'])
  assert.deepEqual(disposed, ['h1'])
})

test('an in-flight old policy target is stale after a robot switch', () => {
  const oldRuntime = { profile: { id: 'g1' } }
  const newRuntime = { profile: { id: 'h1' } }

  assert.equal(isCurrentRuntime(oldRuntime, oldRuntime), true)
  assert.equal(isCurrentRuntime(oldRuntime, newRuntime), false)
})
