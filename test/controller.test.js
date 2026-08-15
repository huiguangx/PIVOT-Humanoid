import assert from 'node:assert/strict'
import { test } from 'node:test'

import * as controllerModule from '../src/simulation/controller.js'

const { SimulationController, isUprightState, pdTorque } = controllerModule

test('PD torque is clipped to the actuator range', () => {
  assert.equal(pdTorque(2, 0, 0, 100, 2, [-50, 50]), 50)
  assert.equal(pdTorque(-2, 0, 0, 100, 2, [-50, 50]), -50)
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
