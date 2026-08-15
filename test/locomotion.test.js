import assert from 'node:assert/strict'
import { test } from 'node:test'

import { buildLocomotionObservation, LocomotionRunner } from '../src/policy/locomotion.js'

const config = {
  policy_joint_names: Array.from({ length: 10 }, (_, index) => `joint_${index}`),
  default_joint_pos: [0, 0, -0.1, 0.3, -0.2, 0, 0, -0.1, 0.3, -0.2],
  action_scale: 0.25,
  control_dt: 0.02,
  gait_period: 0.8,
  command_scale: [2, 2, 0.25],
  command_limits: { forward: [-1, 1], lateral: [-0.5, 0.5], yaw: [-1, 1] },
  observation_scale: { angularVelocity: 0.25, jointPosition: 1, jointVelocity: 0.05 },
  onnx: { path: '/h1.onnx', meta: { in_keys: ['policy'], out_keys: ['action'] } },
}

const state = {
  rootQuat: [1, 0, 0, 0],
  rootAngVel: [0, 0, 0],
  jointPos: Float32Array.from(config.default_joint_pos),
  jointVel: new Float32Array(10),
}

test('H1 observation follows Unitree 41-value layout', () => {
  const obs = buildLocomotionObservation(
    state,
    { forward: 0.5, lateral: 0, yaw: 0 },
    new Float32Array(10),
    0.25,
    config,
  )

  assert.equal(obs.length, 41)
  Array.from(obs.slice(3, 9)).forEach((value, index) => {
    assert.ok(Math.abs(value - [0, 0, -1, 1, 0, 0][index]) < 1e-6)
  })
  assert.ok(Math.abs(obs[39] - 1) < 1e-6)
  assert.ok(Math.abs(obs[40]) < 1e-6)
})

test('H1 commands are clamped and policy output is scaled around default angles', async () => {
  const action = Float32Array.from({ length: 10 }, () => 1)
  const model = { run: async () => [{ action: { data: action } }, {}] }
  const runner = new LocomotionRunner(config, model)
  runner.setCommand({ forward: 4, lateral: -4, yaw: 4 })
  runner.reset(state)

  const target = await runner.step(state)

  assert.deepEqual(runner.command, { forward: 1, lateral: -0.5, yaw: 1 })
  assert.equal(target.length, 10)
  assert.ok(Math.abs(target[0] - 0.25) < 1e-6)
  assert.ok(Math.abs(target[2] - 0.15) < 1e-6)
})

test('H1 runner rejects invalid policy output without updating previous action', async () => {
  const model = { run: async () => [{ action: { data: new Float32Array([NaN]) } }, {}] }
  const runner = new LocomotionRunner(config, model)
  runner.reset(state)

  await assert.rejects(() => runner.step(state), /Invalid locomotion policy output/)
  assert.deepEqual(Array.from(runner.previousAction), Array(10).fill(0))
})

test('H1 runner owns and advances its recurrent state', async () => {
  const recurrentConfig = {
    ...config,
    onnx: {
      ...config.onnx,
      recurrent: { hidden_state: [1, 1, 64], cell_state: [1, 1, 64] },
    },
  }
  let feeds
  const nextHidden = { data: Float32Array.from({ length: 64 }, () => 2) }
  const nextCell = { data: Float32Array.from({ length: 64 }, () => 3) }
  const model = { run: async (value) => {
    feeds = value
    return [{
      action: { data: new Float32Array(10) },
      next_hidden_state: nextHidden,
      next_cell_state: nextCell,
    }, {}]
  } }
  const runner = new LocomotionRunner(recurrentConfig, model)
  runner.reset(state)

  await runner.step(state)

  assert.deepEqual(feeds.hidden_state.dims, [1, 1, 64])
  assert.deepEqual(feeds.cell_state.dims, [1, 1, 64])
  assert.equal(runner.recurrentInput.hidden_state, nextHidden)
  assert.equal(runner.recurrentInput.cell_state, nextCell)
})
