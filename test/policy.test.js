import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'

import { PolicyRunner } from '../src/policy/runner.js'

test('AMASS policy assembles the deployed 475-value observation', async () => {
  const config = JSON.parse(await readFile(
    new URL('../public/examples/checkpoints/g1/tracking_policy_amass.json', import.meta.url),
    'utf8',
  ))
  config.tracking.motions = {
    default: {
      joint_pos: [Array(29).fill(0)],
      root_pos: [[0, 0, 0.78]],
      root_quat: [[1, 0, 0, 0]],
    },
  }

  const runner = new PolicyRunner(config)

  assert.equal(runner.numActions, 29)
  assert.equal(runner.numObservations, 475)
})
