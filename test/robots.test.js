import assert from 'node:assert/strict'
import { test } from 'node:test'

import { getRobotProfile, hasCapability, ROBOT_PROFILES } from '../src/robots/registry.js'

test('registry exposes isolated G1 and H1 profiles', () => {
  const g1 = getRobotProfile('g1')
  const h1 = getRobotProfile('h1')

  assert.equal(g1.driver, 'tracking')
  assert.equal(h1.driver, 'locomotion')
  assert.notEqual(g1.policy, h1.policy)
  assert.equal(hasCapability(g1, 'motionUpload'), true)
  assert.equal(hasCapability(h1, 'motionUpload'), false)
  assert.deepEqual(Object.keys(ROBOT_PROFILES), ['g1', 'h1'])
  assert.throws(() => getRobotProfile('missing'), /Unknown robot/)
})
