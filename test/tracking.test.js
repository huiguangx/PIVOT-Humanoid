import assert from 'node:assert/strict'
import { test } from 'node:test'

import { MotionTracker } from '../src/policy/tracking.js'

const clip = (frames) => ({
  joint_pos: Array.from({ length: frames }, (_, index) => [index, index]),
  root_pos: Array.from({ length: frames }, (_, index) => [index, 0, 1]),
  root_quat: Array.from({ length: frames }, () => [1, 0, 0, 0]),
})

const robotState = {
  jointPos: new Float32Array([0, 0]),
  rootPos: new Float32Array([0, 0, 1]),
  rootQuat: new Float32Array([1, 0, 0, 0]),
}

test('tracker marks a motion done on its last frame', () => {
  const tracker = new MotionTracker({
    transition_steps: 0,
    dataset_joint_names: ['left', 'right'],
    motions: { default: clip(1), walk: clip(2) },
  })

  assert.equal(tracker.requestMotion('walk', robotState), true)
  assert.equal(tracker.playbackState().currentDone, false)
  tracker.advance()
  assert.deepEqual(tracker.playbackState(), {
    available: true,
    currentName: 'walk',
    currentDone: true,
    refIdx: 1,
    refLen: 2,
    transitionLen: 0,
    motionLen: 2,
    inTransition: false,
    isDefault: false,
  })
})

test('tracker rejects invalid and duplicate motions without replacing clips', () => {
  const tracker = new MotionTracker({
    transition_steps: 0,
    motions: { default: clip(1) },
  })

  assert.deepEqual(tracker.addMotions({ default: clip(3), broken: {} }), {
    added: [],
    skipped: ['default'],
    invalid: ['broken'],
  })
  assert.equal(tracker.requestMotion('missing', robotState), false)
  assert.equal(tracker.availableMotions().length, 1)
})

test('tracker rejects empty, mismatched, and wrong-width motion frames', () => {
  const tracker = new MotionTracker({
    transition_steps: 0,
    dataset_joint_names: ['left', 'right'],
    motions: { default: clip(1) },
  })

  assert.deepEqual(tracker.addMotions({
    empty: { joint_pos: [], root_pos: [], root_quat: [] },
    mismatched: { ...clip(2), root_pos: [[0, 0, 1]] },
    wrongWidth: { ...clip(1), joint_pos: [[0]] },
  }), { added: [], skipped: [], invalid: ['empty', 'mismatched', 'wrongWidth'] })
})
