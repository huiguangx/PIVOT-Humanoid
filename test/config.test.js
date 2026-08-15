import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'

import { parseMotionIndex, parsePolicyConfig, populateMujocoFilesystem } from '../src/simulation/assets.js'

const readJson = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'))

test('deployed policy config exposes the 29-joint ONNX tracking contract', async () => {
  const raw = await readJson('../public/examples/checkpoints/g1/tracking_policy_amass.json')
  const config = parsePolicyConfig(raw)

  assert.equal(config.policy_joint_names.length, 29)
  assert.equal(config.onnx.path, './examples/checkpoints/g1/policy_amass.onnx')
  assert.equal(config.tracking.transition_steps, 100)
})

test('deployed motion index resolves all 17 local clips', async () => {
  const raw = await readJson('../public/examples/checkpoints/g1/motions.json')
  const entries = parseMotionIndex(raw, 'https://app.test/examples/checkpoints/g1/motions.json')

  assert.equal(entries.length, 17)
  assert.equal(entries[0].url, 'https://app.test/examples/checkpoints/g1/motions/dance1_subject1.json')
  assert.equal(entries.at(-1).name, 'TaiChi_CMU_12_04_amass')
})

test('deployed H1 policy exposes the official recurrent locomotion contract', async () => {
  const raw = await readJson('../public/examples/checkpoints/h1/locomotion_policy.json')
  const config = parsePolicyConfig(raw)

  assert.equal(config.policy_joint_names.length, 10)
  assert.deepEqual(config.onnx.meta.in_keys, ['policy', 'hidden_state', 'cell_state'])
  assert.deepEqual(config.onnx.meta.out_keys, ['action', 'next_hidden_state', 'next_cell_state'])
  assert.deepEqual(config.onnx.recurrent.hidden_state, [1, 1, 64])
  assert.equal(config.control_dt, 0.02)
})

test('scene population rejects failed asset responses', async () => {
  const mujoco = { FS: { analyzePath: () => ({ exists: true }), mkdir() {}, writeFile() {} } }
  const fetch = async (url) => url.endsWith('files.json')
    ? new Response(JSON.stringify(['missing.stl']))
    : new Response('', { status: 404 })

  await assert.rejects(() => populateMujocoFilesystem(mujoco, '/scenes', '', fetch), /missing\.stl: 404/)
})

test('scene population writes robot assets into an isolated destination', async () => {
  const directories = new Set(['/working', '/working/g1'])
  const writes = new Map([['/working/g1/keep.xml', 'g1']])
  const mujoco = { FS: {
    analyzePath: (path) => ({ exists: directories.has(path) }),
    mkdir: (path) => directories.add(path),
    writeFile: (path, value) => writes.set(path, value),
  } }
  const fetch = async (url) => url.endsWith('files.json')
    ? new Response(JSON.stringify(['scene.xml', 'meshes/pelvis.stl']))
    : new Response(url)

  await populateMujocoFilesystem(mujoco, '/examples/scenes/h1', 'h1', fetch)

  assert.equal(writes.get('/working/g1/keep.xml'), 'g1')
  assert.equal(writes.get('/working/h1/scene.xml'), '/examples/scenes/h1/scene.xml')
  assert.ok(writes.get('/working/h1/meshes/pelvis.stl') instanceof Uint8Array)
})
