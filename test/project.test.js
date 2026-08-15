import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { test } from 'node:test'

test('standalone project owns its runtime inputs', () => {
  assert.ok(existsSync(new URL('../public/examples/scenes/files.json', import.meta.url)))
  assert.ok(existsSync(new URL('../public/examples/checkpoints/g1/policy_amass.onnx', import.meta.url)))
  assert.ok(existsSync(new URL('../public/assets/ort-wasm-simd-threaded.jsep.mjs', import.meta.url)))
  assert.ok(existsSync(new URL('../public/assets/ort-wasm-simd-threaded.jsep.wasm', import.meta.url)))
})
