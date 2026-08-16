import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'

test('demo exposes capability-driven G1 and H1 controls', async () => {
  const source = await readFile(new URL('../src/components/Demo.vue', import.meta.url), 'utf8')

  assert.match(source, /ROBOT_PROFILES/)
  assert.match(source, /hasCapability/)
  assert.match(source, /Robot/)
  assert.match(source, /motionUpload/)
  assert.match(source, /textMotion/)
  assert.match(source, /locomotion/)
  assert.match(source, /switchRobot/)
})
