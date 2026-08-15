import assert from 'node:assert/strict'
import { test } from 'node:test'

import { TextMotionClient } from '../src/text-motion/client.js'

test('generate sends the deployed payload and session headers', async () => {
  const calls = []
  const fetch = async (url, init) => {
    calls.push({ url, init })
    return new Response(JSON.stringify({ success: true, motion_id: 'm1', motion: {} }))
  }
  const client = new TextMotionClient('https://service.test', 'public-token', fetch)
  client.sessionId = 's1'

  const motion = await client.generate('walk forward', 4)

  assert.equal(calls[0].url, 'https://service.test/api/generate')
  assert.deepEqual(calls[0].init.headers, {
    Authorization: 'Bearer public-token',
    'Content-Type': 'application/json',
    'X-Session-ID': 's1',
  })
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    text: 'walk forward',
    motion_length: 4,
    num_inference_steps: 10,
    adaptive_smooth: true,
    static_start: true,
    static_frames: 2,
    blend_frames: 8,
    transition_steps: 100,
  })
  assert.equal(motion.motion_id, 'm1')
  assert.equal(motion.text_prompt, 'walk forward')
})

test('forbidden sessions are cleared before reporting expiry', async () => {
  const client = new TextMotionClient('https://service.test', '', async () => (
    new Response(JSON.stringify({ code: 'SESSION_FORBIDDEN' }), { status: 403 })
  ))
  client.sessionId = 'expired'

  await assert.rejects(() => client.listMotions(), /Session expired/)
  assert.equal(client.sessionId, null)
})

test('browser fetch is called with the global receiver', async () => {
  const fetch = async function () {
    assert.equal(this, globalThis)
    return new Response(JSON.stringify({ motions: [] }))
  }
  const client = new TextMotionClient('https://service.test', '', fetch)

  assert.deepEqual(await client.listMotions(), [])
})
