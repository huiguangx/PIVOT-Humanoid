# Multi-Robot Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add isolated Unitree G1/H1 selection to PIVOT while preserving all existing G1 behavior and giving H1 standing, walking, turning, and drag perturbations.

**Architecture:** A small robot registry describes assets, driver type, and capabilities. `SimulationController` prepares a complete candidate runtime and commits it only after model, joint mapping, and policy validation succeed. G1 keeps the existing reference-tracking runner; H1 gets a separate official-formula locomotion runner backed by an offline-converted ONNX policy.

**Tech Stack:** Vue 3, Vuetify, Three.js, MuJoCo WebAssembly, ONNX Runtime Web, Node test runner, one-time Python/PyTorch ONNX conversion.

## Global Constraints

- G1 behavior and existing motion generation/upload remain unchanged.
- H1 first release supports standing, walking, turning, and drag perturbations only.
- Robot assets, policy state, commands, motions, and timers must not be shared.
- Failed or stale switches must leave the active robot running.
- No backend or runtime Python dependency is introduced.
- Unitree BSD 3-Clause attribution must remain in `NOTICE`.

---

### Task 1: Robot Registry And Capabilities

**Files:**
- Create: `src/robots/registry.js`
- Create: `test/robots.test.js`

**Interfaces:**
- Produces: `ROBOT_PROFILES`, `DEFAULT_ROBOT_ID`, `getRobotProfile(id)`, and `hasCapability(profile, capability)`.
- Profile fields: `{ id, label, scene, assetRoot, assetDestination, policy, driver, capabilities }`.

- [ ] **Step 1: Write the failing registry tests**

```js
test('registry exposes isolated G1 and H1 profiles', () => {
  const g1 = getRobotProfile('g1')
  const h1 = getRobotProfile('h1')
  assert.equal(g1.driver, 'tracking')
  assert.equal(h1.driver, 'locomotion')
  assert.notEqual(g1.policy, h1.policy)
  assert.equal(hasCapability(g1, 'motionUpload'), true)
  assert.equal(hasCapability(h1, 'motionUpload'), false)
  assert.throws(() => getRobotProfile('missing'), /Unknown robot/)
})
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `node --test test/robots.test.js`

Expected: FAIL because `src/robots/registry.js` does not exist.

- [ ] **Step 3: Implement the immutable registry**

```js
export const DEFAULT_ROBOT_ID = 'g1'

export const ROBOT_PROFILES = Object.freeze({
  g1: Object.freeze({
    id: 'g1', label: 'Unitree G1', driver: 'tracking',
    scene: 'g1/g1.xml', assetRoot: '/examples/scenes', assetDestination: '',
    policy: '/examples/checkpoints/g1/tracking_policy_amass.json',
    capabilities: Object.freeze(['motions', 'motionUpload', 'textMotion', 'drag']),
  }),
  h1: Object.freeze({
    id: 'h1', label: 'Unitree H1', driver: 'locomotion',
    scene: 'h1/scene.xml', assetRoot: '/examples/scenes/h1', assetDestination: 'h1',
    policy: '/examples/checkpoints/h1/locomotion_policy.json',
    capabilities: Object.freeze(['stand', 'locomotion', 'drag']),
  }),
})

export function getRobotProfile(id) {
  const profile = ROBOT_PROFILES[id]
  if (!profile) throw new Error(`Unknown robot: ${id}`)
  return profile
}

export function hasCapability(profile, capability) {
  return profile.capabilities.includes(capability)
}
```

- [ ] **Step 4: Run registry and full tests**

Run: `node --test test/robots.test.js && npm test`

Expected: PASS except the previously documented 70x70 versus 140x140 ground-texture test if it remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/robots/registry.js test/robots.test.js
git commit -m "feat: register isolated robot profiles"
```

### Task 2: H1 Locomotion Driver

**Files:**
- Create: `src/policy/locomotion.js`
- Modify: `src/policy/runner.js`
- Create: `test/locomotion.test.js`

**Interfaces:**
- Produces: `LocomotionRunner(config, model?)`, `setCommand({ forward, lateral, yaw })`, `init()`, `reset(state)`, `step(state)`, and `buildLocomotionObservation(state, command, previousAction, phase, config)`.
- `step(state)` returns a `Float32Array` of H1 joint-position targets.
- Existing `PolicyRunner` remains unchanged for G1.

- [ ] **Step 1: Write observation and command tests**

```js
test('H1 observation follows Unitree 41-value layout', () => {
  const obs = buildLocomotionObservation(state, { forward: 0.5, lateral: 0, yaw: 0 }, new Float32Array(10), 0.25, config)
  assert.equal(obs.length, 41)
  assert.deepEqual(Array.from(obs.slice(6, 9)), [1, 0, 0])
  assert.ok(Math.abs(obs[39] - 1) < 1e-6)
  assert.ok(Math.abs(obs[40]) < 1e-6)
})

test('H1 output is scaled around default joint angles', async () => {
  const action = Float32Array.from({ length: 10 }, () => 1)
  const model = { run: async () => [{ action: { data: action } }, {}] }
  const runner = new LocomotionRunner(config, model)
  runner.reset(state)
  const target = await runner.step(state)
  assert.equal(target.length, 10)
  assert.ok(Math.abs(target[0] - (config.default_joint_pos[0] + config.action_scale)) < 1e-6)
})
```

- [ ] **Step 2: Verify the tests fail**

Run: `node --test test/locomotion.test.js`

Expected: FAIL because the locomotion module does not exist.

- [ ] **Step 3: Implement the official H1 observation formula**

Build exactly this ordered vector:

```text
[scaled root angular velocity: 3]
[projected gravity: 3]
[scaled forward/lateral/yaw command: 3]
[scaled joint position offset: 10]
[scaled joint velocity: 10]
[previous action: 10]
[sin/cos gait phase: 2]
```

Clamp commands to profile limits before observation construction. Reject non-finite or incorrectly sized policy outputs before updating `previousAction`.

- [ ] **Step 4: Reuse the existing ONNX wrapper**

Export the internal `OnnxModel` from `src/policy/runner.js` and construct it from the H1 policy configuration. Do not introduce a second ONNX loading implementation.
`LocomotionRunner.step()` passes `{ policy: new Tensor('float32', observation, [1, 41]) }` and reads `output.action.data` from the existing `[output, recurrent]` return value.

- [ ] **Step 5: Run focused and full tests**

Run: `node --test test/locomotion.test.js test/policy.test.js && npm test`

Expected: New locomotion tests pass and the G1 475-value observation test remains green.

- [ ] **Step 6: Commit**

```bash
git add src/policy/locomotion.js src/policy/runner.js test/locomotion.test.js
git commit -m "feat: add isolated H1 locomotion runner"
```

### Task 3: Transactional Robot Runtime Switching

**Files:**
- Modify: `src/simulation/assets.js`
- Modify: `src/simulation/controller.js`
- Modify: `src/simulation/scene.js`
- Modify: `test/config.test.js`
- Modify: `test/controller.test.js`

**Interfaces:**
- Produces: `SimulationController.create(container, robotId = DEFAULT_ROBOT_ID)`.
- Produces: `switchRobot(robotId): Promise<boolean>`, `getRobotId()`, `getCapabilities()`, and `setLocomotionCommand(command)`.
- Candidate runtime shape: `{ profile, model, data, runner, jointNames, policyJointNames, ctrlAddresses, qposAddresses, qvelAddresses, kp, kd, controlType, root, bodies, lights }`.

- [ ] **Step 1: Add asset destination tests**

Verify `populateMujocoFilesystem` can fetch `/examples/scenes/h1/files.json` and write `scene.xml` to `/working/h1/scene.xml` without touching existing `/working/g1/*` files.

- [ ] **Step 2: Add switching rollback tests**

Use a controller object with a working active runtime and an injected candidate loader that throws. Assert `switchRobot('h1')` rejects while the active runtime identity and robot ID stay `g1`.

Add an overlapping-switch test where request 1 resolves after request 2. Assert only request 2 commits.

- [ ] **Step 3: Verify focused failures**

Run: `node --test test/config.test.js test/controller.test.js`

Expected: FAIL because destination loading and `switchRobot` do not exist.

- [ ] **Step 4: Make scene asset loading namespaced**

Change the loader signature to:

```js
populateMujocoFilesystem(mujoco, root = '/examples/scenes', destination = '', fetchImpl = fetch)
```

Write each listed file beneath `/working/${destination}` and preserve the current G1 default behavior.

- [ ] **Step 5: Prepare runtimes without mutating active state**

Refactor joint mapping and state reading into helpers accepting explicit candidate model/data objects. Select the runner with one switch:

```js
const runner = profile.driver === 'tracking'
  ? new PolicyRunner(config)
  : new LocomotionRunner(config)
```

Load model, data, configuration, motions, joint maps, and runner into local candidate variables. Run one inference, validate target length and finite values, then reset the candidate runner.

- [ ] **Step 6: Build the candidate Three.js root off-scene and commit atomically**

Call `buildScene` with a temporary `Scene`. After it succeeds and the switch ID is current, detach its root, remove the active root, assign candidate state, attach the new root to the live scene, and dispose the previous MuJoCo objects. If preparation fails or becomes stale, dispose only candidate objects.

- [ ] **Step 7: Preserve controller compatibility**

Keep `policyRunner` as an alias for the active runner so existing G1 component code and tests continue to work. Route the main loop, reset, state reads, PD control, drag, and follow camera through active runtime fields.

- [ ] **Step 8: Run controller and full tests**

Run: `node --test test/config.test.js test/controller.test.js test/policy.test.js && npm test`

Expected: Switching tests pass and G1 tests retain their previous result.

- [ ] **Step 9: Commit**

```bash
git add src/simulation/assets.js src/simulation/controller.js src/simulation/scene.js test/config.test.js test/controller.test.js
git commit -m "feat: switch robot runtimes transactionally"
```

### Task 4: Official H1 Assets And ONNX Policy

**Files:**
- Create: `tools/convert_h1_policy.py`
- Create: `public/examples/scenes/h1/files.json`
- Create: `public/examples/scenes/h1/scene.xml`
- Create: `public/examples/scenes/h1/h1.xml`
- Create: `public/examples/scenes/h1/meshes/*`
- Create: `public/examples/checkpoints/h1/locomotion_policy.json`
- Create: `public/examples/checkpoints/h1/motion.onnx`
- Modify: `NOTICE`
- Modify: `test/project.test.js`

**Interfaces:**
- H1 config supplies 10 ordered joint names, PD gains, default positions, observation scales, command limits, gait period `0.8`, policy input/output metadata, and ONNX path.

- [ ] **Step 1: Add project asset assertions**

```js
assert.ok(existsSync(new URL('../public/examples/scenes/h1/scene.xml', import.meta.url)))
assert.ok(existsSync(new URL('../public/examples/checkpoints/h1/motion.onnx', import.meta.url)))
assert.ok(existsSync(new URL('../public/examples/checkpoints/h1/locomotion_policy.json', import.meta.url)))
```

- [ ] **Step 2: Download official Unitree H1 model assets and policy**

Use files from `unitreerobotics/unitree_rl_gym` at a recorded commit. Copy only `resources/robots/h1/{scene.xml,h1.xml,meshes/*}` and `deploy/pre_train/h1/motion.pt`. Generate `files.json` from the selected scene files.

- [ ] **Step 3: Add a one-time conversion script**

The script loads TorchScript on CPU, exports a dynamic-batch `[batch, 41]` input to a `[batch, 10]` ONNX output, then runs representative fixed observations through TorchScript and ONNX Runtime and asserts maximum absolute error is below `1e-4`.

- [ ] **Step 4: Convert and verify the policy**

Run: `python tools/convert_h1_policy.py /tmp/motion.pt public/examples/checkpoints/h1/motion.onnx`

Expected: conversion prints input/output shapes and `max_abs_error < 1e-4`.

- [ ] **Step 5: Record the exact H1 configuration**

Use Unitree's official values: timestep `0.002`, decimation `10`, 10 actions, 41 observations, action scale `0.25`, command scale `[2, 2, 0.25]`, angular-velocity scale `0.25`, joint-position scale `1`, joint-velocity scale `0.05`, and official KP/KD/default-angle arrays.

- [ ] **Step 6: Add required attribution**

Append the Unitree Robotics BSD 3-Clause copyright, source repository, and modification note to `NOTICE` without changing existing notices.

- [ ] **Step 7: Run asset, policy, and build checks**

Run: `node --test test/project.test.js test/locomotion.test.js && npm run build`

Expected: H1 assets exist, ONNX loads through the configured path, and production build succeeds.

- [ ] **Step 8: Commit**

```bash
git add tools/convert_h1_policy.py public/examples/scenes/h1 public/examples/checkpoints/h1 NOTICE test/project.test.js
git commit -m "feat: add official H1 simulation pack"
```

### Task 5: Capability-Driven Multi-Robot UI And Final Verification

**Files:**
- Modify: `src/components/Demo.vue`
- Modify: `src/styles.css`
- Modify: `README.md`
- Create: `test/ui-source.test.js`

**Interfaces:**
- UI state adds `robotId`, `switchingRobot`, and `locomotionCommand`.
- G1 controls remain bound to the active tracking runner.
- H1 controls call `simulation.setLocomotionCommand({ forward, lateral, yaw })`.

- [ ] **Step 1: Add source-level capability tests**

Assert the component imports `ROBOT_PROFILES`/`hasCapability`, renders a robot selector, guards motion upload and text generation with `motionUpload`/`textMotion`, and exposes H1 locomotion controls only for `locomotion`.

- [ ] **Step 2: Verify the UI test fails**

Run: `node --test test/ui-source.test.js`

Expected: FAIL because the selector and capability guards are absent.

- [ ] **Step 3: Add the compact robot selector**

Place a Vuetify segmented control or compact select near the top of the existing controls panel. Disable only robot-specific controls during switching. Keep the old scene visible until commit and report failed switches through the existing message/error surface.

- [ ] **Step 4: Gate controls by capabilities**

Render all existing motion, upload, and generation UI only for G1 capabilities. For H1 render stand plus bounded forward/lateral/yaw controls. Reset per-robot UI state after a successful switch; do not clear the inactive robot's stored state.

- [ ] **Step 5: Update documentation**

Document G1 and H1 capabilities, the H1 official source/attribution, and the fact that H1 currently provides locomotion rather than arbitrary full-body motion tracking in both English and Chinese README sections.

- [ ] **Step 6: Run automated verification**

Run: `npm test && npm run build`

Expected: all new tests pass; investigate but do not silently change the pre-existing ground-texture expectation if it remains the sole failure.

- [ ] **Step 7: Run browser verification**

Start the Vite server and verify desktop/mobile views. Exercise G1 -> H1 -> G1, H1 standing/walking/turning, drag perturbations, an intentionally failed switch, and repeated switching. Confirm canvas pixels are nonblank and controls do not overlap.

- [ ] **Step 8: Commit**

```bash
git add src/components/Demo.vue src/styles.css README.md test/ui-source.test.js
git commit -m "feat: expose isolated G1 and H1 controls"
```

- [ ] **Step 9: Final repository check**

Run: `git status --short && git log --oneline -7`

Expected: clean worktree with the design, plan, and five implementation commits. Do not push without explicit user approval.
