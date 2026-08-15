# Multi-Robot Platform Design

## Goal

Extend PIVOT from a G1-only application into an isolated multi-robot Web platform. The first release supports:

- Unitree G1: existing reference-motion tracking, generated/uploaded motions, and drag perturbations.
- Unitree H1: standing, walking, turning, and drag perturbations with Unitree's official locomotion policy.

This phase does not attempt cross-robot motion retargeting or a universal control policy.

## Robot Packs

Each robot is described by one registry entry containing its display name, scene path, policy configuration, control driver, and capabilities. Assets and runtime state are namespaced by robot ID.

```text
G1 pack: scene + tracking ONNX + tracking config + motion library
H1 pack: scene + locomotion ONNX + locomotion config
```

Capabilities determine which controls the UI exposes. G1 supports motion selection and upload. H1 supports velocity and turning commands. Both support reset and drag perturbations.

## Control Drivers

The existing `PolicyRunner` remains the G1 reference-tracking driver. A separate H1 locomotion driver implements the same minimal lifecycle:

```text
init()
reset(state)
step(state)
```

The H1 driver reproduces Unitree's official inference pipeline: a 41-value observation containing angular velocity, projected gravity, velocity command, joint position, joint velocity, previous action, and gait phase; its policy produces 10 joint targets.

The official TorchScript `motion.pt` policy is converted to ONNX offline. The browser ships and runs only the ONNX artifact. Conversion is accepted only when representative PyTorch and ONNX outputs agree within a documented numerical tolerance.

## Transactional Switching

Switching creates a candidate runtime without modifying the active runtime:

1. Load the robot's namespaced MuJoCo assets.
2. Create candidate `MjModel` and `MjData` objects.
3. Validate joint and actuator mappings.
4. Initialize the robot's control driver.
5. Run one policy inference and validate output shape and finite values.
6. Commit the candidate and dispose the previous runtime.

If any step fails, the candidate is disposed and the active robot continues running. A monotonically increasing switch ID prevents an older asynchronous load from replacing a newer selection.

Robot runtimes do not share policy inputs, previous actions, motion libraries, commands, timers, or MuJoCo state. The renderer and canvas may be reused only after the candidate runtime has passed validation.

## Interface

The existing simulation view gains a compact G1/H1 selector. During a switch, only the selector and robot-specific controls are disabled; the current scene remains visible.

- G1 shows the existing motion browser, prompt integration, and upload controls.
- H1 shows stand, planar speed, and yaw-rate controls.
- Unsupported controls are not rendered.
- A failed switch reports the robot and failing stage while retaining the current robot.

Generated or uploaded G1 motions are never passed to the H1 locomotion driver.

## Assets And Licensing

H1 model, mesh, configuration, and policy sources come from Unitree's official repositories. Required BSD 3-Clause attribution is retained in `NOTICE`; upstream names are not used to imply endorsement.

No backend or runtime Python dependency is introduced.

## Verification

The implementation is accepted when:

- Existing G1 tests and behavior remain unchanged.
- G1 -> H1 -> G1 switching succeeds repeatedly.
- A deliberately invalid H1 pack leaves the active G1 runtime usable.
- Stale overlapping switch requests cannot commit out of order.
- H1 remains upright when standing and responds to forward, backward, and yaw commands.
- Drag perturbations affect both robots without stopping their control loops.
- Repeated switching does not leave duplicate animation loops, timers, policies, or MuJoCo objects.
- Production build and browser tests pass on desktop and mobile viewports.

## Deferred Work

- Arbitrary H1 whole-body motion tracking.
- Cross-robot motion retargeting.
- Morphology-conditioned universal policies.
- Additional robot families beyond G1 and H1.
