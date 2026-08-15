export const DEFAULT_ROBOT_ID = 'g1'

export const ROBOT_PROFILES = Object.freeze({
  g1: Object.freeze({
    id: 'g1',
    label: 'Unitree G1',
    driver: 'tracking',
    scene: 'g1/g1.xml',
    assetRoot: '/examples/scenes',
    assetDestination: '',
    policy: '/examples/checkpoints/g1/tracking_policy_amass.json',
    capabilities: Object.freeze(['motions', 'motionUpload', 'textMotion', 'drag']),
  }),
  h1: Object.freeze({
    id: 'h1',
    label: 'Unitree H1',
    driver: 'locomotion',
    scene: 'h1/scene.xml',
    assetRoot: '/examples/scenes/h1',
    assetDestination: 'h1',
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
