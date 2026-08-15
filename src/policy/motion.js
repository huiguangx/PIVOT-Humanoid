export function normalizeMotion(raw, jointCount = 0) {
  if (!raw || typeof raw !== 'object') return null

  const jointPos = toFrames(raw.joint_pos ?? raw.jointPos)
  const rootPos = toFrames(raw.root_pos ?? raw.rootPos)
  const rootQuat = toFrames(raw.root_quat ?? raw.rootQuat)
  const frameCount = jointPos?.length
  const width = jointCount || jointPos?.[0]?.length
  if (!frameCount || rootPos?.length !== frameCount || rootQuat?.length !== frameCount
    || !validFrames(jointPos, width) || !validFrames(rootPos, 3) || !validFrames(rootQuat, 4)) return null

  return { jointPos, rootPos, rootQuat }
}

export function lerpFrames(from, to, count) {
  if (count <= 0) return []
  return Array.from({ length: count }, (_, index) => {
    const amount = (index + 1) / (count + 1)
    return Float32Array.from(from, (value, item) => (1 - amount) * value + amount * to[item])
  })
}

export function slerpFrames(from, to, count) {
  if (count <= 0) return []
  const start = normalizeQuaternion(from)
  let end = normalizeQuaternion(to)
  let dot = start.reduce((sum, value, index) => sum + value * end[index], 0)
  if (dot < 0) {
    dot = -dot
    end = end.map((value) => -value)
  }

  if (1 - dot < 1e-6) {
    return lerpFrames(start, end, count).map((frame) => Float32Array.from(normalizeQuaternion(frame)))
  }

  const angle = Math.acos(dot)
  const sinAngle = Math.sin(angle)
  return Array.from({ length: count }, (_, index) => {
    const amount = (index + 1) / (count + 1)
    const left = Math.sin((1 - amount) * angle) / sinAngle
    const right = Math.sin(amount * angle) / sinAngle
    return Float32Array.from(start, (value, item) => left * value + right * end[item])
  })
}

export function normalizeQuaternion(value) {
  const length = Math.hypot(...value)
  if (length < 1e-9) return [1, 0, 0, 0]
  return Array.from(value, (item) => item / length)
}

function toFrames(value) {
  return Array.isArray(value) ? value.map((frame) => Float32Array.from(frame)) : null
}

function validFrames(frames, width) {
  return width > 0 && frames.every((frame) => frame.length === width && Array.from(frame).every(Number.isFinite))
}
