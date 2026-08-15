export function parsePolicyConfig(raw) {
  if (!raw || !Array.isArray(raw.policy_joint_names) || !raw.policy_joint_names.length) {
    throw new Error('Policy configuration requires policy_joint_names')
  }
  if (!raw.onnx?.path || !raw.onnx?.meta) throw new Error('Policy configuration requires an ONNX model')
  return raw
}

export function parseMotionIndex(raw, indexUrl) {
  if (raw?.format !== 'tracking-motion-index-v1' || !Array.isArray(raw.motions)) {
    throw new Error('Invalid tracking motion index')
  }
  const base = new URL(raw.base_path?.replace(/\/?$/, '/') ?? './', indexUrl)
  return raw.motions.map((entry) => {
    const file = typeof entry === 'string' ? entry : entry.file ?? entry.path
    const name = typeof entry === 'string'
      ? entry.split('/').pop().replace(/\.json$/i, '')
      : entry.name ?? file.split('/').pop().replace(/\.json$/i, '')
    if (!file || !name) throw new Error('Motion index entries require a name and file')
    return { name, url: new URL(file, base).toString() }
  })
}

export async function loadPolicyConfig(url, fetchImpl = fetch) {
  const response = await fetchImpl(url)
  if (!response.ok) throw new Error(`Failed to load policy config: ${response.status}`)
  return parsePolicyConfig(await response.json())
}

export async function loadMotions(url, fetchImpl = fetch) {
  const response = await fetchImpl(url)
  if (!response.ok) throw new Error(`Failed to load motion index: ${response.status}`)
  const entries = parseMotionIndex(await response.json(), response.url || new URL(url, location.href))
  return Object.fromEntries(await Promise.all(entries.map(async ({ name, url: clipUrl }) => {
    const clip = await fetchImpl(clipUrl)
    if (!clip.ok) throw new Error(`Failed to load motion clip: ${clip.status}`)
    return [name, await clip.json()]
  })))
}

export async function populateMujocoFilesystem(mujoco, root = '/examples/scenes', destination = '', fetchImpl = fetch) {
  const index = await fetchImpl(`${root}/files.json`)
  if (!index.ok) throw new Error(`Failed to load scene index: ${index.status}`)
  const files = await index.json()
  const responses = await Promise.all(files.map((file) => fetchImpl(`${root}/${file}`)))
  for (let index = 0; index < files.length; index++) {
    const file = files[index]
    const parts = file.split('/')
    let directory = destination ? `/working/${destination}` : '/working'
    if (!mujoco.FS.analyzePath(directory).exists) mujoco.FS.mkdir(directory)
    for (const part of parts.slice(0, -1)) {
      directory += `/${part}`
      if (!mujoco.FS.analyzePath(directory).exists) mujoco.FS.mkdir(directory)
    }
    const response = responses[index]
    if (!response.ok) throw new Error(`Failed to load scene asset ${file}: ${response.status}`)
    const value = /\.(png|stl|skn)$/i.test(file)
      ? new Uint8Array(await response.arrayBuffer())
      : await response.text()
    mujoco.FS.writeFile(`${destination ? `/working/${destination}` : '/working'}/${file}`, value)
  }
}
