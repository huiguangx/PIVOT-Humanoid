import {
  AmbientLight, ArrowHelper, BoxGeometry, BufferGeometry, CapsuleGeometry, Color, DataTexture,
  CylinderGeometry, DirectionalLight, Float32BufferAttribute, Group, HemisphereLight, Mesh,
  MeshPhysicalMaterial, PCFSoftShadowMap, PerspectiveCamera, PlaneGeometry, PointLight, Quaternion,
  Raycaster, RepeatWrapping, RGBAFormat, Scene, SphereGeometry, SpotLight, Uint32BufferAttribute,
  UnsignedByteType, Vector2, Vector3, WebGLRenderer,
} from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Reflector } from 'three/examples/jsm/objects/Reflector.js'

export function createRenderer(container) {
  const scene = new Scene()
  scene.background = new Color(0.15, 0.25, 0.35)
  scene.add(new AmbientLight(0xffffff, 0.25))
  const camera = new PerspectiveCamera(45, innerWidth / innerHeight, 0.001, 100)
  camera.position.set(3, innerWidth < 500 || innerHeight < 700 ? 1.95 : 2.2, 3)
  scene.add(camera)
  const renderer = new WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(innerWidth < 500 || innerHeight < 700 ? 1 : 2)
  renderer.setSize(innerWidth, innerHeight)
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = PCFSoftShadowMap
  container.appendChild(renderer.domElement)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.set(0, innerHeight < 700 ? 0.88 : 0.7, 0)
  controls.panSpeed = 2
  controls.zoomSpeed = 1
  controls.enableDamping = true
  controls.dampingFactor = 0.1
  controls.screenSpacePanning = true
  controls.update()
  return { scene, camera, renderer, controls }
}

export function buildScene(mujoco, model, data, scene) {
  const root = new Group()
  root.name = 'MuJoCo Root'
  scene.add(root)
  const bodies = {}
  for (let body = 0; body < model.nbody; body++) bodies[body] = new Group()

  for (let index = 0; index < model.ngeom; index++) {
    if (model.geom_group[index] >= 3) continue
    const body = model.geom_bodyid[index]
    const size = Array.from(model.geom_size.slice(index * 3, index * 3 + 3))
    const isPlane = typeIsPlane(mujoco, model, index)
    const geometry = geometryFor(mujoco, model, index, size)
    const material = materialFor(model, index)
    const mesh = isPlane && material.map
      ? new Reflector(geometry, { clipBias: 3e-3, texture: material.map, shader: reflectorShader(material.map) })
      : new Mesh(geometry, material)
    mesh.castShadow = true
    mesh.receiveShadow = true
    mesh.bodyID = body
    setPosition(model.geom_pos, index, mesh.position, false)
    if (isPlane) mesh.rotateX(-Math.PI / 2)
    else setQuaternion(model.geom_quat, index, mesh.quaternion, false)
    bodies[body].add(mesh)
  }
  Object.values(bodies).forEach((body) => root.add(body))
  const lights = buildLights(model, root)
  updateBodies(model, data, bodies, lights)
  return { root, bodies, lights }
}

export class DragInteraction {
  constructor(scene, camera, canvas, controls) {
    this.scene = scene
    this.camera = camera
    this.canvas = canvas
    this.controls = controls
    this.raycaster = new Raycaster()
    this.pointer = new Vector2()
    this.localHit = new Vector3()
    this.worldHit = new Vector3()
    this.currentWorld = new Vector3()
    this.arrow = new ArrowHelper(new Vector3(0, 1, 0), new Vector3(), 1, 0xff3fbb)
    this.arrow.visible = false
    scene.add(this.arrow)
    this.onPointerDown = (event) => this.start(event)
    this.onPointerMove = (event) => this.move(event)
    this.onPointerUp = () => this.end()
    canvas.addEventListener('pointerdown', this.onPointerDown, true)
    document.addEventListener('pointermove', this.onPointerMove, true)
    document.addEventListener('pointerup', this.onPointerUp, true)
    document.addEventListener('pointercancel', this.onPointerUp, true)
  }

  updateRay(event) {
    const bounds = this.canvas.getBoundingClientRect()
    this.pointer.set(
      (event.clientX - bounds.left) / bounds.width * 2 - 1,
      -(event.clientY - bounds.top) / bounds.height * 2 + 1,
    )
    this.raycaster.setFromCamera(this.pointer, this.camera)
  }

  start(event) {
    if (event.button !== 0) return
    this.updateRay(event)
    const hit = this.raycaster.intersectObjects(this.scene.children, true).find(({ object }) => object.bodyID > 0)
    if (!hit) return
    event.preventDefault()
    this.physicsObject = hit.object
    this.grabDistance = hit.distance
    this.localHit.copy(this.physicsObject.worldToLocal(hit.point.clone()))
    this.worldHit.copy(hit.point)
    this.currentWorld.copy(hit.point)
    this.arrow.position.copy(hit.point)
    this.arrow.visible = true
    this.controls.enabled = false
  }

  move(event) {
    if (!this.physicsObject) return
    this.updateRay(event)
    this.currentWorld.copy(this.raycaster.ray.origin).addScaledVector(this.raycaster.ray.direction, this.grabDistance)
    this.update()
  }

  update() {
    if (!this.physicsObject) return
    this.worldHit.copy(this.physicsObject.localToWorld(this.localHit.clone()))
    const offset = this.currentWorld.clone().sub(this.worldHit)
    this.arrow.position.copy(this.worldHit)
    if (offset.lengthSq() > 1e-8) {
      this.arrow.setDirection(offset.clone().normalize())
      this.arrow.setLength(offset.length())
    }
  }

  end() {
    this.physicsObject = null
    this.arrow.visible = false
    this.controls.enabled = true
  }

  dispose() {
    this.end()
    this.canvas.removeEventListener('pointerdown', this.onPointerDown, true)
    document.removeEventListener('pointermove', this.onPointerMove, true)
    document.removeEventListener('pointerup', this.onPointerUp, true)
    document.removeEventListener('pointercancel', this.onPointerUp, true)
    this.arrow.removeFromParent()
  }
}

export function updateBodies(model, data, bodies, lights = []) {
  for (let body = 0; body < model.nbody; body++) {
    if (!bodies[body]) continue
    setPosition(data.xpos, body, bodies[body].position)
    setQuaternion(data.xquat, body, bodies[body].quaternion)
  }
  for (let index = 0; index < lights.length; index++) {
    const light = lights[index]
    if (!data.light_xpos || !data.light_xdir) continue
    setPosition(data.light_xpos, index, light.position)
    const direction = convertedVector(data.light_xdir, index)
    if (light.target) light.target.position.copy(light.position).add(direction)
    else light.lookAt(light.position.clone().add(direction))
  }
}

function geometryFor(mujoco, model, index, size) {
  const type = model.geom_type[index]
  if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) return new PlaneGeometry(Math.max(size[0] * 2, 100), Math.max(size[1] * 2, 100))
  if (type === mujoco.mjtGeom.mjGEOM_SPHERE.value) return new SphereGeometry(size[0], 24, 16)
  if (type === mujoco.mjtGeom.mjGEOM_CAPSULE.value) return new CapsuleGeometry(size[0], size[1] * 2, 8, 16)
  if (type === mujoco.mjtGeom.mjGEOM_CYLINDER.value) return new CylinderGeometry(size[0], size[0], size[1] * 2, 20)
  if (type === mujoco.mjtGeom.mjGEOM_BOX.value) return new BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2)
  if (type === mujoco.mjtGeom.mjGEOM_MESH.value) return meshGeometry(model, model.geom_dataid[index])
  return new SphereGeometry(size[0] || 0.01, 12, 8)
}

function typeIsPlane(mujoco, model, index) {
  return model.geom_type[index] === mujoco.mjtGeom.mjGEOM_PLANE.value
}

function reflectorShader(texture) {
  return {
    uniforms: {
      color: { value: new Color(0x7f7f7f) },
      tDiffuse: { value: null },
      textureMatrix: { value: null },
      groundTexture: { value: texture },
      groundTextureRepeat: { value: texture.repeat },
    },
    vertexShader: `
      uniform mat4 textureMatrix;
      uniform vec2 groundTextureRepeat;
      varying vec4 vReflectUv;
      varying vec2 vGroundUv;
      void main() {
        vReflectUv = textureMatrix * vec4(position, 1.0);
        vGroundUv = uv * groundTextureRepeat;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D tDiffuse;
      uniform sampler2D groundTexture;
      varying vec4 vReflectUv;
      varying vec2 vGroundUv;
      void main() {
        vec3 reflection = texture2DProj(tDiffuse, vReflectUv).rgb;
        vec3 ground = texture2D(groundTexture, vGroundUv).rgb;
        gl_FragColor = vec4(mix(reflection, ground, 0.5), 1.0);
      }
    `,
  }
}

function meshGeometry(model, meshId) {
  const geometry = new BufferGeometry()
  const vertexStart = model.mesh_vertadr[meshId]
  const vertexCount = model.mesh_vertnum[meshId]
  const vertices = []
  for (let i = 0; i < vertexCount; i++) {
    const offset = (vertexStart + i) * 3
    vertices.push(model.mesh_vert[offset], model.mesh_vert[offset + 2], -model.mesh_vert[offset + 1])
  }
  const faceStart = model.mesh_faceadr[meshId]
  const faceCount = model.mesh_facenum[meshId]
  const indices = []
  for (let i = 0; i < faceCount * 3; i++) indices.push(model.mesh_face[faceStart * 3 + i])
  geometry.setAttribute('position', new Float32BufferAttribute(vertices, 3))
  if (model.mesh_normal?.length >= (vertexStart + vertexCount) * 3) {
    const normals = []
    for (let i = 0; i < vertexCount; i++) {
      const offset = (vertexStart + i) * 3
      normals.push(model.mesh_normal[offset], model.mesh_normal[offset + 2], -model.mesh_normal[offset + 1])
    }
    geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3))
  }
  const uvStart = model.mesh_texcoordadr?.[meshId]
  if (uvStart >= 0 && model.mesh_texcoord?.length >= (uvStart + vertexCount) * 2) {
    geometry.setAttribute('uv', new Float32BufferAttribute(model.mesh_texcoord.slice(uvStart * 2, (uvStart + vertexCount) * 2), 2))
  }
  geometry.setIndex(new Uint32BufferAttribute(indices, 1))
  if (!geometry.attributes.normal) geometry.computeVertexNormals()
  return geometry
}

function materialFor(model, geom) {
  const materialId = model.geom_matid?.[geom] ?? -1
  const rgba = materialId >= 0 ? model.mat_rgba.slice(materialId * 4, materialId * 4 + 4) : model.geom_rgba.slice(geom * 4, geom * 4 + 4)
  const map = materialId >= 0 ? textureFor(model, materialId) : null
  const options = {
    color: new Color(rgba[0], rgba[1], rgba[2]),
    opacity: rgba[3],
    transparent: rgba[3] < 1,
    map,
  }
  if (materialId >= 0) {
    options.specularIntensity = model.mat_specular?.[materialId] ?? 1
    options.reflectivity = model.mat_reflectance?.[materialId] ?? 0.5
    options.roughness = 1 + (model.mat_shininess?.[materialId] ?? 0)
    options.metalness = model.mat_metallic?.[materialId] ?? 0
  }
  return new MeshPhysicalMaterial(options)
}

function textureFor(model, materialId) {
  const textureId = model.mat_texid?.[materialId * 10 + 1] ?? -1
  if (textureId < 0) return null
  const width = model.tex_width[textureId], height = model.tex_height[textureId]
  const address = model.tex_adr[textureId], channels = model.tex_nchannel[textureId]
  const rgba = new Uint8Array(width * height * 4)
  for (let pixel = 0; pixel < width * height; pixel++) {
    const source = address + pixel * channels, target = pixel * 4
    rgba[target] = model.tex_data[source]
    rgba[target + 1] = channels > 1 ? model.tex_data[source + 1] : rgba[target]
    rgba[target + 2] = channels > 2 ? model.tex_data[source + 2] : rgba[target]
    rgba[target + 3] = channels > 3 ? model.tex_data[source + 3] : 255
  }
  const texture = new DataTexture(rgba, width, height, RGBAFormat, UnsignedByteType)
  const repeat = model.mat_texrepeat?.slice(materialId * 2, materialId * 2 + 2) ?? [1, 1]
  texture.repeat.set(textureId === 1 ? 70 : repeat[0], textureId === 1 ? 70 : repeat[1])
  texture.wrapS = texture.wrapT = RepeatWrapping
  texture.needsUpdate = true
  return texture
}

function buildLights(model, root) {
  const types = [SpotLight, DirectionalLight, PointLight, HemisphereLight]
  return Array.from({ length: model.nlight ?? 0 }, (_, index) => {
    const Light = types[model.light_type[index]] ?? DirectionalLight
    const light = new Light()
    const diffuse = model.light_diffuse?.slice(index * 3, index * 3 + 3) ?? [1, 1, 1]
    const intensity = Math.max(...diffuse)
    light.color.setRGB(...(intensity ? diffuse.map((value) => value / intensity) : [1, 1, 1]))
    light.intensity = intensity || 1
    light.decay = (model.light_attenuation?.[index] ?? 0) * 100
    light.penumbra = 0.5
    light.castShadow = (model.light_castshadow?.[index] ?? 1) !== 0
    if (light.shadow) {
      light.shadow.mapSize.set(1024, 1024)
      light.shadow.camera.near = 0.1
      light.shadow.camera.far = 10
    }
    root.add(light)
    if (light.target) root.add(light.target)
    return light
  })
}

function convertedVector(values, index) {
  const offset = index * 3
  return new Vector3(values[offset], values[offset + 2], -values[offset + 1])
}

function setPosition(values, index, target, convert = true) {
  const offset = index * 3
  return convert
    ? target.set(values[offset], values[offset + 2], -values[offset + 1])
    : target.set(values[offset], values[offset + 2], -values[offset + 1])
}

function setQuaternion(values, index, target) {
  const offset = index * 4
  return target.set(-values[offset + 1], -values[offset + 3], values[offset + 2], -values[offset])
}
