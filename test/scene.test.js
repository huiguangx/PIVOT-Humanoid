import assert from 'node:assert/strict'
import { test } from 'node:test'
import { MeshPhysicalMaterial, Scene, Vector3 } from 'three'

import { buildScene } from '../src/simulation/scene.js'

test('MuJoCo ground is horizontal in Three.js coordinates', () => {
  const model = {
    nbody: 1,
    ngeom: 1,
    geom_group: [0],
    geom_bodyid: [0],
    geom_size: [1, 1, 0.1],
    geom_type: [0],
    geom_pos: [0, 0, 0],
    geom_quat: [1, 0, 0, 0],
    geom_rgba: [1, 1, 1, 1],
    geom_matid: [0],
    mat_rgba: [1, 1, 1, 1], mat_texid: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], mat_texrepeat: [5, 5],
    mat_specular: [0], mat_reflectance: [0.2], mat_shininess: [0], mat_metallic: [0],
    tex_width: [1, 1], tex_height: [1, 1], tex_adr: [0, 3], tex_nchannel: [3, 3], tex_data: [0, 0, 0, 64, 96, 128],
  }
  const data = { xpos: [0, 0, 0], xquat: [1, 0, 0, 0] }
  const mujoco = { mjtGeom: { mjGEOM_PLANE: { value: 0 } } }

  const { bodies } = buildScene(mujoco, model, data, new Scene())
  const ground = bodies[0].children[0]
  const normal = new Vector3(0, 0, 1).applyQuaternion(ground.quaternion)

  assert.ok(Math.abs(normal.y) > 0.99)
  assert.equal(ground.isReflector, true)
  assert.deepEqual(ground.material.uniforms.groundTextureRepeat.value.toArray(), [70, 70])
})

test('MuJoCo visual meshes retain material, texture, normals, and UVs', () => {
  const model = {
    nbody: 1, ngeom: 1, nlight: 0,
    geom_group: [0], geom_bodyid: [0], geom_size: [1, 1, 1], geom_type: [7],
    geom_pos: [0, 0, 0], geom_quat: [1, 0, 0, 0], geom_rgba: [1, 1, 1, 1],
    geom_dataid: [0], geom_matid: [0],
    mat_rgba: [0.2, 0.3, 0.4, 1], mat_texid: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    mat_texrepeat: [5, 5], mat_specular: [0.6], mat_reflectance: [0.2], mat_shininess: [0.4], mat_metallic: [0.1],
    mesh_vertadr: [0], mesh_vertnum: [3], mesh_vert: [0, 0, 0, 1, 0, 0, 0, 1, 0],
    mesh_normal: [0, 0, 1, 0, 0, 1, 0, 0, 1], mesh_texcoordadr: [0], mesh_texcoord: [0, 0, 1, 0, 0, 1],
    mesh_faceadr: [0], mesh_facenum: [1], mesh_face: [0, 1, 2],
    tex_width: [1], tex_height: [1], tex_adr: [0], tex_nchannel: [3], tex_data: [255, 128, 0],
  }
  const data = { xpos: [0, 0, 0], xquat: [1, 0, 0, 0] }
  const mujoco = { mjtGeom: {
    mjGEOM_PLANE: { value: 0 }, mjGEOM_SPHERE: { value: 2 }, mjGEOM_CAPSULE: { value: 3 },
    mjGEOM_CYLINDER: { value: 5 }, mjGEOM_BOX: { value: 6 }, mjGEOM_MESH: { value: 7 },
  } }

  const mesh = buildScene(mujoco, model, data, new Scene()).bodies[0].children[0]
  assert.ok(mesh.material instanceof MeshPhysicalMaterial)
  assert.ok(mesh.material.map)
  assert.deepEqual(Array.from(mesh.geometry.attributes.uv.array), model.mesh_texcoord)
  assert.equal(mesh.material.color.getHexString(), '7c95aa')
  assert.equal(mesh.bodyID, 0)
})

test('MuJoCo scene lights are recovered from the model', () => {
  const model = {
    nbody: 1, ngeom: 0, nlight: 1,
    light_type: [1], light_diffuse: [0.9, 0.9, 0.9], light_attenuation: [0], light_castshadow: [0],
  }
  const data = { xpos: [0, 0, 0], xquat: [1, 0, 0, 0], light_xpos: [2, 3, 4], light_xdir: [-0.4, -0.6, -1] }
  const result = buildScene({}, model, data, new Scene())

  assert.equal(result.lights?.length, 1)
  assert.equal(result.lights?.[0].type, 'DirectionalLight')
  assert.equal(result.lights?.[0].intensity, 0.9)
})
