<template>
  <div id="mujoco-container" />

  <aside class="controls">
    <v-card class="controls-card">
      <v-card-title class="controls-title">
        <strong>PIVOT:</strong>
        <span>Retargeting-Free Language-to-Motion</span>
        <span>for Closed-Loop Humanoid Control</span>
        <div class="links">
          <span class="disabled-link"><v-icon icon="mdi-file-document-outline" /> Paper (Coming Soon)</span>
          <a href="https://github.com/huiguangx/PIVOT-Humanoid" target="_blank" rel="noopener noreferrer" aria-label="Code – GitHub repository">
            <v-icon icon="mdi-github" /> Code
          </a>
        </div>
      </v-card-title>

      <v-card-text class="controls-body">
        <section class="usage">
          <h3>Usage</h3>
          <ul>
            <li>Type a text description and press Enter (or tap Generate) to create a motion.</li>
            <li>You can switch to a new motion anytime; playback returns to default when it ends.</li>
            <li>Shortcuts below can also be typed in the text field.</li>
          </ul>
          <p>Shortcut buttons:</p>
          <ul class="shortcut-help">
            <li><kbd>default</kbd> — rest pose</li><li><kbd>up</kbd> — get up after a fall</li>
            <li><kbd>last</kbd> — replay last generated motion</li><li><kbd>list</kbd> — show generated motion list</li>
            <li><kbd>status</kbd> — current motion state</li><li><kbd>clear</kbd> — clear generated motions</li>
          </ul>
        </section>

        <v-divider />
        <section>
          <span class="section-label">Shortcuts</span>
          <div class="button-row">
            <v-btn v-for="command in commands" :key="command" size="x-small" variant="tonal" color="primary" :disabled="state !== 1" @click="runCommand(command)">{{ command }}</v-btn>
          </div>
        </section>

        <template v-if="availableMotions.length">
          <v-divider />
          <v-select v-model="currentMotion" :items="availableMotions" label="Motion" density="compact" hide-details @update:model-value="selectMotion" />
          <v-file-input v-model="uploadFiles" class="motion-upload" label="Import motion JSON" accept="application/json,.json" multiple density="compact" hide-details prepend-icon="mdi-upload" @update:model-value="uploadMotions" />
          <v-alert v-if="uploadMessage" :type="uploadType" density="compact" closable class="mt-2" @click:close="uploadMessage = ''">{{ uploadMessage }}</v-alert>
        </template>

        <v-divider />
        <section>
          <div class="generate-heading">
            <span class="section-label">Generate</span>
            <v-chip size="x-small" :color="statusColor" variant="flat">{{ statusLabel }}</v-chip>
          </div>
          <template v-if="showGenerator">
            <v-textarea v-model="prompt" label="Text description" placeholder="e.g. a person walks forward" rows="2" density="compact" hide-details :disabled="generating" @keydown.enter.prevent="submitPrompt" />
            <p class="example-label">Examples (tap to generate):</p>
            <div class="example-row"><v-chip v-for="example in examples" :key="example" size="x-small" variant="tonal" @click="generate(example)">{{ example }}</v-chip></div>
            <v-number-input v-model="motionLength" label="Duration (s)" :min="0.1" :max="9" :step="0.1" density="compact" hide-details />
            <v-btn color="primary" size="small" block :loading="generating" :disabled="!prompt.trim() || state !== 1" @click="submitPrompt"><v-icon icon="mdi-send" /> Generate</v-btn>
          </template>
          <v-btn v-else variant="text" size="small" color="primary" @click="showGenerator = true"><v-icon icon="mdi-robot" /> Generate motions with AI</v-btn>
          <v-alert v-if="textError" type="error" density="compact" closable class="mt-2" @click:close="textError = ''">{{ textError }}</v-alert>
        </section>

        <section v-if="generated.length" class="generated">
          <div class="generate-heading"><span class="section-label">Generated</span><v-chip size="x-small">{{ generated.length }}/10</v-chip></div>
          <div class="example-row"><v-chip v-for="motion in generated" :key="motion.motion_id" size="x-small" variant="tonal" @click="playGenerated(motion)"><v-icon icon="mdi-play-circle" />{{ motion.text_prompt || motion.motion_id }}</v-chip></div>
        </section>
        <p v-if="message" class="status-message">{{ message }}</p>
      </v-card-text>
      <v-card-actions><v-btn color="primary" block :disabled="state !== 1" @click="reset">Reset</v-btn></v-card-actions>
    </v-card>
  </aside>

  <v-dialog :model-value="state === 0" persistent max-width="600">
    <v-card title="Loading Simulation Environment"><v-card-text><v-progress-linear indeterminate color="primary" /><p>Loading MuJoCo and ONNX policy, please wait.</p></v-card-text></v-card>
  </v-dialog>
  <v-dialog :model-value="state < 0" persistent max-width="600">
    <v-card title="Simulation Environment Loading Error"><v-card-text>{{ error }}</v-card-text></v-card>
  </v-dialog>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { SimulationController } from '../simulation/controller.js'
import { TextMotionClient } from '../text-motion/client.js'
import { TEXT_MOTION_TOKEN, TEXT_MOTION_URL } from '../text-motion/config.js'

const state = ref(0), error = ref(''), currentMotion = ref('default'), availableMotions = ref([])
const prompt = ref(''), motionLength = ref(4), showGenerator = ref(false), textStatus = ref('disconnected')
const textError = ref(''), generating = ref(false), generated = ref([]), lastGenerated = ref(null), message = ref('')
const uploadFiles = ref([]), uploadMessage = ref(''), uploadType = ref('success')
const commands = ['default', 'up', 'last', 'list', 'status', 'clear']
const examples = ['walk in a circle', 'do jumping jacks', 'a person is jogging on the spot']
const getUpMotions = new Set(['fallAndGetUp2_subject2', 'fallAndGetUp1_subject1'])
const client = new TextMotionClient(TEXT_MOTION_URL, TEXT_MOTION_TOKEN)
let simulation, trackingTimer, uprightChecks = 0, monitoringUpright = false

const statusLabel = computed(() => generating.value ? 'Generating...' : textStatus.value === 'connected' ? 'Ready' : textStatus.value === 'error' ? 'Error' : 'Not Connected')
const statusColor = computed(() => generating.value ? 'warning' : textStatus.value === 'connected' ? 'success' : textStatus.value === 'error' ? 'error' : 'grey')
const tracker = () => simulation?.policyRunner?.tracking

onMounted(async () => {
  if (!globalThis.WebAssembly) { state.value = -2; error.value = 'Your browser does not support WebAssembly.'; return }
  try {
    simulation = await SimulationController.create(document.getElementById('mujoco-container'))
    simulation.start((cause) => {
      console.error(cause)
      state.value = -1
      error.value = `Simulation stopped: ${cause}`
    })
    availableMotions.value = tracker().availableMotions()
    currentMotion.value = simulation.params.current_motion
    state.value = 1
    trackingTimer = setInterval(updateTrackingState, 33)
    await connectTextMotion()
  } catch (cause) {
    console.error(cause)
    state.value = -1
    error.value = String(cause)
  }
})

onBeforeUnmount(() => { clearInterval(trackingTimer); simulation?.dispose() })

async function connectTextMotion() {
  try {
    await client.createSession()
    textStatus.value = 'connected'
    const items = await client.listMotions()
    for (const item of items.slice(-10)) {
      const motion = await client.getMotion(item.motion_id)
      addGenerated({ ...motion, ...item })
    }
  } catch (cause) { console.warn(cause); textStatus.value = 'disconnected' }
}

function selectMotion(name) {
  if (!tracker()?.requestMotion(name, simulation.readPolicyState())) return false
  simulation.params.current_motion = name
  currentMotion.value = name
  if (!getUpMotions.has(name)) { monitoringUpright = false; uprightChecks = 0 }
  return true
}

function runCommand(command) {
  if (command === 'default') selectMotion('default')
  else if (command === 'up') {
    const name = availableMotions.value.includes('fallAndGetUp2_subject2') ? 'fallAndGetUp2_subject2' : availableMotions.value.includes('fallAndGetUp1_subject1') ? 'fallAndGetUp1_subject1' : null
    if (!name) return showMessage('No get-up motion loaded.')
    if (selectMotion(name)) { monitoringUpright = true; uprightChecks = 0; showMessage('Get-up loaded. Will switch to default when standing.') }
  }
  else if (command === 'last') lastGenerated.value ? playGenerated(lastGenerated.value) : showMessage('No generated motion to replay.')
  else if (command === 'list') showMessage(generated.value.length ? generated.value.map((item) => item.text_prompt || item.motion_id).join(' | ') : 'No generated motions yet.')
  else if (command === 'status') { const value = tracker()?.playbackState(); showMessage(value ? `Motion: ${value.currentName}; ${value.currentDone ? 'done' : 'playing'}` : 'Status: not ready') }
  else if (command === 'clear') clearGenerated()
}

async function submitPrompt() {
  const value = prompt.value.trim()
  if (!value) return
  if (commands.includes(value.toLowerCase())) { runCommand(value.toLowerCase()); prompt.value = ''; return }
  await generate(value)
}

async function generate(text) {
  generating.value = true; textError.value = ''
  try {
    const motion = await client.generate(text, motionLength.value)
    addGenerated(motion); playGenerated(motion); prompt.value = ''; textStatus.value = 'connected'
  } catch (cause) { textError.value = cause.message; textStatus.value = 'error' }
  finally { generating.value = false }
}

function addGenerated(motion) {
  const converted = tracker().convertMotionJointPosPolicyToDataset(motion.joint_pos)
  const result = tracker().addMotions({ [motion.motion_id]: { joint_pos: converted, root_pos: motion.root_pos, root_quat: motion.root_quat } }, { overwrite: true })
  if (!result.added.length) throw new Error('Generated motion has invalid frame data')
  generated.value = [...generated.value.filter((item) => item.motion_id !== motion.motion_id), motion].slice(-10)
  lastGenerated.value = motion
}

function playGenerated(motion) { selectMotion(motion.motion_id) }
async function clearGenerated() {
  textError.value = ''
  try { await client.clearMotions() }
  catch (cause) { textError.value = cause.message; textStatus.value = 'error'; return }
  generated.value = []; lastGenerated.value = null; showMessage('Cleared generated motions for this session.')
}

async function uploadMotions(value) {
  const files = Array.isArray(value) ? value : value ? [value] : []
  if (!files.length) return
  let added = 0, skipped = 0, invalid = 0
  for (const file of files) {
    try {
      const raw = JSON.parse(await file.text())
      const base = file.name.replace(/\.[^/.]+$/, '').trim() || 'motion'
      const name = base.startsWith('[new] ') ? base : `[new] ${base}`
      const result = tracker().addMotions({ [name]: raw })
      added += result.added.length; skipped += result.skipped.length; invalid += result.invalid.length
    } catch (cause) { console.warn('Failed to import motion JSON:', cause); invalid++ }
  }
  if (added) availableMotions.value = tracker().availableMotions()
  const parts = []
  if (added) parts.push(`Added ${added} motion${added === 1 ? '' : 's'}`)
  if (skipped) parts.push(`Skipped ${skipped} duplicate${skipped === 1 ? '' : 's'}`)
  if (invalid) parts.push(`Ignored ${invalid} invalid file${invalid === 1 ? '' : 's'}`)
  uploadMessage.value = parts.length ? `${parts.join('. ')}.` : 'No motions were added.'
  uploadType.value = invalid ? 'warning' : added ? 'success' : 'info'
  uploadFiles.value = []
}

function updateTrackingState() {
  const playback = tracker()?.playbackState()
  if (!playback?.available) return
  if (monitoringUpright && getUpMotions.has(playback.currentName)) {
    uprightChecks = simulation.isUpright({ thresholdDeg: 15, kneeThresholdRad: 0.6 }) ? uprightChecks + 1 : 0
    if (uprightChecks >= 8) { selectMotion('default'); showMessage('Standing detected. Switched to default.'); return }
  }
  if (monitoringUpright && (!getUpMotions.has(playback.currentName) || playback.currentDone)) { monitoringUpright = false; uprightChecks = 0 }
  if (!playback.isDefault && playback.currentDone && selectMotion('default')) showMessage('Motion done. Switched to default.')
}

function showMessage(value) { message.value = value; setTimeout(() => { message.value = '' }, 4000) }
function reset() { simulation?.resetSimulation(); currentMotion.value = 'default' }
</script>
