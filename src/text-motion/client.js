import { SESSION_STORAGE_KEY } from './config.js'

export class TextMotionClient {
  constructor(baseUrl, token = '', fetchImpl = globalThis.fetch, storage = globalThis.sessionStorage) {
    this.baseUrl = baseUrl.replace(/\/$/, '')
    this.token = token.trim()
    this.fetch = fetchImpl.bind(globalThis)
    this.storage = storage
    this.sessionId = storage?.getItem?.(SESSION_STORAGE_KEY) ?? null
  }

  async createSession() {
    let response = await this.#request('/api/session', { method: 'POST', json: true }, false)
    if (response.forbidden) {
      this.clearSession()
      response = await this.#request('/api/session', { method: 'POST', json: true }, false)
    }
    if (!response.ok) throw new Error(response.data.error || 'Failed to create session')

    this.sessionId = response.data.session_id
    this.storage?.setItem?.(SESSION_STORAGE_KEY, this.sessionId)
    return this.sessionId
  }

  async listMotions() {
    const response = await this.#request('/api/motions')
    return Array.isArray(response.motions) ? response.motions : []
  }

  getMotion(id) {
    return this.#request(`/api/motions/${encodeURIComponent(id)}`)
  }

  async generate(text, motionLength) {
    const response = await this.#request('/api/generate', {
      method: 'POST',
      json: true,
      body: {
        text,
        motion_length: motionLength,
        num_inference_steps: 10,
        adaptive_smooth: true,
        static_start: true,
        static_frames: 2,
        blend_frames: 8,
        transition_steps: 100,
      },
    })
    if (!response.success) throw new Error(response.error || 'Failed to generate motion')
    return { ...response.motion, motion_id: response.motion_id, text_prompt: text }
  }

  async deleteMotion(id) {
    return this.#request(`/api/motions/${encodeURIComponent(id)}`, { method: 'DELETE' })
  }

  async clearMotions() {
    return this.#request('/api/motions', { method: 'DELETE' })
  }

  clearSession() {
    this.sessionId = null
    this.storage?.removeItem?.(SESSION_STORAGE_KEY)
  }

  #headers(json) {
    const headers = {}
    if (this.token) headers.Authorization = `Bearer ${this.token}`
    if (json) headers['Content-Type'] = 'application/json'
    if (this.sessionId) headers['X-Session-ID'] = this.sessionId
    return headers
  }

  async #request(path, options = {}, throwForbidden = true) {
    const { json = false, body, ...init } = options
    const response = await this.fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: this.#headers(json),
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    })
    const data = await response.json().catch(() => ({}))
    const forbidden = response.status === 403 && data.code === 'SESSION_FORBIDDEN'
    if (forbidden && throwForbidden) {
      this.clearSession()
      throw new Error('Session expired')
    }
    if (throwForbidden && !response.ok) throw new Error(data.error || `Request failed: ${response.status}`)
    return throwForbidden ? data : { ok: response.ok, forbidden, data }
  }
}
