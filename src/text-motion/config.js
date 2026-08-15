const env = import.meta.env ?? {}

export const TEXT_MOTION_URL = env.VITE_TEXT_MOTION_URL
  ?? ''

export const TEXT_MOTION_TOKEN = env.VITE_TEXT_MOTION_TOKEN
  ?? ''

export const SESSION_STORAGE_KEY = 'pivot_motion_session_id'
