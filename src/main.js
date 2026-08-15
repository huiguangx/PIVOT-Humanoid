import '@mdi/font/css/materialdesignicons.css'
import 'roboto-fontface/css/roboto/roboto-fontface.css'
import 'vuetify/styles'
import './styles.css'

import { createApp } from 'vue'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import App from './App.vue'

createApp(App).use(createVuetify({ components, directives })).mount('#app')
