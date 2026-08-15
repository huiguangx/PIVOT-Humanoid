# PIVOT

**Retargeting-Free Language-to-Motion for Closed-Loop Humanoid Control**

[English](#english) | [中文](#中文)

## English

PIVOT turns natural-language motion descriptions into robot-native motion references and executes them through a closed-loop humanoid tracking policy. The standalone web application combines MuJoCo WebAssembly, ONNX Runtime Web, Three.js, and Vue.

### Highlights

- Robot-native language-to-motion without a human-to-robot retargeting stage
- Compact motion references for online whole-body tracking
- Closed-loop 29-joint humanoid control with future-reference conditioning
- Interactive MuJoCo simulation with physical perturbations
- Browser-local ONNX policy execution

### Run locally

```bash
npm install
cp .env.example .env.local
npm run dev
```

The simulator works without a text-to-motion service. To enable prompt generation, set `VITE_TEXT_MOTION_URL` in `.env.local`. Keep private tokens out of version control.

### Test and build

```bash
npm test
npm run build
```

### Repository

<https://github.com/huiguangx/PIVOT-Humanoid>

### License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## 中文

PIVOT 将自然语言动作描述转换为机器人原生动作参考，并通过闭环人形机器人跟踪策略完成物理执行。该独立 Web 应用基于 MuJoCo WebAssembly、ONNX Runtime Web、Three.js 和 Vue 构建。

### 项目亮点

- 直接生成机器人原生动作，无需人体到机器人的动作重定向
- 使用紧凑动作参考实现在线全身动作跟踪
- 基于未来参考条件的 29 关节闭环人形机器人控制
- 支持物理扰动交互的 MuJoCo 仿真环境
- 在浏览器本地执行 ONNX 控制策略

### 本地运行

```bash
npm install
cp .env.example .env.local
npm run dev
```

即使没有文本动作生成服务，仿真器也可以独立运行。如需启用提示词动作生成，请在 `.env.local` 中配置 `VITE_TEXT_MOTION_URL`。请勿将私密令牌提交到版本控制系统。

### 测试与构建

```bash
npm test
npm run build
```

### 项目仓库

<https://github.com/huiguangx/PIVOT-Humanoid>

### 开源协议

本项目采用 MIT 协议，详情参见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。
