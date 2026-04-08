# Z.ai OpenAI 兼容代理

本项目是一个将 [chat.z.ai](https://chat.z.ai) 逆向封装为 OpenAI 兼容 API 的代理服务。默认使用的模型为 `glm-5`。它支持标准对话、流式输出 (SSE) 以及 Toolify 风格的函数调用能力。

## 功能特性

- 提供兼容 OpenAI 格式的 `/v1/chat/completions` 和 `/v1/models` 接口。
- 自动管理匿名（Guest）会话认证，开箱即用。
- 充分支持各类兼容 OpenAI 规范的第三方客户端调用。

## Docker 部署步骤

项目根目录下已准备好 `docker-compose.yml` 文件，便于通过 Docker 一键部署服务。

### 1. 克隆仓库与准备

首先，将本项目克隆到本地并进入项目目录：

```bash
git clone https://github.com/xchxh/glm.git
cd glm
```


### 2. 启动服务

进入项目根目录，也就是 `docker-compose.yml` 所在的文件夹，执行如下命令即可在后台启动服务：

```bash
docker-compose up -d
```
> **提示**：该命令会自动拉取 `python:3.11-slim` 镜像，并安装所需依赖后启动服务。

### 3. 查看服务日志

如果你想要查看代理服务产生的日志或排查问题，可以运行：

```bash
docker-compose logs -f zai-openai
```

### 4. 停止服务

当你需要停止并移除容器时，可以运行：

```bash
docker-compose down
```

## Render 部署步骤

您可以轻松地将本项目部署到 [Render](https://render.com/) 平台。

### 1. 使用 Blueprint 一键部署

1. 将本项目 Fork 到您的 GitHub 账号。
2. 登录 Render 控制台，点击 **"New"** -> **"Blueprint"**。
3. 连接您的仓库，Render 会自动识别 `render.yaml` 文件。
4. 点击 **"Apply"** 即可开始部署。

### 2. 手动创建 Web Service

1. 点击 **"New"** -> **"Web Service"**。
2. 选择您的仓库。
3. Runtime 选择 **"Docker"**。
4. 在 **"Environment"** 中，可以根据需要设置 `LOG_LEVEL` (默认 `INFO`)。
5. Render 会自动识别并暴露端口。

部署完成后，您将获得一个类似 `https://zai-openai-proxy.onrender.com` 的 URL。

## API 调用说明

服务启动后，默认暴漏于宿主机的 **30016** 端口。

- **Base URL**: `http://localhost:30016/v1`
- **可用端点**: `/v1/chat/completions`, `/v1/models`

在任何兼容 OpenAI 的客户端软件（如 NextChat, Chatbox 等）中设置时：
- 将自定义 API 地址（Base URL）设置为：`http://你的服务器IP或localhost:30016/v1`
- **API Key**：可随意填写任意字符串（系统会自动调用匿名用户 Token 完成底层验证）。
- **模型名称 (Model)**：输入 `glm-5` 即可。
