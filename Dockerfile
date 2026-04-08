FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口 (Render 会自动覆盖此端口，但显式声明是好习惯)
EXPOSE 30016

# 启动命令
CMD ["python", "openai.py"]
