# 使用官方 Python 3.9 镜像作为基础
FROM python:3.9

# 设置工作目录
WORKDIR /code

# 为了避免权限问题，创建一个非 root 用户 (Hugging Face 的安全要求)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 复制依赖文件并安装
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 复制当前目录下的所有文件到容器中
COPY --chown=user . .

# 暴露 7860 端口 (Hugging Face Spaces 的默认端口)
EXPOSE 7860

# 启动命令
CMD ["python", "app.py"]