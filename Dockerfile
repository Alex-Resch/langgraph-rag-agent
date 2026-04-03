FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./pyproject.toml pyproject.toml
COPY --chown=user ./uv.lock uv.lock

RUN pip install uv
RUN uv sync --frozen

COPY --chown=user . /app

EXPOSE 7860

CMD ["uv", "run", "chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "7860"]