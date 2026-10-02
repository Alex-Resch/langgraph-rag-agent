FROM python:3.13-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN chown user:user /home/user/app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

RUN pip install --no-cache-dir uv

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Download the embedding model at build time instead of on every container start.
COPY --chown=user config.py ./
RUN uv run --no-dev python -c "from sentence_transformers import SentenceTransformer; from config import EMBEDDING_MODEL; SentenceTransformer(EMBEDDING_MODEL)"

COPY --chown=user . .

EXPOSE 7860

CMD ["uv", "run", "--no-dev", "chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "7860", "--headless"]
