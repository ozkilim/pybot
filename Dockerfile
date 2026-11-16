# PyBot Hybrid demo: PyBot (plain Python) + a small model served by Ollama, in one container.
# Works as a Hugging Face Docker Space (port 7860) or anywhere else that runs containers:
#
#   docker build -t pybot-hybrid .
#   docker run -p 7860:7860 pybot-hybrid
#
# The model is baked into the image, so the container needs no network at run time.
FROM python:3.12-slim

ARG MODEL=smollm2:1.7b
ENV MODEL=${MODEL} \
    HYBRID_LLM_BACKEND=ollama \
    HYBRID_LLM_MODEL=${MODEL} \
    OLLAMA_MODELS=/opt/ollama-models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    HOME=/home/user \
    PORT=7860 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# Pull the model at build time.
RUN mkdir -p $OLLAMA_MODELS $HOME && (ollama serve > /tmp/ollama-build.log 2>&1 &) \
    && until curl -sf http://127.0.0.1:11434/api/version > /dev/null; do sleep 1; done \
    && ollama pull $MODEL

WORKDIR /app
COPY bot ./bot
COPY hybrid ./hybrid
COPY demo ./demo
COPY bench/results ./bench/results
COPY pybot.py LICENSE ./

# Compile PyBot's 640k lines once, so the first start is not the slow one.
RUN python -m compileall -q bot hybrid && chmod -R a+rwX /app $OLLAMA_MODELS $HOME

USER 1000
EXPOSE 7860
CMD ["sh", "demo/start.sh"]
