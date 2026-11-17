# PyBot-70 demo: PyBot, SmolLM2-360M and PyBot-70 side by side, with Ollama in one container.
# Works as a Hugging Face Docker Space (port 7860) or anywhere else that runs containers:
#
#   docker build -t pybot70 .
#   docker run -p 7860:7860 pybot70
#
# The model is baked into the image, so the container needs no network at run time.
FROM python:3.12-slim

ARG MODEL=smollm2:360m
ENV MODEL=${MODEL} \
    HYBRID_LLM_BACKEND=ollama \
    HYBRID_LLM_MODEL=${MODEL} \
    OLLAMA_MODELS=/opt/ollama-models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    HF_HOME=/opt/hf \
    HOME=/home/user \
    PORT=7860 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# Pull the model at build time.
RUN mkdir -p $OLLAMA_MODELS $HOME && (ollama serve > /tmp/ollama-build.log 2>&1 &) \
    && until curl -sf http://127.0.0.1:11434/api/version > /dev/null; do sleep 1; done \
    && ollama pull $MODEL && ollama pull all-minilm

# The PyBot-70 scorer was pickled with these versions; keep them pinned.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy==2.5.5 scikit-learn==1.9.1 sentence-transformers==6.3.0 \
    && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

WORKDIR /app
COPY bot ./bot
COPY hybrid ./hybrid
COPY demo ./demo
COPY bench/results/moneyshot.png ./bench/results/moneyshot.png
COPY pybot.py LICENSE ./

# Compile PyBot's 640k lines once, so the first start is not the slow one.
RUN python -m compileall -q bot hybrid && chmod -R a+rwX /app $OLLAMA_MODELS $HOME $HF_HOME

USER 1000
EXPOSE 7860
CMD ["sh", "demo/start.sh"]
