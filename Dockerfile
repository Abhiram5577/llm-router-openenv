FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser requirements.txt .

# 🚀 THE FIX: Use 'uv' instead of 'pip' to prevent lockfile mismatches
RUN pip install --no-cache-dir uv
RUN uv pip compile requirements.txt -o uv.lock
RUN uv pip install --system -r requirements.txt

COPY --chown=appuser:appuser . .

RUN chown -R appuser:appuser /home/appuser/app
USER appuser

EXPOSE 7860

CMD ["python", "-m", "server.app"]