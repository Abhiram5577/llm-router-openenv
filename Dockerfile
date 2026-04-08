FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

COPY --chown=appuser:appuser requirements.txt .
RUN touch uv.lock

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .

# Set permissions
RUN chown -R appuser:appuser /home/appuser/app
USER appuser

EXPOSE 7860

# This tells the system to run the app.py inside the server folder
CMD ["python", "server/app.py"]