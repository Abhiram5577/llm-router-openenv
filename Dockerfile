# Use a lightweight, stable Python base image
FROM python:3.10-slim

# Prevent Python from buffering stdout and stderr (so prints show up instantly)
ENV PYTHONUNBUFFERED=1

# Set up a non-root user (Required for Hugging Face Spaces and good security practice)
RUN useradd -m -u 1000 appuser
USER appuser

# Set the working directory
WORKDIR /home/appuser/app

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies (This layer will be cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY --chown=appuser:appuser . .

# Set the default command to run our baseline evaluation script
# When judges run the container, it will train the agent and print the score.
CMD ["python", "baseline.py"]