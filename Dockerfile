# GraphSentinel Container Image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and install python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir scikit-learn scipy lightgbm \
    && pip install --no-cache-dir -e .[dev]

# Copy project source code
COPY . /app/

# Expose interactive Analyst Console port
EXPOSE 4173

# Default entrypoint launches the demo console
CMD ["python", "scripts/run_demo.py"]
