FROM python:3.12-slim

# Slim base image — smaller attack surface, faster pulls (Container Security)
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs media

# Run as non-root user — don't run the app as root inside the container
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]