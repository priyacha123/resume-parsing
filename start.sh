#!/bin/bash
set -e

echo "Starting Gunicorn..."
exec gunicorn config.wsgi:application --bind 0.0.0.0:$PORT --workers 2



# #!/bin/bash
# set -e

# echo "Starting Celery worker in background..."
# celery -A config worker -l info &

# echo "Starting Gunicorn..."
# exec gunicorn config.wsgi:application --bind 0.0.0.0:$PORT --workers 1 --timeout 120