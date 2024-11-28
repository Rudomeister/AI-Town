FROM python:3.13.0-alpine3.19

WORKDIR /app
# Kopier requirements.txt til Docker-image
COPY ./app/requirements.txt /app/

# Installer dependencies fra requirements.txt med pip
RUN apk add --no-cache gcc musl-dev
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mysql-connector-python

# Kopier resten av prosjektet til Docker-image
COPY ./app /app/

# Angi kommandoen som skal kjøre applikasjonen
CMD ["python", "wavetrend_alerts.py"]
