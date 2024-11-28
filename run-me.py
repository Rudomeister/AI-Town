import os
import json
import random

# Define symbols and intervals
symbols = [
    "LTCUSDT", "MANAUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "EOSUSDT", "BTCUSDT", "ETHUSDT", "ARBUSDT", "DOTUSDT", "1000PEPEUSDT", "CROUSDT", "BNBUSDT", "DOGEUSDT", "SUIUSDT", "1000BONKUSDT", "ICPUSDT", "MNTUSDT", "TONUSDT"
]
intervals = [15]

# Create config directory if it doesn't exist
os.makedirs("config", exist_ok=True)

# Generate configuration files for all symbols and intervals
for symbol in symbols:
    for interval in intervals:
        config = {
            "symbol": symbol,
            "interval": interval
        }
        config_filename = f"config/config-{symbol}-{interval}.json"
        with open(config_filename, "w") as config_file:
            json.dump(config, config_file, indent=4)

        print(f"Generated {config_filename} for {symbol} at {interval} interval.")

# Kombinere med generering av docker-compose.yml
compose_template = """
version: '3.8'
services:
  mysql:
    image: mysql:8.0
    container_name: mysql_wavetrend
    environment:
      MYSQL_ROOT_PASSWORD: flower
      MYSQL_DATABASE: wavetrend
      MYSQL_USER: l-ted
      MYSQL_PASSWORD: p-ted
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    restart: unless-stopped

  phpmyadmin:
    image: phpmyadmin/phpmyadmin
    container_name: phpmyadmin
    restart: unless-stopped
    ports:
      - "8080:80"
    environment:
      PMA_HOST: mysql
      MYSQL_ROOT_PASSWORD: flower
"""

for symbol in symbols:
    for interval in intervals:
        service_name = f"{symbol.lower()}_{interval}"
        # Generate a random delay between 0 and 5 seconds
        random_delay = random.randint(0, 5)
        compose_template += f"""
  {service_name}:
    build: .
    image: wave-monitor-{symbol.lower()}
    container_name: wave-monitor-{symbol.lower()}-{interval}
    environment:
      - BYBIT_API_KEY_DEMO=${{BYBIT_API_KEY_DEMO}}
      - BYBIT_SECRET_KEY_DEMO=${{BYBIT_SECRET_KEY_DEMO}}
      - SYMBOL={symbol}
      - INTERVAL={interval}
      - START_DELAY={random_delay}
      - DB_HOST=mysql
      - DB_PORT=3306
      - DB_USER=l-ted
      - DB_PASSWORD=p-ted
      - DB_NAME=wavetrend
    volumes:
      - ./config/config-{symbol}-{interval}.json:/app/config.json
    restart: unless-stopped
    network_mode: host
"""

# Add volumes section for MySQL data persistence
compose_template += """
volumes:
  mysql_data:
"""

# Write docker-compose.yml to file
with open('docker-compose.yml', 'w') as f:
    f.write(compose_template)

print("docker-compose.yml and all the config.json files generated successfully!")
