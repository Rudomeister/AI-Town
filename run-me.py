import os
import json
import random

# Define symbols and intervals
symbols = [
    "LTCUSDT", "MANAUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "EOSUSDT", "BTCUSDT", "ETHUSDT", "ARBUSDT", "DOTUSDT", "1000PEPEUSDT", "CROUSDT", "BNBUSDT", "DOGEUSDT", "SUIUSDT", "1000BONKUSDT", "ICPUSDT", "MNTUSDT", "TONUSDT"
]
intervals = [5]

# Create config directory if it doesn't exist
os.makedirs("config", exist_ok=True)

# Generate configuration files for all symbols and intervals
for symbol in symbols:
    for interval in intervals:
        config = {
            "symbol": symbol,
            "interval": interval,
            "quantity": 100
        }
        config_filename = f"config/config-{symbol}-{interval}.json"
        with open(config_filename, "w") as config_file:
            json.dump(config, config_file, indent=4)

        print(f"Generated {config_filename} for {symbol} at {interval} interval.")


# Kombinere med generering av docker-compose.yml
#!/usr/bin/env python3
# generate_compose.py

compose_template = """
version: '3.8'
services:
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
    volumes:
      - ./config/config-{symbol}-{interval}.json:/app/config.json
    restart: unless-stopped
    network_mode: host
"""

# Write docker-compose.yml to file
with open('docker-compose.yml', 'w') as f:
    f.write(compose_template)

print("docker-compose.yml and all the config.json files generated successfully!")
