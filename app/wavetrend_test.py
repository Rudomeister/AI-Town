import pandas as pd
import dateparser
import time
import os
import logging
import json
from datetime import datetime, timezone
import numpy as np
from matplotlib import pyplot as plt
import requests
from matplotlib.dates import DateFormatter  # Added for date formatting
import paho.mqtt.client as mqtt
import hashlib
import hmac
import time
from decimal import Decimal, getcontext

from dotenv import load_dotenv

# Setup logging

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for GUI updates
latest_signal_text = ""
position_status_text = ""
# Setup Bybit API session



    
    
def send_mqtt_signal(signals):
    signal_data = {
        "cross_up": signals['cross_up'],
        "cross_down": signals['cross_down'],
        "strong_cross_up": signals['strong_cross_up'],
        "strong_cross_down": signals['strong_cross_down'],
        "timestamp": datetime.now().isoformat()
    }
    try:
        client = mqtt.Client(callback_api_version=2, protocol=mqtt.MQTTv5)
        client.connect("localhost", 1883, 60)

        message = None  # Vi starter uten en melding
        if signal_data['cross_up']:
            message = {
                "signal": f"Buy Signal for symbol {symbol} at the {interval} minute interval.",
                "timestamp": signal_data["timestamp"]
            }
        elif signal_data['cross_down']:
            message = {
                "signal": f"Sell Signal for symbol {symbol} at the {interval} minute interval.",
                "timestamp": signal_data["timestamp"]
            }
        elif signal_data['strong_cross_up']:
            message = {
                "signal": f"Strong Buy Signal for symbol {symbol} at the {interval} minute interval.",
                "timestamp": signal_data["timestamp"]
            }
        elif signal_data['strong_cross_down']:
            message = {
                "signal": f"Strong Sell Signal for symbol {symbol} at the {interval} minute interval.",
                "timestamp": signal_data["timestamp"]
            }

        # Hvis det ikke er noen signaler, logg informasjon og ikke send melding
        if not message:
            logging.info(f"No activity for symbol {symbol} at the {interval} minute interval.")
            return

        # Lagre meldingen til en fil for historikk
        with open('latest_signals.json', 'w') as f:
            json.dump(message, f, indent=4)

        # Publiser til 'wavetrend_signals' topic
        client.publish("wavetrend_signals", json.dumps(message), qos=0)
        logging.info(f"MQTT signal sent: {message}")

        client.loop_start()
        time.sleep(1)  # Vent litt for å sikre at meldingen sendes
        client.loop_stop()
        client.disconnect()
    except mqtt.MQTTException as mqtt_e:
        logging.error(f"MQTT error occurred: {mqtt_e}")
    except Exception as e:
        logging.error(f"Error sending MQTT signal: {e}")

        
        # Function to get historical data
def get_historical_data(symbol, interval, start_time=None, end_time=None):
    url = f"https://api-demo.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit=400"
    
    if start_time and end_time:
        # Convert directly to timestamp if datetime objects
        if isinstance(start_time, (datetime, pd.Timestamp)):
            start_timestamp = int(start_time.timestamp() * 1000)
        else:
            start_timestamp = int(dateparser.parse(start_time).timestamp() * 1000)
                
        if isinstance(end_time, (datetime, pd.Timestamp)):
            end_timestamp = int(end_time.timestamp() * 1000)
        else:
            end_timestamp = int(dateparser.parse(end_time).timestamp() * 1000)
                
        url += f"&start={start_timestamp}&end={end_timestamp}"
    
    response = requests.get(url)
    data = response.json()

    
    df = pd.DataFrame(columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume'])
    
    if 'result' in data and data['result'] is not None:
        rows = []
        for bar in data['result']['list']:
            rows.append({
                'startTime': pd.to_datetime(int(bar[0]), unit='ms'),
                'openPrice': float(bar[1]),
                'highPrice': float(bar[2]),
                'lowPrice': float(bar[3]),
                'closePrice': float(bar[4]),
                'volume': float(bar[5])
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(by='startTime')
        
        # Verify interval spacing
        time_diff = df['startTime'].diff().median()
        logging.info(f"Bar interval: {time_diff.total_seconds() / 60} minutes")
        
        df.rename(columns={
            'startTime': 'Date',
            'openPrice': 'Open',
            'highPrice': 'High',
            'lowPrice': 'Low',
            'closePrice': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        df.to_json(f'./historical_data.json', orient='records', date_format='iso')    
    return df

def calculate_stoploss(df, n=20, multiplier=2):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    atr = typical_price.rolling(n).mean()
    stoploss = typical_price - multiplier * atr
    return stoploss
def calculate_quantity(position_size_usd, price):
    quantity = position_size_usd / price
    # Round down to the acceptable precision for the asset
    return quantity

def calculate_takeprofit(df, n=20, multiplier=2):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    atr = typical_price.rolling(n).mean()
    takeprofit = typical_price + multiplier * atr
    return takeprofit



def make_private_get_request(endpoint, params):
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    api_key = os.environ.get('BYBIT_API_KEY_DEMO')
    api_secret = os.environ.get('BYBIT_SECRET_KEY_DEMO')

    query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    signature_payload = timestamp + api_key + recv_window + query_string
    signature = hmac.new(
        bytes(api_secret, 'utf-8'),
        signature_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    headers = {
        'X-BAPI-API-KEY': api_key,
        'X-BAPI-TIMESTAMP': timestamp,
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': recv_window,
    }
    url = 'https://api-demo.bybit.com' + endpoint
    response = requests.get(url, params=params, headers=headers)
    return response.json()

def make_private_post_request(endpoint, params):
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    api_key = os.environ.get('BYBIT_API_KEY_DEMO')
    api_secret = os.environ.get('BYBIT_SECRET_KEY_DEMO')

    body = json.dumps(params)
    signature_payload = timestamp + api_key + recv_window + body
    signature = hmac.new(
        bytes(api_secret, 'utf-8'),
        signature_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    headers = {
        'X-BAPI-API-KEY': api_key,
        'X-BAPI-TIMESTAMP': timestamp,
        'X-BAPI-SIGN': signature,
        'X-BAPI-RECV-WINDOW': recv_window,
        'Content-Type': 'application/json'
    }
    url = 'https://api-demo.bybit.com' + endpoint
    response = requests.post(url, data=body, headers=headers)
    return response.json()



def get_wallet_balance():
    try:
        endpoint = '/v5/account/wallet-balance'
        params = {'accountType': 'UNIFIED', 'coin': 'USDT'}
        response = make_private_get_request(endpoint, params)
        if response['retCode'] == 0:
            balance = response['result']['list'][0]['coin'][0]['walletBalance']
            logging.info(f"Account balance: {balance} USDT")
            return balance
        else:
            logging.error(f"Error getting account overview: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"Error getting account overview: {e}")
        return None
def get_positions(symbol):
    try:
        endpoint = '/v5/position/list'
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        response = make_private_get_request(endpoint, params)
        print(json.dumps(response, indent=4))  # Corrected print statement
        if response['retCode'] == 0:
            positions = response['result']['list']
            logging.info(f"Current positions for {symbol}: {positions}")
            return positions
        else:
            logging.error(f"Error getting positions: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"Error getting positions: {e}", exc_info=True)
        return None


def place_order(symbol, side, leverage, stoploss, takeprofit, qty, price=None, order_type="Market", time_in_force="GTC"):
    try:
        endpoint = '/v5/order/create'
        params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'qty': str(qty),
            'timeInForce': time_in_force,
            'stopLoss': stoploss,
            'takeProfit': takeprofit,
            'positionIdx': 0  # 0 for One-Way Mode, adjust if using Hedge Mode
        }
        # if price:
        #     params['price'] = str(price)
        # if stoploss:
        #     params['stopLoss'] = str(stoploss)
        # if takeprofit:
        #     params['takeProfit'] = str(takeprofit)
        #if leverage:
        #    params['leverage'] = str(leverage)
        response = make_private_post_request(endpoint, params)
        if response['retCode'] == 0:
            logging.info(f"Order placed: {response}")
            return response
        else:
            logging.error(f"Error placing order: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"Error placing order: {e}", exc_info=True)
        return None


def get_symbol_info(symbol):
    url = f"https://api.bybit.com/v5/market/instruments-info?category=linear&symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    if data['retCode'] == 0 and data['result']['list']:
        info = data['result']['list'][0]
        return info
    else:
        logging.error(f"Error fetching symbol info: {data['retMsg']}")
        return None

def get_current_price(symbol):
    url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={symbol}"
    response = requests.get(url)
    data = response.json()
    if data['retCode'] == 0 and data['result']['list']:
        price = float(data['result']['list'][0]['lastPrice'])
        return price
    else:
        logging.error(f"Error fetching current price: {data['retMsg']}")
        return None


def wavetrend(df, n1=10, n2=21):
    hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
    esa = hlc3.ewm(span=n1, adjust=False).mean()
    d = abs(hlc3 - esa).ewm(span=n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=4).mean()
    return pd.DataFrame({'wt1': wt1, 'wt2': wt2, 'wtDiff': wt1 - wt2}, index=df.index)

# Function to generate WaveTrend signals
def get_wavetrend_signals(df):
    wt = wavetrend(df)
    ob_level1, ob_level2 = 53, 60
    os_level1, os_level2 = -53, -60

    signals = pd.DataFrame({
        'overbought': wt.wt1 > ob_level1,
        'oversold': wt.wt1 < os_level1,
        'very_overbought': wt.wt1 > ob_level2,
        'very_oversold': wt.wt1 < os_level2,
        # Buy signal when crossing above wt2 and exiting oversold zone
        'cross_up': ((wt.wt1 > wt.wt2) & (wt.wt1.shift(1) <= wt.wt2.shift(1)) & (wt.wt1.shift(1) < os_level1)),
        # Sell signal when crossing below wt2 and exiting overbought zone
        'cross_down': ((wt.wt1 < wt.wt2) & (wt.wt1.shift(1) >= wt.wt2.shift(1)) & (wt.wt1.shift(1) > ob_level1)),
        # Strong Buy signal when crossing above wt2 and exiting very oversold zone
        'strong_cross_up': ((wt.wt1 > wt.wt2) & (wt.wt1.shift(1) <= wt.wt2.shift(1)) & (wt.wt1.shift(1) < os_level2)),
        # Strong Sell signal when crossing below wt2 and exiting very overbought zone
        'strong_cross_down': ((wt.wt1 < wt.wt2) & (wt.wt1.shift(1) >= wt.wt2.shift(1)) & (wt.wt1.shift(1) > ob_level2))
    }, index=df.index)

    return wt, signals, ob_level1, ob_level2, os_level1, os_level2

def monitor_wavetrend():
    while True:
        current_time = datetime.now(timezone.utc)
        logging.info(f"Current time: {current_time}")

        # Calculate time to wait until the next update
        if interval == "D":
            next_update = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
            wait_seconds = (next_update - current_time).total_seconds()
        else:
            minutes_to_next = int(interval) - (current_time.minute % int(interval))
            wait_seconds = minutes_to_next * 60 - current_time.second
            if wait_seconds < 0:
                wait_seconds += int(interval) * 60

        logging.info(f"Waiting {wait_seconds / 60:.2f} minutes until next update")
        time.sleep(wait_seconds)

        # Update current time after sleeping
        current_time = datetime.now(timezone.utc)

        # Fetch historical data
        end_time = current_time
        start_time = end_time - pd.Timedelta(days=30) if interval == "D" else end_time - pd.Timedelta(hours=24)

        data = get_historical_data(symbol=symbol, interval=interval, start_time=start_time, end_time=end_time)
        if data.empty:
            logging.error("Historical data is empty. Skipping this iteration.")
            continue  # Skip to the next iteration

        data.set_index('Date', inplace=True)

        # Calculate indicators and signals
        wave_trend_data, signals, ob_level1, ob_level2, os_level1, os_level2 = get_wavetrend_signals(data)

        # Calculate stoploss and takeprofit prices
        stoploss = calculate_stoploss(data).iloc[-1]
        takeprofit = calculate_takeprofit(data).iloc[-1]

        # Fetch current price
        current_price = get_current_price(symbol)
        if current_price is None:
            logging.error("Could not fetch current price. Skipping this iteration.")
            continue

        # Define position size in USD
        position_size_usd = 100  # Example: $100 per trade

        # Calculate quantity based on USD position size
        quantity = calculate_quantity(position_size_usd, current_price)
        logging.info(f"Calculated quantity: {quantity} {symbol}")

        # Ensure quantity meets the exchange's requirements
        symbol_info = get_symbol_info(symbol)
        if symbol_info:
            lot_size_filter = symbol_info.get('lotSizeFilter', {})
            min_trading_qty = float(lot_size_filter.get('minOrderQty', 0))
            qty_step = float(lot_size_filter.get('qtyStep', 0))
            if min_trading_qty == 0 or qty_step == 0:
                logging.error("Could not retrieve 'minOrderQty' or 'qtyStep' from symbol info.")
            else:
                quantity = max(quantity, min_trading_qty)
                quantity = (quantity // qty_step) * qty_step
                logging.info(f"Adjusted quantity: {quantity}")
        else:
            logging.error("Could not fetch symbol info. Skipping quantity adjustment.")

        # Define leverage
        leverage = 10  # Example leverage value

        # Log current WaveTrend values
        logging.info(f"Current WT1 value: {wave_trend_data.wt1.iloc[-1]:.2f}")
        logging.info(f"Current WT2 value: {wave_trend_data.wt2.iloc[-1]:.2f}")
        logging.info(f"Current WTDiff: {wave_trend_data.wtDiff.iloc[-1]:.2f}")

        latest = signals.iloc[-1]

        # Check for signals and place orders
        if latest['cross_up']:
            logging.info(f"🔊 WaveTrend Buy Signal for {symbol} at {interval}!")
            signal_text = f"Buy Signal for {symbol} at {interval} interval."
            place_order(symbol=symbol, side="Buy", leverage=leverage, stoploss=stoploss, takeprofit=takeprofit, qty=quantity)
        elif latest['cross_down']:
            logging.info(f"🔊 WaveTrend Sell Signal for {symbol} at {interval}!")
            signal_text = f"Sell Signal for {symbol} at {interval} interval."
            place_order(symbol=symbol, side="Sell", leverage=leverage, stoploss=stoploss, takeprofit=takeprofit, qty=quantity)
        elif latest['strong_cross_up']:
            logging.info(f"🔊 WaveTrend Strong Buy Signal for {symbol} at {interval}!")
            signal_text = f"Strong Buy Signal for {symbol} at {interval} interval."
            place_order(symbol=symbol, side="Buy", leverage=leverage, stoploss=stoploss, takeprofit=takeprofit, qty=quantity)
        elif latest['strong_cross_down']:
            logging.info(f"🔊 WaveTrend Strong Sell Signal for {symbol} at {interval}!")
            signal_text = f"Strong Sell Signal for {symbol} at {interval} interval."
            place_order(symbol=symbol, side="Sell", leverage=leverage, stoploss=stoploss, takeprofit=takeprofit, qty=quantity)
        else:
            logging.info(f"No new signals for {symbol} at {interval}.")

        # Send the latest signals via MQTT
        send_mqtt_signal(latest)

        






if __name__ == "__main__":
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
    symbol = config['symbol']
    interval = config['interval']  # Keep as string
#    quantity = float(config['quantity'])

    # Get positions and wallet balance before starting
    positions = get_positions(symbol)
    balance = get_wallet_balance()
    print(f"Positions: {positions}")
    print(f"Wallet Balance: {balance}")

    monitor_wavetrend()


