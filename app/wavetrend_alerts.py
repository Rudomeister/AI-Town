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


        message = {}
        if signal_data['cross_up']:
            message["signal"] = f"Buy Signal for symbol {symbol} at the {interval} minute interval."
            message["signal"] = datetime.now().isoformat()
        elif signal_data['cross_down']:
            message["signal"] = f"Sell Signal for symbol {symbol} at the {interval} minute interval."
            message["signal"] = datetime.now().isoformat()
        elif signal_data['strong_cross_up']:
            message["signal"] = f"Strong Buy Signal for symbol {symbol} at the {interval} minute interval."
            message["signal"] = datetime.now().isoformat()
        elif signal_data['strong_cross_down']:
            message["signal"] = f"Strong Sell Signal for symbol {symbol} at the {interval} minute interval."
            message["signal"] = datetime.now().isoformat()


            #message["signal"] = f"No significant activity for symbol {symbol} at the {interval} minute interval."
        # Save the message to a file
        with open('latest_signals.json', 'w') as f:
            json.dump(message, f, indent=4)
#        message["There are no activity this round. The time is: "] = datetime.now().isoformat()
        


        # Publish to the topic 'wavetrend_signals' with QoS level 0 (can be adjusted)
        if message:
            client.publish("wavetrend_signals", json.dumps(message), qos=0)
            logging.info(f"MQTT signal sent: {message}")
        else:
            logging.info(f"No activity for symbol {symbol} at the {interval} minute interval.")
        client.loop_start()
        time.sleep(1)  # Wait for the message to be sent
        client.loop_stop()
        client.disconnect()

        logging.info(f"MQTT signal sent: {message}")
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

def calculate_takeprofit(df, n=20, multiplier=2):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    atr = typical_price.rolling(n).mean()
    takeprofit = typical_price + multiplier * atr
    return takeprofit


def calculate_position_size(df, balance, entry_price):
    leverage = calculate_leverage(df, balance, entry_price)
    position_size = quantity * leverage
    return position_size


def calculate_position_size(df, balance, entry_price):
    leverage = calculate_leverage(df, balance, entry_price)
    position_size = balance * leverage
    return position_size


def calculate_leverage(df, balance, entry_price):
    position_size = calculate_position_size(df, balance, entry_price)
    leverage = position_size / balance
    return leverage


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

# Doesen't work on demo account. Making a dummy replacement for now.
# def get_account_overview():
#     try:
#         endpoint = '/v5/account/wallet-balance'
#         params = {'accountType': 'CONTRACT', 'coin': 'USDT'}
#         response = make_private_get_request(endpoint, params)
#         if response['retCode'] == 0:
#             balance = response['result']['list'][0]['coin'][0]['walletBalance']
#             logging.info(f"Account balance: {balance} USDT")
#             return balance
#         else:
#             logging.error(f"Error getting account overview: {response['retMsg']}")
#             return None
#     except Exception as e:
#         logging.error(f"Error getting account overview: {e}")
#         return None

def calculate_quantity(balance=100000, risk_percentage=0.01):
    # Beregner kvantitet basert på en prosentandel av balansen
    return balance * risk_percentage
# Function to get current positions

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
        print(response.text)
        if response['retCode'] == 0:
            positions = response['result']['list']
            logging.info(f"Current positions for {symbol}: {positions}")
            return positions
        else:
            logging.error(f"Error getting positions: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"Error getting positions: {e}")
        return None

# Function to place an order
def place_order(symbol, side, leverage, stoploss, takeprofit, qty, price=None, order_type="Market", time_in_force="GTC"):
    try:
        endpoint = '/v5/order/create'
        params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'isLeverage': str(leverage),
            'qty': str(qty),
            'timeInForce': time_in_force,
            'stopLoss': str(stoploss),
            'takeProfit': str(takeprofit),
        }
        if price:
            params['price'] = str(price)
        response = make_private_post_request(endpoint, params)
        if response['retCode'] == 0:
            logging.info(f"Order placed: {response}")
            return response
        else:
            logging.error(f"Error placing order: {response['retMsg']}")
            return None
    except Exception as e:
        logging.error(f"Error placing order: {e}")
        return None
# Function to calculate WaveTrend indicator




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
        print(f"Current time: {current_time}")

        # Handle different interval types
        if interval == "D":
            next_update = current_time.replace(hour=0, minute=0, second=0) + pd.Timedelta(days=1)
            wait_seconds = (next_update - current_time).total_seconds()
        else:
            minutes_to_next = int(interval) - (current_time.minute % int(interval))
            wait_seconds = minutes_to_next * 60 - current_time.second

        logging.info(f"Waiting {wait_seconds / 60:.2f} minutes until next update")
        time.sleep(wait_seconds)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - pd.Timedelta(days=30) if interval == "D" else end_time - pd.Timedelta(hours=24)

        data = get_historical_data(symbol=symbol, interval=interval, start_time=start_time, end_time=end_time)
        data.set_index('Date', inplace=True)  # Ensure datetime index is set
        wave_trend_data, signals, ob_level1, ob_level2, os_level1, os_level2 = get_wavetrend_signals(data)

        logging.info(f"Current WT1 value: {wave_trend_data.wt1.iloc[-1]:.2f}")
        logging.info(f"Current WT2 value: {wave_trend_data.wt2.iloc[-1]:.2f}")
        logging.info(f"Current WTDiff: {wave_trend_data.wtDiff.iloc[-1]:.2f}")

        latest = signals.iloc[-1]
        close_price = data['Close'].iloc[-1]
        plt.figure(figsize=(12, 6))
        plt.title(f"WaveTrend for {symbol} at {interval}")
        plt.xlabel('Time')
        plt.ylabel('Value')


        plot_range = wave_trend_data.index[-100:]

    # Plot WaveTrend data with datetime index
        plt.plot(plot_range, wave_trend_data.wt1[-100:], label='WT1')
        plt.plot(plot_range, wave_trend_data.wt2[-100:], label='WT2')
        plt.plot(plot_range, wave_trend_data.wtDiff[-100:], label='WT1 - WT2')

    # Plot overbought and oversold levels
        plt.hlines(y=[ob_level1, ob_level2], xmin=plot_range[0], xmax=plot_range[-1],
               colors='red', linestyles='dashed', label='Overbought Levels')
        plt.hlines(y=[os_level1, os_level2], xmin=plot_range[0], xmax=plot_range[-1],
               colors='green', linestyles='dashed', label='Oversold Levels')

        plt.legend()
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))  # Format x-axis dates
        plt.gcf().autofmt_xdate()  # Auto-format date labels
        plt.savefig(f'latest_wavetrend-{symbol}-{interval}.png')
        plt.close()

        # signal_text = "No new signals."
        if latest['cross_up']:
            logging.info(f"🔊 WaveTrend Buy Signal for {symbol} at {interval}!")
            signal_text = f"Buy Signal for {symbol} at {interval} interval."
            # Place Buy Order
            place_order(symbol=symbol, side="Buy", qty=quantity)
        elif latest['cross_down']:
            logging.info(f"🔊 WaveTrend Sell Signal for {symbol} at {interval}!")
            signal_text = f"Sell Signal for {symbol} at {interval} interval."
            # Place Sell Order
            place_order(symbol=symbol, side="Sell", qty=quantity)
        elif latest['strong_cross_up']:
            logging.info(f"🔊 WaveTrend Strong Buy Signal for {symbol} at {interval}!")
            signal_text = f"Strong Buy Signal for {symbol} at {interval} interval."
            # Place Buy Order
            place_order(symbol=symbol, side="Buy", qty=quantity)
        elif latest['strong_cross_down']:
            logging.info(f"🔊 WaveTrend Strong Sell Signal for {symbol} at {interval}!")
            # signal_text = f"Strong Sell Signal for {symbol} at {interval} interval."
            # Place Sell Order
            place_order(symbol=symbol, side="Sell", qty=quantity)
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
    interval = int(config['interval'])
    quantity = float(config['quantity'])
    
    monitor_wavetrend()
    get_positions()
    get_wallet_balance()

