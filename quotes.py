import time
import psycopg2
import logging
import os
import tinkoff.invest 
from dotenv import load_dotenv

from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
from urllib.parse import urlparse
from datetime import datetime, timedelta
#from config_reader import config
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)

FIGI_FILE = 'figi.txt'

def load_figies():
    try:
        with open(FIGI_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines] # очищаем строки
        return lines
    except FileNotFoundError:
        print(f"Файл '{filepath}' не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def get_historical_candles(client, figi, days_back = 365): 
    now = datetime.utcnow()
    from_ = now - timedelta(days=days_back)

    try:
        candles = client.market_data.get_candles(
            figi=figi,
            from_=from_,
            to=now,
            interval=CandleInterval.CANDLE_INTERVAL_DAY 
        ).candles
        return candles
    except Exception as e:
        print(f'Error getting historical candles for {figi}: {e}')
        return []


def save_candles_to_db(candles, figi):
    try:
        database_dsn = os.getenv('DATABASE_URL')
        connection = psycopg2.connect(database_dsn)
        cursor = connection.cursor()
        for candle in candles:
            insert_query = """
                INSERT INTO quotes (figi, open_price, close_price, high_price, low_price, volume, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (figi, timestamp) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                close_price = EXCLUDED.close_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                volume = EXCLUDED.volume;
            """
            cursor.execute(insert_query, (
                figi,
                candle.open.units + candle.open.nano / 1e9,
                candle.close.units + candle.close.nano / 1e9,
                candle.high.units + candle.high.nano / 1e9,
                candle.low.units + candle.low.nano / 1e9,
                candle.volume,
                candle.time,
            ))
        connection.commit()
        print(f"Saved {len(candles)} candles to DB.")
    except Exception as e:
        print(f'Error saving candles to DB: {e}')
    finally:
        if connection:
            cursor.close()
            connection.close()


def main():
    load_dotenv()
    api_token = os.getenv('TINKOFF_API_TOKEN')

    with Client(api_token) as client:
        figies = load_figies()
        for figi in figies:
            print(figi)
            candles = get_historical_candles(client, figi)
            print(candles)
            save_candles_to_db(candles, figi)



if __name__ == '__main__':
    main()
