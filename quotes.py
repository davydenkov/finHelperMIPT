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
import asyncio

logging.basicConfig(level=logging.INFO)

FIGI_FILE = 'figi.txt'

async def load_figies():
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

async def get_historical_candles(client, figi, days_back = 105):
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

async def get_instruments(client, figi):
    try:
        await asyncio.sleep(1)
        r = client.instruments.find_instrument(query=figi)
        return r.instruments
    except Exception as e:
        print(f'Error getting instruments for {figi}: {e}')
        return []


async def save_candles_instruments_to_db(candles, instruments, figi):
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
            print(insert_query)
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

        for instrument in instruments:
            insert_query = """
                INSERT INTO figi (figi, name, instrument_type, ticker, class_code, isin)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (figi, ticker) DO UPDATE SET
                name = EXCLUDED.name,
                instrument_type = EXCLUDED.instrument_type,
                isin = EXCLUDED.isin,
                class_code = EXCLUDED.class_code;
            """
            print(insert_query)
            cursor.execute(insert_query, (
                figi,
                instrument.name,
                instrument.instrument_type,
                instrument.ticker,
                instrument.class_code,
                instrument.isin

            ))
        connection.commit()
        print(f"Saved {len(instruments)} to DB.")


    except Exception as e:
        print(f'Error saving to DB: {e}')
    finally:
        if connection:
            cursor.close()
            connection.close()


async def main():
    load_dotenv()
    api_token = os.getenv('TINKOFF_API_TOKEN')

    with Client(api_token) as client:
        figies = await load_figies()
        for figi in figies:
            print(figi)
            candles = await get_historical_candles(client, figi)
            #candles = {}
            instruments = await get_instruments(client, figi)
            print(instruments)
            await save_candles_instruments_to_db(candles, instruments, figi)




if __name__ == '__main__':
    asyncio.run(main())
