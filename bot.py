import asyncio
import logging
import os
import requests
import json
import aiohttp
import urllib.parse
import re

from bs4 import BeautifulSoup

#from aiogram import Bot, Dispatcher, types
#from aiogram.utils import executor

from aiogram.filters import Command, StateFilter
from dotenv import load_dotenv
#from aiogram.contrib.fsm_storage.memory import MemoryStorage
#from aiogram import FSMContext
from aiogram.fsm.context import FSMContext
#from aiogram.types import Message, PhotoSize
#from states import SaveCommon
#from storage import add_photo
from aiogram.fsm.state import StatesGroup, State
#from aiogram.dispatcher.filters.state import State, StatesGroup
#from aiogram_calendar import simple_cal_callback, SimpleCalendar
from datetime import datetime

from aiogram_calendar import SimpleCalendar, SimpleCalendarCallback, DialogCalendar, DialogCalendarCallback, \
    get_user_locale
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, CallbackQuery, FSInputFile
from aiogram.utils.markdown import hbold
from aiogram.client.default import DefaultBotProperties
from aiogram.utils.keyboard import ReplyKeyboardBuilder,  InlineKeyboardBuilder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import catboost as cb
import lightgbm as lgb


import pandas as pd
import asyncpg
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent

import xml.etree.ElementTree as ET
import html

import mplfinance as mpf
import yfinance as yf
import uuid



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
bot_token = os.getenv('BOT_TOKEN')
yandex_folder = os.getenv('YANDEX_FOLDER')
yandex_api_key = os.getenv('YANDEX_API_KEY')
bot = Bot(token=bot_token)
dp = Dispatcher()
TAG_RE = re.compile(r'<[^>]+>')
database_dsn = os.getenv('DATABASE_URL')


async def get_suggestions(query, instrument_type='share'):
    conn = await asyncpg.connect(database_dsn)
    try:
        
        sql = f"SELECT ticker FROM figi WHERE instrument_type = '{instrument_type}' and ticker LIKE '{query.upper()}%' LIMIT 10"
        logger.info(sql)
        rows = await conn.fetch(sql) 
        suggestions = [row["ticker"] for row in rows]
        return suggestions
    finally:
        await conn.close()

async def get_data_stock(ticker, date_from=0, date_to = 'current_date', instrument_type='share'):
    conn = await asyncpg.connect(database_dsn)
    try:
        sql = f"SELECT  figi.ticker as ticker, open_price as Open, close_price as Close, low_price as Low, high_price as High, timestamp as Date, volume as Volume FROM quotes left join figi on figi.figi = quotes.figi WHERE instrument_type = '{instrument_type}' and figi.ticker = '{ticker}' and timestamp between '{date_from}' and '{date_to}'"
        logger.info(sql);
        rows = await conn.fetch(sql) 
        #suggestions = [row["ticker"] for row in rows]
        return rows
    finally:
        await conn.close()


async def fetch_news(url):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status() # Проверка на ошибки HTTP
                xml_string = await response.text()

            root = ET.fromstring(xml_string)
            data = []
             
            if root.findall('.//doc'): # Yandex Search API, некоторые RSS
                for doc in root.findall('.//doc'):
                    item_data = {}
                    item_data['title'] =  BeautifulSoup(html.unescape(ET.tostring(doc.find('.//title'), method='xml').decode()), "html.parser").get_text()
                    item_data['url'] = doc.find('.//url').text
                    item_data['snippet'] = doc.find('.//extended-text').text
                    item_data['yandex_cache_url'] = doc.find('.//saved-copy-url').text

                    data.append(item_data)

            # Преобразование в JSON
            #json_data = json.dumps(data, indent=2, ensure_ascii=False)
            
            return data

        except (aiohttp.ClientError, ET.ParseError) as e:
            print(f"Ошибка: {e}")
            return None

class InputData(StatesGroup):
    waiting_for_text = State()
    waiting_for_date_from = State()
    waiting_for_date_to = State()

@dp.message(StateFilter(None), Command("news"))
async def cmd_news(message: types.Message, state: FSMContext):
    await message.reply("Введите ключевое слово поиска:")
    await state.set_state(InputData.waiting_for_text)


@dp.message(InputData.waiting_for_text)
async def process_text(message: types.Message, state: FSMContext):
    await state.update_data(text=message.text.lower())
    await message.reply("Выберите дату 'От':", reply_markup=await SimpleCalendar().start_calendar())
    await state.set_state(InputData.waiting_for_date_from)


@dp.callback_query(SimpleCalendarCallback.filter(), InputData.waiting_for_date_from)
async def process_date_from(callback_query: CallbackQuery, callback_data: CallbackData, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        await state.update_data(date_from=date.strftime("%Y%m%d"))
        await callback_query.message.answer(f"Выбрана дата 'От': " + date.strftime("%Y%m%d"))
        await callback_query.message.answer("Выберите дату 'До':", reply_markup=await SimpleCalendar().start_calendar())
        await state.set_state(InputData.waiting_for_date_to)


@dp.callback_query(SimpleCalendarCallback.filter(), InputData.waiting_for_date_to)
async def process_date_to(callback_query: CallbackQuery, callback_data: CallbackData, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        data = await state.get_data()
        data['date_to'] = date.strftime("%Y%m%d")
        text = data['text']
        date_from = data['date_from']
        date_to = data['date_to']

        await callback_query.message.answer(f"Выбрана дата 'До': {data['date_to']}\n\n"
                                            f"Введенный текст: {text}\n"
                                            f"Дата 'От': {date_from}\n"
                                            f"Дата 'До': {date_to}")
        sites = ['forbes.ru','frankmedia.ru','finance.mail.ru','rbc.ru','finmarket.ru','finam.ru']
        for site in sites:
         date_period = date_from + '..' + date_to
         parameters = {'folderid' : yandex_folder,'apikey' : yandex_api_key, 'query' : 'date: ' + date_period + ' site:'+site+' '+ text, 'lr':'11316','l10n':'ru','sortby':'rlv','filter':'strict','maxpassages':5,'page':0}

         url = f"https://yandex.ru/search/xml?" + urllib.parse.urlencode(parameters) + "&groupby=attr%3D.mode%3Dflat.groups-on-page%3D10.docs-in-group%3D1" 
         #await callback_query.message.answer(url)
         news_data = await fetch_news(url)

        #if news_data and news_data['items']:
        #   summarizer = pipeline("summarization")
        #   await process_news(message, news_data, summarizer)
        #else:
        #    await callback_query.message.answer("Новостей по вашему запросу не найдено.")

         await process_news (callback_query.message, news_data)
        await state.clear()




async def process_news(message, news_data): 
    for i, item in enumerate(news_data):
        title = item['title']
        snippet = item['snippet']
        url = item['url']
        #full_text = await fetch_full_text(url)

        # Суммаризация
        #try:
        #    summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        #except Exception as e:
        #    print(f"Ошибка суммаризации: {e}")
        #    summary = "Суммаризация недоступна."

        await message.answer(f"<b>{title}</b>\n\n<a href='{url}'>Читать далее</a>\n\n{snippet}\n\n", parse_mode="HTML")
 
        



async def fetch_full_text(url): 
    await asyncio.sleep(1) 
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()



        except aiohttp.ClientError as e:
            print(f"Error fetching URL {url}: {e}")
            return f"Ошибка загрузки полного текста: {e}"

class InputDataStock(StatesGroup):
    waiting_for_text = State()
    waiting_for_ticker = State()
    waiting_for_date_from = State()
    waiting_for_date_to = State()

@dp.message(StateFilter(None), Command("stock"))
async def cmd_stock(message: types.Message, state: FSMContext):
    await message.reply("Введите первые буквы тикера:")
    await state.set_state(InputDataStock.waiting_for_text)

@dp.message(InputDataStock.waiting_for_text)
async def process_suggestions(message: types.Message, state: FSMContext):
    await message.reply("Введите тикер акции:")
    words = await get_suggestions(message.text, 'share')

    if words:
        builder = ReplyKeyboardBuilder()
        for word in words:
            builder.add(types.KeyboardButton(text=word, callback_data=word))
        builder.adjust(1)
        await message.answer("Выберите тикер:",reply_markup=builder.as_markup(resize_keyboard=True))
        await state.set_state(InputDataStock.waiting_for_ticker)
    else:
        await message.reply("Тикеров, начинающихся с этого текста, не найдено.")
        await state.clear()
    #await state.set_state(InputDataStock.waiting_for_ticker)


@dp.message(InputDataStock.waiting_for_ticker)
async def process_text_stock(message: types.Message, state: FSMContext):
    await message.answer('Выберите дату ', reply_markup=types.ReplyKeyboardRemove())
    await state.update_data(text=message.text.upper())
    await message.answer("От:", reply_markup=await SimpleCalendar().start_calendar())
    await state.set_state(InputDataStock.waiting_for_date_from)


@dp.callback_query(SimpleCalendarCallback.filter(), InputDataStock.waiting_for_date_from)
async def process_date_from_stock(callback_query: CallbackQuery, callback_data: CallbackData, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        await state.update_data(date_from=date.strftime("%Y-%m-%d"))
        await callback_query.message.answer(f"Выбрана дата 'От': " + date.strftime("%Y-%m-%d"))
        await callback_query.message.answer("Выберите дату 'До':", reply_markup=await SimpleCalendar().start_calendar())
        await state.set_state(InputDataStock.waiting_for_date_to)


@dp.callback_query(SimpleCalendarCallback.filter(), InputDataStock.waiting_for_date_to)
async def process_date_to_stock(callback_query: CallbackQuery, callback_data: CallbackData, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        data = await state.get_data()
        data['date_to'] = date.strftime("%Y-%m-%d")
        ticker = data['text']
        date_from = data['date_from']
        date_to = data['date_to']

        await callback_query.message.answer(f"Выбрана дата 'До': {data['date_to']}\n\n"
                                            f"Введенный ticker: {ticker}\n"
                                            f"Дата 'От': {date_from}\n"
                                            f"Дата 'До': {date_to}")
         #await callback_query.message.answer(url)
        stock_data = await get_data_stock(ticker, date_from, date_to, 'share')

        #if news_data and news_data['items']:
        #   summarizer = pipeline("summarization")
        #   await process_news(message, news_data, summarizer)
        #else:
        #    await callback_query.message.answer("Новостей по вашему запросу не найдено.")

        await process_plot (callback_query.message, stock_data, 'share')
        await state.clear()

class InputDataCurrency(StatesGroup):
    waiting_for_text = State()
    waiting_for_ticker = State()
    waiting_for_date_from = State()
    waiting_for_date_to = State()

@dp.message(StateFilter(None), Command("currency"))
async def cmd_currency(message: types.Message, state: FSMContext):
    await message.reply("Введите первые буквы тикера валюты:")
    await state.set_state(InputDataCurrency.waiting_for_text)

@dp.message(InputDataCurrency.waiting_for_text)
async def process_suggestions_currency(message: types.Message, state: FSMContext):
    await message.reply("Выберите тикер валюты:")
    words = await get_suggestions(message.text, 'currency')

    if words:
        builder = ReplyKeyboardBuilder()
        for word in words:
            builder.add(types.KeyboardButton(text=word, callback_data=word))
        builder.adjust(1)
        await message.answer("Выберите тикер:",reply_markup=builder.as_markup(resize_keyboard=True))
        await state.set_state(InputDataCurrency.waiting_for_ticker)
    else:
        await message.reply("Тикеров, начинающихся с этого текста, не найдено.")
        await state.clear()
    #await state.set_state(InputDataStock.waiting_for_ticker)


@dp.message(InputDataCurrency.waiting_for_ticker)
async def process_text_currency(message: types.Message, state: FSMContext):
    await message.answer('Выберите дату ', reply_markup=types.ReplyKeyboardRemove())
    await state.update_data(text=message.text.upper())
    await message.answer("От:", reply_markup=await SimpleCalendar().start_calendar())
    await state.set_state(InputDataCurrency.waiting_for_date_from)

@dp.callback_query(SimpleCalendarCallback.filter(), InputDataCurrency.waiting_for_date_from)
async def process_date_from_currency(callback_query: CallbackQuery, callback_data: CallbackData, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        await state.update_data(date_from=date.strftime("%Y-%m-%d"))
        await callback_query.message.answer(f"Выбрана дата 'От': " + date.strftime("%Y-%m-%d"))
        await callback_query.message.answer("Выберите дату 'До':", reply_markup=await SimpleCalendar().start_calendar())
        await state.set_state(InputDataCurrency.waiting_for_date_to)


@dp.callback_query(SimpleCalendarCallback.filter(), InputDataCurrency.waiting_for_date_to)
async def process_date_to_currency(callback_query: CallbackQuery, callback_data: CallbackData, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        data = await state.get_data()
        data['date_to'] = date.strftime("%Y-%m-%d")
        ticker = data['text']
        date_from = data['date_from']
        date_to = data['date_to']

        await callback_query.message.answer(f"Выбрана дата 'До': {data['date_to']}\n\n"
                                            f"Введенный ticker: {ticker}\n"
                                            f"Дата 'От': {date_from}\n"
                                            f"Дата 'До': {date_to}")
         #await callback_query.message.answer(url)
        currency_data = await get_data_stock(ticker, date_from, date_to, 'currency')

        #if news_data and news_data['items']:
        #   summarizer = pipeline("summarization")
        #   await process_news(message, news_data, summarizer)
        #else:
        #    await callback_query.message.answer("Новостей по вашему запросу не найдено.")

        await process_plot (callback_query.message, currency_data, 'share')
        await state.clear()


async def process_plot (message, data, instrument_type = 'share'):
    #for i, item in enumerate(stock_data):
    #    await message.answer(item['ticker'] + " " + item['timestamp'].strftime('%Y-%m-%d') + " " + str(item['close_price']) + "\n\n")
    try:
        # Загрузка данных с Yahoo Finance
        #data = yf.download(tickers='PONY', period='1mo', interval="1d", multi_level_index=False)
        # Создание DataFrame из данных PostgreSQL
        data = pd.DataFrame(data, columns=["ticker","Open","Close","Low","High","Date", "Volume"])
        data = data.set_index('Date')
        
        data.applymap(str)
        logger.info(data)
        
        filename = '/tmp/' + uuid.uuid4().hex + '.png'    
        mpf.plot(data, type="candle", volume=True, style="yahoo", savefig=filename)
        await message.answer_photo(FSInputFile(filename))
    except Exception as e:
        await message.reply(f"Ошибка: {e}")

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_message = """

Курс валюты 
кнопка «валюта»
Выбор валюты 
Выбор периода  
Выдача курса  

Новости по ключевым словам
кнопка «новости»
Впишите необходимое слово 
Бот ищет новости с данным словом по ТОП СМИ 
Выдача новостей(сокращенных с помощью нейросети) с ссылками на первоисточник и полный текст 

Котировки акций
кнопка акции 
Введите тикер акции (бот подсказывает по введенным буквам) 
Выбор периода 
Выдача данных 

Анализ рынка акции 
Кнопка анализ рынка 
Выбор способа (модель AI или анализ аномальных торгов)
При выборе «анализ аномалий» выводятся все пары акции согласна формуле (формулу скину)
При выборе AI, AI ищет точки роста с процентом вероятности (модель можно использовать на основе нескольких показателей рынок, «настроение» новостей, объемы торгов»
Выдают список акции с рекомендацией к покупкам 

Инструкция 
раздел с инструкцией по пользованию бота

Раздел оплата 
Пробный период - 7 дней 
Подписка на 1 год 
Подписка на 6 месяцев 
Подписка на 1 месяц
    """
    await message.answer(help_message)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Please click menu")


@dp.message(Command("subscription"))
async def cmd_start(message: types.Message):
    words = ["Подписка на 1 месяц", "Подписка на 6 месяцев", "Подписка на 1 год"  ]
    builder = ReplyKeyboardBuilder()
    for word in words:
            builder.add(types.KeyboardButton(text=word, callback_data=word)) 
    builder.adjust(1)
    await message.answer("Выберите подписку:",reply_markup=builder.as_markup(resize_keyboard=True))
    await message.answer("Выберите подписку")

@dp.message(Command("analyze"))
async def cmd_start(message: types.Message):
    words = ["LSTM", "GRU", "RNN", "XGBoost","CatBoost","LightGBM","Anomaly" ]
    builder = ReplyKeyboardBuilder()
    for word in words:
            builder.add(types.KeyboardButton(text=word, callback_data=word)) 
    builder.adjust(1)
    #await message.answer("Выберите модель анализа:",reply_markup=builder.as_markup(resize_keyboard=True))
    #await message.answer("Выберите модель")
    await message.answer("Рекомендуемые инструменты к покупке")
    tickers = ["SBER", "VTB", "GPG","ALFA","PSBR"]
    for ticker in tickers:
        await message.answer(ticker)


dict_new = {
    "Сталь": ['CHMF', 'MAGN', 'NLMK'],
}

def get_currency_rate(currency, period):
    pass


def final_table():
    pass

def create_dataset(dataset, look_back=60):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def analyze_market(method, ticker):
     
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

    data['SMA_5'] = data['Close'].rolling(window=5).mean()  
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data.dropna(inplace=True) 

    X = data.drop('Close', axis=1)
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    train_data_len = int(np.ceil(len(data) * .8))
    train_data = scaled_data[0:train_data_len, :]
    test_data = scaled_data[train_data_len - 60:, :]
    look_back = 60
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    # Reshape для GRU и CNN
    trainX_gru = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX_gru = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    trainX_cnn = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX_cnn = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    models = {
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        "CatBoost": cb.CatBoostRegressor(loss_function='RMSE', verbose=0, random_state=42),
        "LSTM": lgb.LGBMRegressor(objective='regression', random_state=42),
        "GRU": Sequential([
            GRU(50, return_sequences=True, input_shape=(trainX_gru.shape[1], 1)),
            Dropout(0.2),
            GRU(50),
            Dropout(0.2),
            Dense(1)
            ]),
        "CNN": Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(trainX_cnn.shape[1], 1)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
         ])
      }

    results = {}
    # XGBoost
    models["XGBoost"].fit(trainX, trainY)
    y_pred_xgb = models["XGBoost"].predict(testX)
    rmse_xgb = np.sqrt(mean_squared_error(testY, y_pred_xgb))
    results["XGBoost"] = rmse_xgb

    # GRU
    models["GRU"].compile(optimizer='adam', loss='mse')
    models["GRU"].fit(trainX_gru, trainY, epochs=25, batch_size=32, verbose=1)
    y_pred_gru = models["GRU"].predict(testX_gru)
    rmse_gru = np.sqrt(mean_squared_error(testY, y_pred_gru))
    results["GRU"] = rmse_gru

    # CNN
    models["CNN"].compile(optimizer='adam', loss='mse')
    models["CNN"].fit(trainX_cnn, trainY, epochs=25, batch_size=32, verbose=1)
    y_pred_cnn = models["CNN"].predict(testX_cnn)
    rmse_cnn = np.sqrt(mean_squared_error(testY, y_pred_cnn))
    results["CNN"] = rmse_cnn
    for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = rmse 
    

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

