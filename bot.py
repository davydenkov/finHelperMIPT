import asyncio
import logging
import os

#from aiogram import Bot, Dispatcher, types
#from aiogram.utils import executor

from aiogram.filters.command import Command
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
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, CallbackQuery
from aiogram.utils.markdown import hbold
from aiogram.client.default import DefaultBotProperties

import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO)
load_dotenv()
bot_token = os.getenv('BOT_TOKEN')
bot = Bot(token=bot_token)

dp = Dispatcher()

class InputData(StatesGroup):
    waiting_for_text = State()
    waiting_for_date_from = State()
    waiting_for_date_to = State()

def parse_xml_to_array(xml_string):
    try:
        root = ET.fromstring(xml_string) # Парсим XML-строку
        data = []

        # Проверяем, есть ли элементы 'item' в XML
        if root.findall('.//item'): # Yandex Search API
            for item in root.findall('.//item'):
                item_data = {}
                for child in item:
                    item_data[child.tag] = child.text
                data.append(item_data)
        elif root.findall('.//channel/item'): # RSS, Atom and others
            for item in root.findall('.//channel/item'):
                item_data = {}
                for child in item:
                    item_data[child.tag] = child.text
                data.append(item_data)    
        else: # Если 'item' не найден, обрабатываем все элементы
            for element in root.iter():
                if element.text and element.text.strip(): # Проверяем на пустые элементы
                    data.append({element.tag: element.text.strip()})

        return data

    except ET.ParseError as e:
        print(f"Ошибка парсинга XML: {e}")
        return None



@dp.message(Command("help"))
async def cmd_start(message: types.Message):
    await message.reply("Введите текст:")
    await InputData.waiting_for_text.set()


@dp.message(state=InputData.waiting_for_text)
async def process_text(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['text'] = message.text

    await message.reply("Выберите дату 'От':", reply_markup=await SimpleCalendar().start_calendar())
    await InputData.waiting_for_date_from.set()


@dp.callback_query_handler(simple_cal_callback.filter(), state=InputData.waiting_for_date_from)
async def process_date_from(callback_query: types.CallbackQuery, callback_data: dict, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        async with state.proxy() as data:
            data['date_from'] = date.strftime("%Y-%m-%d")

        await callback_query.message.answer(f"Выбрана дата 'От': {data['date_from']}")
        await callback_query.message.answer("Выберите дату 'До':", reply_markup=await SimpleCalendar().start_calendar())
        await InputData.waiting_for_date_to.set()


@dp.callback_query_handler(simple_cal_callback.filter(), state=InputData.waiting_for_date_to)
async def process_date_to(callback_query: types.CallbackQuery, callback_data: dict, state: FSMContext):
    selected, date = await SimpleCalendar().process_selection(callback_query, callback_data)
    if selected:
        async with state.proxy() as data:
            data['date_to'] = date.strftime("%Y-%m-%d")
            text = data['text']
            date_from = data['date_from']


        await callback_query.message.answer(f"Выбрана дата 'До': {data['date_to']}\n\n"
                                            f"Введенный текст: {text}\n"
                                            f"Дата 'От': {date_from}\n"
                                            f"Дата 'До': {data['date_to']}")
        await state.finish()

@dp.message_handler(state=NewsRequest.waiting_for_query)
async def process_query(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['query'] = message.text

    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(types.KeyboardButton("За сегодня"))
    keyboard.add(types.KeyboardButton("За 7 дней"))
    keyboard.add(types.KeyboardButton("За месяц"))
    await message.reply("Выберите период:", reply_markup=keyboard)
    await state.set_state(NewsRequest.waiting_for_period.state) 


class NewsRequest(StatesGroup): 
    waiting_for_query = State()
    waiting_for_period = State() 



@dp.message_handler(state=NewsRequest.waiting_for_period)
async def process_period(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        query = data['query']
        period = message.text

    days = 0
    if period == "За сегодня":
        days = 1
    elif period == "За 7 дней":
        days = 7
    elif period == "За месяц":
        days = 30

    date_to = datetime.now().strftime('%Y-%m-%d')
    date_from = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = f"https://yandex.ru/search/xml?folderid=b1g0vfs986msn9pjmb4d&apikey=AQVN1YdOnecDm7AOI24D8gixs0BbZMfQMirJJ7NX&query=site%3Aforbes.ru%20%D0%BF%D0%BE%D0%BB%D1%8E%D1%81%D0%B7%D0%BE%D0%BB%D0%BE%D1%82%D0%BE&lr=11316&l10n=ru&sortby=rlv&filter=strict&groupby=attr%3D.mode%3Dflat.groups-on-page%3D10.docs-in-group%3D1&maxpassages=5&page=0"
    
    await state.finish()



async def fetch_news(session, url):
    async with session.get(url) as response:
        return await response.json()



async def process_news(message, news_data, summarizer): 
    for i, item in enumerate(news_data['items'][:MAX_NEWS]):
        title = item['title']
        snippet = item['snippet']
        url = item['link']
        full_text = await fetch_full_text(url)

        # Суммаризация
        try:
            summary = summarizer(full_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        except Exception as e:
            print(f"Ошибка суммаризации: {e}")
            summary = "Суммаризация недоступна."

        await message.answer(f"*{title}*\n\n{summary}\n\n[Читать далее]({url})\n\n```\n{full_text}\n```",
                             parse_mode="MarkdownV2")



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





@dp.message_handler(state=NewsRequest.waiting_for_period)
async def process_period(message: types.Message, state: FSMContext):

    async with aiohttp.ClientSession() as session:
        news_data = await fetch_news(session, url)

        if news_data and news_data['items']:
           summarizer = pipeline("summarization") 
           await process_news(message, news_data, summarizer)
        else:
            await message.reply("Новостей по вашему запросу не найдено.")

    await state.finish()


@dp.message(Command("help"))
async def cmd_start(message: types.Message):
    await message.answer("Help docs")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Please sea menu")


dict_new = {
    "Сталь": ['CHMF', 'MAGN', 'NLMK'],
    # ... остальные данные ...
}

def get_currency_rate(currency, period):
    pass

def get_news(keyword, period):
    # Возвращает список словарей: [{'title': заголовок, 'short_summary': краткое описание, 'link': ссылка, 'full_text': полный текст}]
    pass

def get_stock_quotes(ticker, period):
    async with state.proxy() as data:
        data['days'] = int(message.text)

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        ticker = data['ticker']
        days = data['days']

        query = f"""
            SELECT timestamp, close_price
            FROM quotes
            WHERE figi = '{ticker}' 
            AND timestamp >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY timestamp;
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            await message.reply(f"Нет данных для тикера {ticker} за последние {days} дней.")
            await state.finish()
            return

        df = pd.DataFrame(rows, columns=['timestamp', 'close_price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        plt.plot(df['timestamp'], df['close_price'])
        plt.xlabel('Дата')
        plt.ylabel('Цена закрытия')
        plt.title(f'Котировки {ticker} за {days} дней')
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        await message.reply_photo(buf)
        plt.close() 

    except Exception as e:
        await message.reply(f"Произошла ошибка: {e}")
        logging.error(f"Error: {e}")

    finally:
        if conn:
            cursor.close()
            conn.close()
        await state.finish()

def final_table():
    pass


def analyze_market(method, ticker):
    if method == "AI":
     pass
    elif method == "аномалии":
        return final_table() # Вызов вашей функции
    else:
        return "Неверный метод анализа."

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

