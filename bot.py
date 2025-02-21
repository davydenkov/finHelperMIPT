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
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, CallbackQuery
from aiogram.utils.markdown import hbold
from aiogram.client.default import DefaultBotProperties

import xml.etree.ElementTree as ET
import html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
bot_token = os.getenv('BOT_TOKEN')
yandex_folder = os.getenv('YANDEX_FOLDER')
yandex_api_key = os.getenv('YANDEX_API_KEY')
bot = Bot(token=bot_token)
dp = Dispatcher()
TAG_RE = re.compile(r'<[^>]+>')

class InputData(StatesGroup):
    waiting_for_text = State()
    waiting_for_date_from = State()
    waiting_for_date_to = State()


class NewsRequest(StatesGroup):
    waiting_for_query = State()
    waiting_for_period = State()

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
        sites = ['forbes.ru','frankmedia.ru','finance.mail.ru','t-j.ru','finmarket.ru']
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

@dp.message(NewsRequest.waiting_for_query)
async def process_query(message: types.Message, state: FSMContext):
    async with state.proxy() as data:
        data['query'] = message.text

    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(types.KeyboardButton("За сегодня"))
    keyboard.add(types.KeyboardButton("За 7 дней"))
    keyboard.add(types.KeyboardButton("За месяц"))
    await message.reply("Выберите период:", reply_markup=keyboard)
    await state.set_state(NewsRequest.waiting_for_period.state) 




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





@dp.message(NewsRequest.waiting_for_period)
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
async def cmd_help(message: types.Message):
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

