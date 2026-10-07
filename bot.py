import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup

import config
from neural_network import classifier

session = AiohttpSession(proxy=config.PROXY_URL)

logging.basicConfig(level=logging.INFO)
bot = Bot(token=config.BOT_TOKEN, session=session)
dp = Dispatcher()

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Помощь")]
    ],
    resize_keyboard=True
)

@dp.message(Command("start"))
async def start_handler(message: Message):
    text = "Привет! Я бот для распознавания жестового языка."
    await message.answer(text, reply_markup=start_keyboard)

@dp.message(F.text == "Помощь")
async def help_handler(message: Message):
    text = "Отправьте жест в виде фото, чтобы получить предсказанный жест."
    await message.answer(text)

@dp.message(F.photo)
async def handle_photo(message: Message):
    try:        
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        photo_bytes = await bot.download_file(file.file_path)
        
        predicted_class, confidence = classifier.predict(photo_bytes.read())
        
        response = f"Жест: {config.GESTURE_CLASSES[predicted_class]}\nТочность: {confidence:.1%}"
        await message.answer(response)
        
    except Exception:  # noqa: BLE001
        await message.answer("Ошибка обработки фото")

@dp.message()
async def other_messages(message: Message):
    await message.answer("Используйте кнопки или отправьте фото")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())