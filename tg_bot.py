import asyncio
from io import BytesIO

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram.enums import ContentType
from aiogram.types import FSInputFile
from aiogram.types import MessageReactionUpdated

BOT_TOKEN = "8690757293:AAGemRDl7JJKJPCJ2ssXKPZ9yNLxkrs28RM"
API_URL = "http://127.0.0.1:8000/api/chat"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! Отправь текстовое сообщение или .docx файл.\n"
        "Можно сначала прислать файл, потом текст.\n"
    )




@dp.message(F.content_type == ContentType.TEXT)
async def text_handler(message: Message):
    user_text = message.text or ""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                API_URL,
                data={"user_text": user_text},
            )

        payload = response.json()

        if response.status_code != 200 or not payload.get("ok"):
            await message.answer(f"Ошибка API: {payload}")
            return

        await message.answer(payload["answer"])

    except Exception as e:
        await message.answer(f"Ошибка при обращении к API: {e}")


@dp.message(F.content_type == ContentType.DOCUMENT)
async def document_handler(message: Message):
    document = message.document

    if document is None:
        await message.answer("Документ не получен.")
        return

    filename = document.file_name or "file"
    if not filename.lower().endswith(".docx"):
        await message.answer("Пожалуйста, отправьте файл в формате .docx")
        return

    try:
        telegram_file = await bot.get_file(document.file_id)
        file_buffer = BytesIO()
        await bot.download_file(telegram_file.file_path, destination=file_buffer)
        file_buffer.seek(0)

        files = {
            "docx_file": (
                filename,
                file_buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                API_URL,
                data={"user_text": ""},
                files=files,
            )

        payload = response.json()

        if response.status_code != 200 or not payload.get("ok"):
            await message.answer(f"Ошибка API: {payload}")
            return

        answer = payload["answer"]
        await message.answer(answer)

    except Exception as e:
        await message.answer(f"Ошибка при обработке документа: {e}")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())