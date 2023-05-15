"""
Necessary libs for bot running.
"""
from __future__ import annotations
from aiogram import Bot, Dispatcher, executor, types
from embeddings import TenClosestNews


class MyTelegramBot:
    """
    Class of our beloved telegram Bot.
    """

    def __init__(self, token: str):
        self.bot = Bot(token)
        self.dispatcher = Dispatcher(self.bot)
        self.tcn = TenClosestNews()
        self.tcn.create_faiss_index()

    def run(self):
        """
        Method for running the bot.
        """

        @self.dispatcher.message_handler(
            commands=["start", "help"], content_types=["text"]
        )
        async def start_response(message: types.Message) -> types.Message:
            """
            Reaction to /start or /help commands.
            Sends a welcome message and an example of enquiry.
            """
            start_msg_1 = (
                f"Привет, {message.from_user.first_name} {message.from_user.last_name}!"
                + " Этот бот позволяет найти 10 самых похожих новостей в датасете "
                + "IlyaGusev/gazeta. Напиши заголовок новости, по которому будем искать схожие :)"
            )

            await message.answer(start_msg_1)

        @self.dispatcher.message_handler(content_types=["text"])
        async def query_response(message: types.Message) -> types.Message:
            """
            This method reacts to any "enquery" message and tries to find
            similar news.
            """

            query_text = message.text

            df_resp = self.tcn.get_closest_news(query_text=query_text)

            response_text = ""
            for idx in df_resp.index:
                response_text += (
                    f"[{df_resp.loc[idx, 'title']}]({df_resp.loc[idx, 'url']})\n"
                )

            await self.bot.send_message(
                message.chat.id,
                response_text,
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )

        executor.start_polling(self.dispatcher)
