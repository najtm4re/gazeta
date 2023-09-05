"""
Importing bot class
"""
from lib.similar_news_bot import MyTelegramBot

bot = MyTelegramBot(token="xxxxx")

while True:
    bot.run()
