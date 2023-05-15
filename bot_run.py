"""
Importing bot class
"""
from similar_news_bot import MyTelegramBot

bot = MyTelegramBot(token="xxxxxxxxxxxxxxxxxxxxxxxx")

while True:
    bot.run()
