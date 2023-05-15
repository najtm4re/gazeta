# Что происходит?

В одном из чатов мне попалось на глаза весьма занятное тестовое задание на позицию MLE в неизвестную компанию. Стало интересно, какое количество времени уйдет на решение и смогу ли вообще его осилить.
Спойлер: ноутбук с реализацией поиска был готов к концу вечера. За следующей вечер была тетрадка была разбита в py-файлы с закосом под ООП и создан ТГ бот.
Получившееся решение представляет собой скорее некий baseline того, как можно решить задачу, но что хорошо - за счет нехитрых изменений в коде можно подсунуть любой другой датасет и энкодер, благодаря чему, к примеру, моментально перейти к решению аналогичной задачи на другом языке. 

# **Содержание задания:** 

Вам предлагается разработать систему обратного поиска (reverse text search) для кратких описаний статей с ресурса Gazeta.ru.

Система обратного поиска позволяет искать самые похожие тексты из некоторого датасета на введенный пользователем текст.

Датасет: https://huggingface.co/datasets/IlyaGusev/gazeta

Вам необходимо для кратких описаний (поле **summary**) по получить векторное представление (эмбеддинг) с помощью предобученного нейросетевого энкодера и для призвольного текстового запроса также рассчитать эмбеддинг и найти топ-10 схожих описаний по убыванию косинусной близости (cosine similarity) эмбеддингов датасета.

В качестве энкодера предлагается взять модель [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2), но можно выбрать любой другой на основе архитектуры трансформер. 

На запрос пользователя система должна возвращать 10 результатов: само похожее краткое описание и ссылка на исходную новость. 

Код необходимо разметить на Github. Необходимо также отдельным файлом приложить результаты работы системы по следующим запросам:

1. МЧС прилагает все усилия для тушения лесных пожаров в Западной Сибири
2. Президент России Владимир Путин провёл встречу с министром МВД Колокольцевым
3. Британские учёные обнаружили новый вид рыб в Тихом океане
4. Астронавты НАСА провели выход в открытый космос на МКС

Во время демонстрации работы системы, необходимо будет рассказать о системе и принятых в процессе разработки решениях, продемонстрировать работу.

Будет плюсом (выполняется по желанию):

- Использование библиотеки FAISS для ускорения поиска похожих векторов
- Интерактивная работа с системой (простенький веб-интерфейс на Streamlit, телеграм-бот и т.д.)

# Запуск системы на своем устройстве

В репозитории присутствует файл pyproject.toml, в котором указаны все зависимости. Поэтому если выгрузить репозиторий и прозвести poetry install, по идее достаточно будет запустить файл bot_run, предварительно вписав туда токен своего бота :)
Токенанизация датасета занимает довольно много времени (в моем случае около 5 часов), однако, если она была выполнена, то результат должен сохраниться в отдельном файле и при следующих вызываться из него.

# Результат работы системы

[Запись создания запросов и получения ответов](https://youtu.be/GGg_lbkZOuc)

Запрос "МЧС прилагает все усилия для тушения лесных пожаров в Западной Сибири":

[Огонь не уходит из Сибири](https://www.gazeta.ru/social/2012/09/21/4782541.shtml)  
[Меньше леса, больше торф](https://www.gazeta.ru/social/2010/08/06/3405114.shtml)  
[Сибирь загорелась в «Столбах»](https://www.gazeta.ru/social/2012/06/15/4627485.shtml)  
[Россия выгорает с центра](https://www.gazeta.ru/social/2010/07/30/3402842.shtml)  
[Путин поручил спорные вопросы в отношении пострадавших от пожаров решать в пользу людей](https://www.gazeta.ru/social/2021/08/06/13841186.shtml)  
[Лес сгорел молча](https://www.gazeta.ru/social/2010/09/14/3419595.shtml)  
[«Дымовая ловушка»: как борются с пожарами в Сибири](https://www.gazeta.ru/social/2019/07/31/12545935.shtml)  
[Хакасия в дыму без огня](https://www.gazeta.ru/social/2012/08/04/4710533.shtml)  
[Смог вернулся](https://www.gazeta.ru/social/2014/07/30/6151689.shtml)  
[Россия горит с востока](https://www.gazeta.ru/social/2012/06/18/4630265.shtml)  

Запрос "Президент России Владимир Путин провёл встречу с министром МВД Колокольцевым":

[«Они хотят слишком много»: когда встретятся Путин с Трампом](https://www.gazeta.ru/politics/2018/06/02_a_11784391.shtml)  
[«Необычайный человек»: Трамп рассказал о Путине](https://www.gazeta.ru/politics/2019/06/29_a_12464053.shtml)  
[«Это не тот Путин, которого боится Запад»](https://www.gazeta.ru/politics/2011/11/11_a_3830502.shtml)  
[Путин прошелся по палатам](https://www.gazeta.ru/politics/2017/12/25_a_11534888.shtml)  
[Лавров ответил шуткой на вопрос о встрече Путина и Байдена](https://www.gazeta.ru/politics/2021/05/20_a_13600082.shtml)  
[Лукашенко заявил о готовности к усилению отношений Белоруссии и России](https://www.gazeta.ru/politics/2020/11/26_a_13375843.shtml)  
[Верить или нет? Новые обещания Пашиняна в Москве](https://www.gazeta.ru/politics/2018/06/13_a_11796091.shtml)  
[Президент России встретился с канцлером Германии](https://www.gazeta.ru/politics/2021/08/20_a_13897628.shtml)  
[Кремль и Белый дом прорабатывают встречу президентов](https://www.gazeta.ru/politics/2017/11/08_a_10976174.shtml)  
[Слово России: Москва не оставит Корею Америке](https://www.gazeta.ru/politics/2018/05/31_a_11781769.shtml)  

Запрос "Британские учёные обнаружили новый вид рыб в Тихом океане":

[Осьминоги скрываются от одиночества в городах](https://www.gazeta.ru/science/2017/09/19_a_10896728.shtml)  
[«Какое-то чудовище»: к берегам России приплыл сельдяной король](https://www.gazeta.ru/science/2019/08/08_a_12565819.shtml)  
[«Они даже не поняли, что их погубило»](https://www.gazeta.ru/science/2017/12/21_a_11508686.shtml)  
[Прозрачные гады покраснели](https://www.gazeta.ru/science/2011/11/15_a_3834514.shtml)  
[Медуза-НЛО и бронированный петух с Самоа](https://www.gazeta.ru/science/2017/03/11_a_10569263.shtml)  
[Ученые обнаружили на дне океана горы, напоминающие Мордор](https://www.gazeta.ru/science/2021/07/22_a_13787168.shtml)  
[«Понятия не имеем, как»: ученые нашли тюленя с угрем в носу](https://www.gazeta.ru/science/2018/12/07_a_12086899.shtml)  
[Первые дайверы: как неандертальцы ныряли за ракушками](https://www.gazeta.ru/science/2020/01/16_a_12913028.shtml)  
[Акула улыбнулась «Фортуне»](https://www.gazeta.ru/social/2011/09/12/3763649.shtml)  
[Чудо-юдо мшанка-мозг](https://www.gazeta.ru/science/2017/09/04_a_10873310.shtml)  

Запрос "Астронавты НАСА провели выход в открытый космос на МКС":

[Международный космический юбилей](https://www.gazeta.ru/social/2010/06/16/3385595.shtml)  
[С Байконура на МКС отправился российско-американский экипаж](https://www.gazeta.ru/science/2021/04/09_a_13552172.shtml)  
[10 лет висело: космонавты забрали из космоса полотенце](https://www.gazeta.ru/science/2019/05/29_a_12383551.shtml)  
[Трое в «Союзе»: космонавтов отправили на МКС](https://www.gazeta.ru/science/2017/12/17_a_11493818.shtml)  
[Дыра в «Союзе»: что обнаружили эксперты](https://www.gazeta.ru/science/2018/12/13_a_12094165.shtml)  
[Поздно отстыковались](https://www.gazeta.ru/social/2012/11/19/4858529.shtml)  
[«Не хотите ли черешни?»: как встречали космонавта](https://www.gazeta.ru/science/2018/06/03_a_11785513.shtml)  
[Никакого «Прогресса»](https://www.gazeta.ru/social/2011/08/24/3743729.shtml)  
[«Слишком поздно»: Starliner не попал на МКС](https://www.gazeta.ru/science/2019/12/20_a_12875804.shtml)  
[Назло ветрам и шатдауну: США запустили спутник-шпион](https://www.gazeta.ru/science/2019/01/20_a_12135457.shtml)  
