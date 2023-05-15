# Что происходит?

В одном из чатов мне попалось на глаза весьма занятное тестовое задание на позицию MLE в неизвестную компанию. Стало интересно, какое количество времени уйдет на решение и смогу ли вообще его осилить.
Спойлер: рабочий вариант кода был готов в концу вечера, хоть и в формате джупайтеровского ноутбука. За следующей вечер была проведена декомпозиция из тетрадки в py-файлы с закосом под ООП и создание бота в ТГ.
Получившееся решение представляет собой скорее некий baseline того, как можно решить задачу, но что хорошо - за счет нехитрных изменений в коде можно подсунуть любой другой датасет и энкодер, благодаря чему, к примеру, моментально перейти к решению аналогичной задачи на другом языке. 

# **Содержание задания:** 

Вам предлагается разработать систему обратного поиска (reverse text search) для кратких описаний статей с ресурса Gazeta.ru.

Система обратного поиска позволяет искать самые похожие тексты из некоторого датасета на введенный пользователем текст.

Датасет: https://huggingface.co/datasets/IlyaGusev/gazeta

Вам необходимо для кратких описаний (поле **summary**) по получить векторное представление (эмбеддинг) с помощью предобученного нейросетевого энкодера и для призвольного текстового запроса также рассчитать эмбеддинг и найти топ-10 схожих описаний по убыванию косинусной близости (cosine similarity) эмбеддингов датасета.

В качестве энкодера предлагается взять модель [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2), но можно выбрать любой другой на основе архитектуры трансформер. 

На запрос пользователя система должна возвращать 10 результатов: само похожее краткое описание и ссылка на исходную новость. 

Код необходимо разметить на Github. Необходимо также отдельным файлом приложить результаты работы системы по следующим запрсоам:

1. МЧС прилагает все усилия для тушения лесных пожаров в Западной Сибири
2. Президент России Владимир Путин провёл встречу с министром МВД Колокольцевым
3. Британские учёные обнаружили новый вид рыб в Тихом океане
4. Астронавты НАСА провели выход в открытый космос на МКС

Во время демонстрации работы системы, необходимо будет рассказать о системе и принятых в процессе разработки решениях, продемонстрировать работу.

Будет плюсом (выполняется по желанию):

- Использование библиотеки FAISS для ускорения поиска похожих векторов
- Интерактивная работа с системой (простенький веб-интерфейс на Streamlit, телеграм-бот и т.д.)