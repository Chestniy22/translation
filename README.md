# translation
## Авторы

||||
----
||||

## Цель: 
Определение тональности текста отзывов на фильмы из IDB (Internet Movie Database).
## Задачи:
1. анализ существующих решений;
2. сбор данных;
3. исследование набора данных и предварительная подготовка;
4. создание нейронной сети, обучение и оценка моделей на исходном датасете;
5. проверка работы нейронной сети на тестовом наборе данных;
6. формирование выводов о работе нейронной сети;
7. выявление направлений улучшения распознавания.

## Набор данных:
https://ai.stanford.edu/~amaas/data/sentiment/ - набор включает отзывы на фильмы с сайта [IMDb](https://www.imdb.com/). Отзывы только явно положительные  (оценка >=7) или отрицательные (оценка <=4), нейтральные отзывы в набор данных не включались.
Размер набора  данных 50 тысяч отзывов:
Набор данных для обучения - 25 тыс.отзывов.
Набор данных для тестирования - 25  тыс. отзывов.

## Целесообразность:
Для повышения качества работы своего сайта или сервиса нужна обратна связь от пользователей.

## Разработка архитектуры системы
На рисунке ниже приведена диаграмма компонентов, описывающая структуру системы.
![image](https://user-images.githubusercontent.com/119978648/234894484-bcee3529-7626-4f05-b83d-3ebca514eb60.png)


На рисунке ниже представлена диаграмма компонентов процесса работы системы, архитектура которой представлена выше в виде диаграммы компонентов.
![image](https://user-images.githubusercontent.com/119978648/234893041-7f9b8418-4e70-4158-8601-22682b403893.png)


