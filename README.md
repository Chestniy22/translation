# translation

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
https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz - набор включает отзывы на фильмы с сайта [IMDb](https://www.imdb.com/). Отзывы только явно положительные  (оценка >=7) или отрицательные (оценка <=4), нейтральные отзывы в набор данных не включались.
Размер набора  данных 50 тысяч отзывов:
Набор данных для обучения - 25 тыс. отзывов.
Набор данных для тестирования - 25  тыс. отзывов.

## Целесообразность:
Для повышения качества работы своего сайта или сервиса нужна обратна связь от пользователей. Описание набора данных - https://ai.stanford.edu/~amaas/data/sentiment/.

## Разработка архитектуры системы
На рисунке ниже приведена диаграмма компонентов, описывающая структуру системы.
![Архитектура системы drawio (2)](https://user-images.githubusercontent.com/119978648/234927516-092c9400-98e0-4c87-855d-17b103ea073f.png)

На рисунке ниже представлена диаграмма компонентов процесса работы системы, архитектура которой представлена выше в виде диаграммы компонентов.
![Диаграмма активностей drawio](https://user-images.githubusercontent.com/119978648/234905526-56c624ef-60df-4f5a-900c-94ef69553ee9.png)


