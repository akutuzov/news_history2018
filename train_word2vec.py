#! python3
# coding: utf-8

import gensim
import logging
import multiprocessing
import argparse

# Этот скрипт обучает word2vec-модель, используя Gensim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--corpus', help='Путь к обучающему корпусу', required=True)
    args = parser.parse_args()

    # Настраиваем логирование:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Это обучающий корпус, из которого мы получим эмбеддинги для слов
    # Скорее всего, это сжатый текстовый файл с одним предложением/документом на строку:
    corpus = args.corpus

    data = gensim.models.word2vec.LineSentence(corpus)  # Итератор по строкам корпуса

    cores = multiprocessing.cpu_count()  # Используем все доступные процессорные ядра для обучения
    logger.info('Используем ядер: %d' % cores)

    # Гиперпараметры обучения:
    skipgram = 0  # Алгоритм Skipgram (1) или CBOW (0)?
    window = 5  # Размер симметрического контекстного окна (например, 2 слова слева и 2 слова справа).
    vocabsize = 10000  # Сколько максимально слов мы хотим в словаре модели (сортируются по частоте)?
    vectorsize = 300  # Размерность векторов слов.
    mincount = 5  # Игнорировать слов, встретившиеся в корпусе реже, чем этот порог.
    iterations = 5  # Сколько эпох будем обучаться (сколько проходов по корпусу)?

    # Начинаем обучение!
    # Субсэмплинг (гиперпараметр 'sample') используется, чтобы стохастически снижать влияние очень частых слов.
    # Поскольку наши корпуса уже очищены от стоп-слов (функциональных частей речи),
    # мы не нуждаемся в субсэмплинге и устанавливаем sample=0.
    model = gensim.models.Word2Vec(data, size=vectorsize, window=window, workers=cores, sg=skipgram,
                                   max_final_vocab=vocabsize, min_count=mincount, iter=iterations, sample=0)

    model = model.wv  # После окончания обучения удаляем служебные матрицы весов, они нам больше не нужны.

    # Сохраняем модель в файл
    filename = corpus.replace('.txt.gz', '') + '.model'
    logger.info(filename)
    model.save(filename)
