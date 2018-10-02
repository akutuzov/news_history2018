#! python3
# coding: utf-8

import logging
import argparse
from collections import OrderedDict
from helpers import load_model, jaccard_f, plot_diffs
from prettytable import PrettyTable

# Скрипт для сравнения ближайших ассоциатов слова в нескольких моделях

if __name__ == '__main__':
    # Настраиваем логирование:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--models', help='Список файлов с моделями', required=True)
    args = parser.parse_args()

    embeddings_files = args.models.split(',')  # Файлы, содержащие word embedding модели, через запятую

    models = OrderedDict()
    for modelfile in embeddings_files:
        logger.info('Загрузка модели %s...' % modelfile)
        emb_model = load_model(modelfile)
        models[modelfile.split('/')[-1].replace('.model', '')] = emb_model
        logger.info('Модель загружена')
        logger.info('Размер словаря модели: %d' % len(emb_model.vocab))
        logger.info('Пример слова в модели: "%s"' % emb_model.wv.index2word[10])

    while True:
        query = input("Введите ваш запрос (чтобы выйти, введите 'exit'):")
        if query == "exit":
            exit()
        word = query.strip()
        # Проверяем, есть ли слово во всех загруженных моделях:
        present = True
        for model in models:
            if word not in models[model].vocab:
                print('Слово отсутствует в модели %s' % model)
                present = False
        if present:
            similarities, associations = jaccard_f(word, models)
            logger.info('Ближайшие ассоциаты слова в загруженных моделях:')
            print('==========')
            table = PrettyTable(associations)
            for pos in range(9):
                table.add_row([associations[mod][pos] for mod in associations])
            print(table)
            logger.info('Разница с предыдущим годом по коэффициенту Жаккара:')
            years = []
            diffs = []
            for sim in similarities[word]:
                diff = 1 - similarities[word][sim]
                print(sim, round(diff, 2))
                years.append(sim)
                diffs.append(diff)
            plot_diffs(years, diffs, word)
            print('==========')
