#! python3
# coding: utf-8

import logging
import argparse
from helpers import load_model

# Небольшой скрипт с примерами того, что можно делать с дистрибутивными моделями в Gensim
# Модели можно найти на http://rusvectores.org/news_history/models/

if __name__ == '__main__':
    # Настраиваем логирование:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', help='Путь к файлу с моделью', required=True)
    args = parser.parse_args()

    embeddings_file = args.model  # Файл, содержащий word embedding модель

    logger.info('Загрузка модели...')
    emb_model = load_model(embeddings_file)

    logger.info('Модель загружена')

    logger.info('Размер словаря модели: %d' % len(emb_model.wv.vocab))

    logger.info('Пример слова в модели: "%s"' % emb_model.wv.index2word[10])

    while True:
        query = input("Введите ваш запрос (чтобы выйти, введите 'exit'):")
        if query == "exit":
            exit()
        words = query.strip().split()
        # Если в запросе 1 слово, вывести список ближайших ассоциатов
        if len(words) == 1:
            word = words[0]
            print(word)
            if word in emb_model:
                print('=====')
                print('Ассоциат\tБлизость')
                for i in emb_model.most_similar(positive=[word], topn=10):
                    print(i[0]+'\t', i[1])
                print('=====')
            else:
                print('%s отсутствует в модели' % word)

        # Если слов больше, найти среди них лишнее
        else:
            print('=====')
            print('Это слово выглядит странным среди остальных:', emb_model.doesnt_match(words))
            print('=====')

