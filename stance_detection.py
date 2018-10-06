#! python
# coding: utf-8

import argparse
import logging
import random
import time

import numpy as np
import pandas as pd
from keras import backend, preprocessing
from keras.layers import Dense, Input, LSTM, Bidirectional
from keras.models import Model
from keras.models import load_model as load_keras_model
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from helpers import load_model, get_number

if __name__ == '__main__':
    # Настраиваем логирование:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--trainfile', help='Размеченный обучающий корпус', required=True)
    arg('--embeddings', help='Дистрибутивная модель', required=True)
    arg('--name', help='Человекочитаемое имя обучаемой модели', default='testrun')
    args = parser.parse_args()

    trainfile = args.trainfile
    embeddings_file = args.embeddings
    run_name = args.name

    logger.info('Загружаем корпус...')
    train_dataset = pd.read_csv(trainfile, sep='\t', header=0, compression="gzip")
    logger.info('Загрузка корпуса завершена')

    logger.info('Загружаем дистрибутивную модель...')
    emb_model = load_model(embeddings_file)
    logger.info('Загрузка дистрибутивной модели завершена')
    vocabulary = emb_model.vocab
    embedding_layer = emb_model.get_keras_embedding()

    (x_train, y_train) = train_dataset['text'], train_dataset['label']

    logger.info('%d обучающих текстов' % len(x_train))

    logger.info('Средняя длина обучающего текста: %s слов'
                % "{0:.1f}".format(np.mean(list(map(len, x_train.str.split()))), 1))

    classes = sorted(list(set(y_train)))
    num_classes = len(classes)
    logger.info('%d целевых классов' % num_classes)

    print('===========================')
    print('Распределение целевых классов в обучающих данных:')
    print(train_dataset.groupby('label').count())
    print('===========================')

    # Конвертируем текстовые метки классов в индексы
    y_train = [classes.index(i) for i in y_train]

    # Конвертируем индексы в бинарные матрицы (чтобы потом использовать с categorical_crossentropy loss)
    y_train = to_categorical(y_train, num_classes)
    print('Размерность целевых классов в обучающих данных:', y_train.shape)

    doc = random.choice(range(20))  # На всякий случай смотрим глазами на случайный документ из данных
    print('Случайный документ:', x_train[doc])
    x_train = [[get_number(w, vocab=vocabulary) for w in text.split()] for text in x_train]
    print('Он же после конвертации:', x_train[doc])

    max_seq_length = 20  # Паддинг: приводим все документы к этой длине (лишнее обрезаем, недостающее заполняем нулями)
    vectorized_train = preprocessing.sequence.pad_sequences(
        x_train, maxlen=max_seq_length, truncating='post', padding='post')
    print('Он же после паддинга:', vectorized_train[doc])
    print('Размерность текстов в обучающих данных:', vectorized_train.shape)

    #  Начинаем собирать нейронную сеть

    #  На вход принимаем последовательности фиксированной длины (max_seq_length)
    word_input = Input(shape=(max_seq_length,), name='Word_sequences')

    training = embedding_layer(word_input)  # Заменяем идентификаторы слов на их эмбеддинги

    lstm = LSTM(64)  # Определяем размерность выхода LSTM (https://keras.io/layers/recurrent/#LSTM)
    training = Bidirectional(lstm, name='LSTM')(training)  # Проходимся по последовательности двунаправленной LSTM

    output = Dense(num_classes, activation='softmax', name='Output')(training)  # Выходной слой, предсказывает классы

    model = Model(inputs=[word_input], outputs=output)  # Собираем функциональную модель Keras
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # компиляция модели

    # Выводим архитектуру модели, сохраняем её картинкой
    print(model.summary())
    plot_model(model, to_file=run_name + '.png', show_shapes=True)

    val_split = 0.1  # Какая часть данных будет использоваться в качестве оценочного датасета (validation data)

    # Обучаем скомпилированную модель на наших данных
    # Детали на https://keras.io/getting-started/functional-api-guide/
    start = time.time()
    history = model.fit(vectorized_train, y_train, epochs=10, verbose=1, validation_split=val_split, batch_size=8)
    end = time.time()
    training_time = int(end - start)
    logger.info('Обучение заняло %d секунд' % training_time)

    # Вручную отделяем от данных оценочную часть, чтобы натравить на неё полноценные метрики
    val_nr = int(len(vectorized_train) * val_split)
    x_val = vectorized_train[-val_nr:, :]
    y_val = y_train[-val_nr:, :]

    # Оцениваем обученную модель:
    score = model.evaluate(x_val, y_val, verbose=2)
    logger.info('Значение функции потерь на оценочном датасете: %s' % "{0:.4f}".format(score[0]))
    logger.info('Точность на оценочном датасете: %s' % "{0:.4f}".format(score[1]))

    # Используем функцию из sklearn чтобы посчитать F1 по каждому классу:
    predictions = model.predict(x_val)
    predictions = np.around(predictions)  # проецируем предсказания модели в бинарный диапазон {0, 1}

    # Конвертируем предсказания обратно из чисел в текстовые метки классов
    y_test_real = [classes[np.argmax(pred)] for pred in y_val]
    predictions = [classes[np.argmax(pred)] for pred in predictions]

    logger.info('Качество классификации на оценочном датасете (взвешенная F1):')
    print(classification_report(y_test_real, predictions))

    fscore = precision_recall_fscore_support(y_test_real, predictions, average='macro')[2]
    logger.info('Macro-F1 на оценочном датасете: %s' % "{0:.4f}".format(fscore))

    # Сохранение модели в файл
    model_filename = run_name + '.h5'
    model.save(model_filename)
    print('Модель сохранена в', model_filename)
    
    # Загрузка модели
    print('Загрузка готовой модели')
    model = load_keras_modelmodel(model_filename)
    text = input('Введите ваш текст: ')
    x = [[get_number(w, vocab=vocabulary) for w in text.split()]]
    vectorized = preprocessing.sequence.pad_sequences(
        x, maxlen=max_seq_length, truncating='post', padding='post')
    pred = model.predict(vectorized)
    print(np.around(pred))
    
    backend.clear_session()
