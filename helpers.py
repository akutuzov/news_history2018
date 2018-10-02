#! python3
# coding: utf-8

import gensim
from collections import OrderedDict
import matplotlib.pyplot as plt


def load_model(embeddings_file):
    # Определяем формат модели по её расширению:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):  # Бинарный формат word2vec
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):  # Текстовый формат word2vec
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    else:  # Нативный формат Gensim?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)
    emb_model.init_sims(replace=True)  # На всякий случай приводим вектора к единичной длине (нормируем)
    return emb_model


def jaccard(list0, list1):
    set_0 = set(list0)
    set_1 = set(list1)
    n = len(set_0.intersection(set_1))
    return n / (len(set_0) + len(set_1) - n)


def jaccard_f(word, models, row=10):
    associations = OrderedDict()
    similarities = {word: OrderedDict()}
    previous_state = None
    for m in models:
        model = models[m]
        word_neighbors = [i[0] for i in model.most_similar(positive=[word], topn=row)]
        associations[m] = word_neighbors
        if previous_state:
            similarity = jaccard(previous_state[1], word_neighbors)
            similarities[word][m] = similarity
        previous_state = (m, word_neighbors)
    return similarities, associations


def plot_diffs(years, diffs, word, savefigure=False):
    plt.figure(1)
    plt.plot(years, diffs, 'bo--', linewidth=2)
    plt.xlabel('Годы')
    plt.ylabel('Расстояние Жаккара по сравнению с предыдущим годом')
    plt.title('Изменения в значении слова "%s"' % word)
    if savefigure:
        plt.savefig(savefigure)
    else:
        plt.show()
