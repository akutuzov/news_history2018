#! python3
# coding: utf-8

import numpy as np
import gensim
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


def visual(focusword, wordslists, matrices, usermodels):
    fig = plt.figure()
    for words, matrix, usermodel, nr in zip(wordslists, matrices, usermodels, range(len(wordslists))):
        pca = PCA(n_components=2)
        y = pca.fit_transform(matrix)

        xpositions = y[:, 0]
        ypositions = y[:, 1]

        if len(wordslists) <= 4:
            rows = 2
            columns = 2
        elif 4 < len(wordslists) < 7:
            rows = 2
            columns = 3
        else:
            rows = 3
            columns = len(wordslists) / rows
        ax = fig.add_subplot(rows, columns, nr + 1)
        for word, x, y in zip(words, xpositions, ypositions):
            lemma = word.split('_')[0].replace('::', ' ')
            bias = 0.05
            if word == focusword:
                ax.scatter(x, y, 200, marker='*', color='red')
                ax.annotate(lemma, xy=(x - bias, y), size='x-large', weight='bold', color='red', alpha=0.8)
            else:
                ax.scatter(x, y, 150, marker='.', color='green')
                ax.annotate(lemma, xy=(x - bias, y), size='large', alpha=0.8)

        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax.set_title('"%s" в %s' % (focusword.split('_')[0].replace('::', ' '), usermodel))

    plt.show()


def wordvectors(words, emb_model):
    matrix = np.zeros((len(words), emb_model.vector_size))
    for i in range(len(words)):
        matrix[i, :] = emb_model[words[i]]
    return matrix


def get_number(word, vocab=None):
    if word in vocab:
        return vocab[word].index
    else:
        return 0
