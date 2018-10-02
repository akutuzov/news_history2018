#! python3
# coding: utf-8

import gensim
from itertools import combinations


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


def jaccard(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)


def jaccard_f(words, models, row=10):
    distances = {}
    for word in words:
        distances[word] = {}
        associations = {}
        for m in models:
            model = models[m]
            word_neighbors = [i[0] for i in model.most_similar(positive=[word], topn=row)]
            associations[m.replace('.model', '')] = set(word_neighbors)
        for pair in combinations(associations.keys(), 2):
            similarity = jaccard(associations[pair[0]], associations[pair[1]])
            if len(associations.keys()) > 2:
                distances[word]['-'.join(pair)] = similarity
            else:
                distances[word] = similarity
    return distances
