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


def jaccard(list0, list1):
    set_0 = set(list0)
    set_1 = set(list1)
    n = len(set_0.intersection(set_1))
    return n / (len(set_0) + len(set_1) - n)


def jaccard_f(word, models, row=10):
    associations = {}
    similarities = {word: {}}
    for m in models:
        model = models[m]
        word_neighbors = [i[0] for i in model.most_similar(positive=[word], topn=row)]
        associations[m] = word_neighbors
    for pair in combinations(associations.keys(), 2):
        similarity = jaccard(associations[pair[0]], associations[pair[1]])
        similarities[word]['-'.join(pair)] = similarity
    return similarities, associations
