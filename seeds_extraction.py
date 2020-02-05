#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np
from utils import clean_str, stopsets, category_id_dict

import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])


def line_to_words(line, stop_words, lemmatize=True):
    # clean sentence and break it into EDUs
    edu = clean_str(line.strip())
    segs = []
    doc = nlp(edu)
    if lemmatize:
        words_lemma = [token.lemma_ for token in doc]
        # words_lemma = [token.lemma_ for token in doc]
    else:
        words_lemma = [token.text for token in doc]
    stop_idx = set([i for i, token in enumerate(words_lemma) if token in stop_words])
    words = [w for w in words_lemma if w not in stop_words]
    return [words]


if __name__ == '__main__':

    corpus_doc = []
    corpus_sent_cnt = []

    title_doc = []
    corpus_prod_cnt = [0] * 6  # cnt of products in each domain

    domain_list = category_id_dict.keys()
    for d_idx, domain in enumerate(domain_list):
        corpus_doc.append([])
        corpus_sent_cnt.append([])
        with open('data/preprocessed/description/{}_product_info.json'.format(domain.upper()), 'r') as f:
            desc_data = json.load(f)

            for prod in desc_data:
                corpus_sent_cnt[-1].append(0)
                if prod is None:
                    continue
                if len(prod['DESCRIPTION']) > 8 or len(prod['DESCRIPTION']) < 3:
                    continue
                corpus_doc[-1].append([])
                title = line_to_words(prod['NAME'], stopsets, True)
                title_doc.append(' '.join([w for w in title[0] if w not in stopsets]))
                corpus_prod_cnt[d_idx] += 1
                title_word_set = set(title[0])
                for desc in prod['DESCRIPTION']:
                    desc = line_to_words(desc, stopsets, True)
                    if not len(desc):
                        continue
                    word_list = [w for w in desc[0]]
                    corpus_doc[-1][-1].append(' '.join(word_list))
                    corpus_sent_cnt[-1][-1] += 1

    small_doc_list = [desc for domain_doc in corpus_doc for prod_doc in domain_doc for desc in prod_doc]
    mid_doc_list = [' '.join(prod_doc) for domain_doc in corpus_doc for prod_doc in domain_doc]
    large_doc_list = [' '.join([desc for prod_doc in domain_doc for desc in prod_doc]) for domain_doc in corpus_doc]

    corpus_sent_cnt = [sum(d) for d in corpus_sent_cnt]  # cnt of descriptive sentences in each domain

    corpus_sent_slides = [0]
    corpus_title_slides = [0]
    for c1, c2 in zip(corpus_sent_cnt, corpus_prod_cnt):
        corpus_sent_slides.append(corpus_sent_slides[-1] + c1)
        corpus_title_slides.append(corpus_title_slides[-1] + c2)

    for i in range(len(domain_list)):
        domain_tfidf_doc = [large_doc_list[i]]
        begin_idx, end_idx = corpus_sent_slides[i], corpus_sent_slides[i + 1]
        domain_tfidf_doc += (small_doc_list[:begin_idx] + small_doc_list[end_idx:])

        # domain_tfidf_doc += (title_doc[corpus_title_slides[i]:corpus_title_slides[i+1]]*10)

        vectorizer = TfidfVectorizer(sublinear_tf=False)
        X = vectorizer.fit_transform(domain_tfidf_doc)
        keywords_num = 200
        top_idx = X.toarray()[0].argsort()[-keywords_num:][::-1]
        key_words = np.array(vectorizer.get_feature_names())[top_idx]
        tfidf_value = X.toarray()[0][top_idx]

        with open('data/preprocessed/aspects/{}_keywords.txt'.format(domain_list[i]), 'w') as fw:
            for w, v in zip(key_words, tfidf_value):
                if w == 'pron' or w in domain_list[i]:
                    continue
                fw.write('{}:{:.4f}\t'.format(w, v))
