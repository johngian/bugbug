# -*- coding: utf-8 -*-
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from bugbug import bug_features
from bugbug.model import Model
from bugbug.nn import KerasClassifier, KerasTextToSequences


class TriageModel(Model):
    def __init__(self, *args, **kwargs):
        self.short_desc_maxlen = 20
        self.short_desc_vocab_size = 25000
        self.short_desc_emb_sz = 300
        self.long_desc_maxlen = 100
        self.long_desc_vocab_size = 25000
        self.long_desc_emb_sz = 300

        feature_extractors = [
            bug_features.bug_reporter(),
            bug_features.platform(),
            bug_features.op_sys()
        ]

        cleanup_functions = []

        self.extraction_pipeline = Pipeline([
            ('bug_extractor', bug_features.BugExtractor(feature_extractors, cleanup_functions)),
            ('union', ColumnTransformer([
                ('categorical', DictVectorizer(), 'data'),
                ('title_sequence', KerasTextToSequences(
                    self.short_desc_maxlen, self.short_desc_vocab_size), 'title'),
                ('first_comment_sequence', KerasTextToSequences(
                    self.long_desc_maxlen, self.long_desc_vocab_size), 'first_comment'),
                ('title_char_tfidf', TfidfVectorizer(
                    strip_accents='unicode',
                    analyzer='char',
                    stop_words='english',
                    ngram_range=(2, 4),
                    max_features=2500,
                    sublinear_tf=True
                ), 'title'),
                ('title_word_tfidf', TfidfVectorizer(
                    strip_accents='unicode',
                    min_df=0.0001,
                    max_df=0.1,
                    analyzer='word',
                    token_pattern=r'\w{1,}',
                    stop_words='english',
                    ngram_range=(2, 4),
                    max_features=30000,
                    sublinear_tf=True
                ), 'title')
            ])),
        ])


class TriageClassifier(KerasClassifier):
    def __init__(self, **kwargs):
        super(TriageClassifier, self).__init__(epochs=2, batch_size=256)
