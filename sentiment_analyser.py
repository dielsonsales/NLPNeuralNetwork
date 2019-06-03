import cytoolz
import numpy
import spacy
from keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from keras.layers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from spacy.compat import pickle


pt_br_model = 'pt_wikipedia_md'


class SentimentAnalyser(object):
    @classmethod
    def load(cls, path, nlp, max_length=100):
        """
        Loads the language model file.
        """
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    def __init__(self, model, max_length=100):
        self._model = model
        self.max_length = max_length

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for doc in docs:
            Xs = get_features([doc], self.max_length)
            ys = self._model.predict(Xs)
            for doc, label in zip([doc], ys):
                doc.sentiment = label
            yield doc

    def set_sentiment(self, doc, y):
        doc.sentiment = float(y[0])


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        sentences.append(doc)
        labels.append(y)
    return sentences, numpy.asarray(labels, dtype='int32')


def get_features(docs: list, max_length):
    Xs = numpy.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs


def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings]
        )
    )
    model.add(TimeDistributed(Dense(shape['nr_hidden'], use_bias=False)))
    model.add(Bidirectional(LSTM(shape['nr_hidden'], recurrent_dropout=settings['dropout'], dropout=settings['dropout'])))
    #model.add(Dropout(settings['dropout']))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, lstm_optimizer, batch_size=100,
          nb_epoch=5, by_sentence=True):
    print("Loading spaCy")
    nlp = spacy.load(pt_br_model)
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))
    print("Starting get_labelled_sentences()")
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_shape['max_length'])
    dev_X = get_features(dev_docs, lstm_shape['max_length'])
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              epochs=nb_epoch, batch_size=batch_size)
    print("Model ready")
    return model


def evaluate(model_dir, texts, labels, max_length=100):
    nlp = spacy.load(pt_br_model)
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    nlp.add_pipe(SentimentAnalyser.load(model_dir, nlp, max_length=max_length))
    correct = 0
    i = 0
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        if doc.sentiment >= 0.5 and bool(labels[i]) is True:
            print(texts[i] + " correctly classified as True " + str(doc.sentiment))
        elif doc.sentiment >= 0.5 and bool(labels[i]) is False:
            print(texts[i] + " wrongly classified as True " + str(doc.sentiment))
        elif doc.sentiment < 0.5 and bool(labels[i]) is True:
            print(texts[i] + " wrongly classified as False " + str(doc.sentiment))
        elif doc.sentiment < 0.5 and bool(labels[i]) is False:
            print(texts[i] + " correctly classified as False " + str(doc.sentiment))
        correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
        i += 1
    return float(correct) / i


