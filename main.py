import numpy
import os
import pathlib
from sentiment_analyser import train
from sentiment_analyser import evaluate

train_samples = [
    ('bom', 1),
    ('perfeito', 0),
    ('ruim', 0),
    ('melhorar', 0),
    ('demorar', 0)
]

validation_samples = [
    ('boa proposta', 1),
    ('não gostar', 0)
]

test_samples = [
    ('a empresa tem uma boa proposta, gostei', 1),
    ('a empresa é tudo de bom', 1),
    ('pode melhorar', 0),
    ('não é o que eu esperava', 0)
]


def execute(model_dir=None, train_dir=None, dev_dir=None,
            is_runtime=False,
            nr_hidden=64, max_length=100, # Shape
            dropout=0.5, learn_rate=0.001, # General NN config
            nb_epoch=5, batch_size=256, nr_examples=-1):  # Training params
    train_texts = [item[0] for item in train_samples];
    train_labels = [item[1] for item in train_samples];
    val_texts = [item[0] for item in validation_samples];
    val_labels = [item[1] for item in validation_samples];
    
    
    if model_dir is not None:
        if not os.path.exists(model_dir):
          os.makedirs(model_dir)
        model_dir = pathlib.Path(model_dir)
    if is_runtime:
        test_texts = [item[0] for item in test_samples];
        test_labels = [item[1] for item in test_samples];
        acc = evaluate(model_dir, test_texts, test_labels, max_length=max_length)
        print(acc)
    else:
        print("Training neural network...")
        train_labels = numpy.asarray(train_labels, dtype='int')
        val_labels = numpy.asarray(val_labels, dtype='int')
        lstm = train(train_texts, train_labels, val_texts, val_labels,
                     {'nr_hidden': nr_hidden, 'max_length': max_length, 'nr_class': 1},
                     {'dropout': dropout, 'lr': learn_rate},
                     {}, nb_epoch=nb_epoch, batch_size=batch_size)
        weights = lstm.get_weights()
        if model_dir is not None:
            with (model_dir / 'model').open('wb') as file_:
                pickle.dump(weights[1:], file_)
            with (model_dir / 'config.json').open('w') as file_:
                file_.write(lstm.to_json())


def train_network():
	execute(model_dir='binary_classification', is_runtime=False, nb_epoch=25, nr_hidden=256)


def evaluate_network():
	execute(model_dir='binary_classification', is_runtime=True)


if __name__ == '__main__':
	train_network()
	# evaluate_network()
