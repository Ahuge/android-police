# Builtins
from cStringIO import StringIO
import os
import random

# Third-party libs
# Tensorflow? mxnet? keras? pytorch?
# Mxnet example
# https://mxnet.incubator.apache.org/tutorials/unsupervised_learning/gan.html
import tensorflow

# Custom libs
import classifier
import generator


DATA_FOLDER = os.path.join(
    os.path.dirname(__file__), "data"
)
OUTPUT_TEST_WAV = os.path.join(
    os.path.dirname(__file__), ("test_wav_%d.wav" % random.randint(0, 1000000)),
)
BATCH_SIZE = 100
EPOCH_COUNT = 100


def train_classifier(txts, classif=None):
    """
    For each data in txts
         train_classifier that it is real.

    :param txts: List of string paths to wav txts
    :type txts: list[str]
    :param classif: Classifier or None
    :type classif: classifier.Classifier|None
    :return: Classifier that has been trained
    :rtype: classifier.Classifier
    """


def generate_wav(gen):
    """
    Generate a wav from the generator.Generator

    :param gen: Generator that we want to use to generate.
    :type gen: generator.Generator
    :return: A StringIO with the wav bytes written into it.
    :rtype: StringIO
    """


def train_generator(times, classif, gen=None):
    """
    For time in range(times):
        generate new wav
        classify chance it's real.
        if classify chance > average_classifier_percent:
            reward generator
        else:
            don't reward generator

    :param times: Number of mp3s to generate
    :type times: int
    :param classif: Trained classifier
    :type classif: classifier.Classifier
    :param gen: Generator or None
    :type gen: generator.Generator|None
    :return: Generator that has been trained.
    :rtype: generator.Generator
    """


def randomize_data_folder(data_folder, count=1000):
    """
    Pick a random <count> datas from data_folder and return them
    :param data_folder: path to wav files transcribed to txt
    :type data_folder: str
    :param count: number of results we want back max.
    :type count: int
    :return: List of paths to txt files
    :rtype: list[str]
    """


def run():
    """
    run will train our GAN and then write one result out to disk afterwards.

    :return: String path to the written file.
    :rtype: str
    """
    sess = tensorflow.Session()
    classif = None
    gen = None
    for epoch in range(EPOCH_COUNT):
        paths = randomize_data_folder(DATA_FOLDER, count=BATCH_SIZE)
        classif = train_classifier(paths, classif)
        gen = train_generator(BATCH_SIZE, classif, gen)

    data = generate_wav(gen)
    with open(OUTPUT_TEST_WAV, "wb") as fh:
        for byte_ in data.readlines():
            fh.write(byte_.rstrip("\n"))
    return OUTPUT_TEST_WAV
