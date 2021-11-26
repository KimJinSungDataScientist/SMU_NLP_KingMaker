import numpy as np
import cv2
import sys
import os

from vec_text import load_pre_model, main
from drawing import draw
from img_vec import *


import tensorflow as tf
from tensorflow.keras import Model, layers, losses
import tensorflow_datasets as tfds
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import re

print(os.getcwd())

# colab_path = '/content/pr' # python 실행 경로
# drive_path = '/content/drive/MyDrive/nlp/pr' # 현재 파일 경로
drive_path = ''
colab_path = ''

d1_path = 'data_1'
d2_path = 'data_3'

# 주석
d1_path = os.path.join(colab_path,d1_path)
d2_path = os.path.join(colab_path,d2_path)
img_shape = (80, 80, 3)

# -------dataset--------


def gen(file_path):
    file_paths = file_path.decode('utf-8')
    lists_img_name = os.listdir(file_paths)
    for x in lists_img_name:
        img = cv2.imread(os.path.join(file_paths, x))
        img = img / 255.
        yield (img, img)


def test_gen(file_path):
    file_paths = file_path.decode('utf-8')
    lists_img_name = os.listdir(file_paths)
    for x in lists_img_name:
        img = cv2.imread(os.path.join(file_paths, x))
        img = img / 255.
        img = img.astype(np.float32)
        yield (img)


dataset_1 = tf.data.Dataset.from_generator(gen,
                                           (tf.dtypes.float64, tf.dtypes.float64),
                                           (img_shape, img_shape),
                                           args=(d1_path,))
dataset_2 = tf.data.Dataset.from_generator(gen,
                                           (tf.dtypes.float64, tf.dtypes.float64),
                                           (img_shape, img_shape),
                                           args=(d2_path,))
dataset = dataset_1.concatenate(dataset_2)
dataset = dataset.shuffle(128)
dataset = dataset.batch(8)

test_dataset_1 = tf.data.Dataset.from_generator(test_gen,
                                                (tf.dtypes.float32),
                                                (img_shape),
                                                args=(d1_path,))
test_dataset_2 = tf.data.Dataset.from_generator(test_gen,
                                                (tf.dtypes.float32),
                                                (img_shape),
                                                args=(d2_path,))
test_dataset = test_dataset_1.concatenate(test_dataset_2)
test_dataset = test_dataset.shuffle(128)
test_dataset = test_dataset.batch(8, drop_remainder=True)


def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence


def evaluate(model, tokenizer, sentence):
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    MAX_LENGTH = 40
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(model, tokenizer, sentence):
    prediction = evaluate(model, tokenizer, sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


def get_vec(path):
    img = cv2.imread(path)/255.
    img = img.reshape((-1, 80, 80, 3))
    vec = net.vec_extract(img).flatten()
    ls = ''
    for x in vec:
        ls += str(x)[2:5] + ' '
    return ls


# model train
net = U_net(img_shape)
net.compile()
net.train(dataset, epochs=50,)

dic = {}
files = os.listdir(d1_path)
for x in files:
    img = cv2.imread(os.path.join(d1_path, x)) / 255.
    img = img.reshape((-1, 80, 80, 3))
    vec = net.vec_extract(img).flatten()
    dic[x] = vec

df = pd.DataFrame(dic)
df.to_csv(os.path.join(colab_path,'csv/test.csv'))
df = pd.read_csv(os.path.join(colab_path,'csv/test.csv'))

model, tokenizer = main(df)
king_path = os.path.join(colab_path,'pr/kings_img')
img_name = os.listdir(king_path)
print(img_name)
dic = {}

for x in img_name:
  ls = get_vec(net, os.path.join(king_path,x))
  sentence = predict(model, tokenizer, ls)
  dic[x] = sentence

df = pd.DataFrame({'king':dic.keys(), 'description':dic.values()})

# model save
net.model.save(os.path.join(colab_path,'models/vec_extract'))
model.save_weights(os.path.join(colab_path,'models/transformer'))
tokenizer.save_to_file(os.path.join(colab_path, 'models/tokenizer'))
df.to_csv(os.path.join(colab_path,'csv/kings_corpus.csv'),index=None)