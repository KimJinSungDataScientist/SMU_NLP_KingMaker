import tensorflow as tf
import tensorflow_datasets as tfds
from vec_text import *
from img_vec import *
from text_sim import *
import pandas as pd
import cv2, os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# test image path
path = 'test_img/d.png'
tmp_path = ''

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def img_crop(img):
    face_cascade = cv2.CascadeClassifier(os.path.join(tmp_path,'face_detector.xml'))
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    x, y, w, h = faces[0]
    cropped_img = img[y: y + h, x: x + w]
    return cropped_img

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


def get_vec(net, path):
    try:
      img = img_crop(cv2.imread(path))
      img = cv2.resize(img,dsize=(80, 80))/255.
    except:
      print('Not found face')
    img = img.reshape((-1, 80, 80, 3))
    vec = net.vec_extract(img).flatten()
    ls = ''
    for x in vec:
        ls += str(x)[2:5] + ' '
    return ls

img_shape = (80,80,3)

test = tf.keras.models.load_model(os.path.join(tmp_path,'models/vec_extract'))
net = U_net(img_shape)
net.load_trained_model(test)


test_token = tfds.deprecated.text.SubwordTextEncoder.load_from_file(os.path.join(tmp_path,'models/tokenizer'))
test_model = load_pre_model(os.path.join(tmp_path,'models/transformer'),test_token)
df = pd.read_csv(os.path.join(tmp_path,'csv/kings_corpus.csv'))


ls = get_vec(net, os.path.join(tmp_path,path))
sentence = predict(test_model, test_token, ls)
ki = df.iloc[txt2simg(df, sentence)]['king']


fig = plt.figure(figsize=(20,80))
rows = 1
cols = 2

ax1 = fig.add_subplot(rows, cols, 1)
img = cv2.imread(os.path.join(tmp_path,'kings_img',ki))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax1.imshow(img)

ax2 = fig.add_subplot(rows, cols, 2)
img = cv2.imread(os.path.join(tmp_path,path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax2.imshow(img)