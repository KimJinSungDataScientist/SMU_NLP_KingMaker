import os,sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def draw(model,d1_path):
    data = os.listdir(d1_path)
    # np.random.randint
    fig = plt.figure(figsize=(10,40))
    rows = 10
    cols = 3
    im = np.random.choice(data,5)

    idx = 1
    for x in im:
      img = cv2.imread(os.path.join(d1_path,x))
      test = img.reshape((1,80,80,3))/255.
      out = model(test).numpy()
      test = model.vec_extract(test)

      k = test.reshape((13,13))
      out = out.reshape((80,80,3))
      out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      ax1 = fig.add_subplot(rows, cols, idx)
      ax1.imshow(img)
      ax1.set_title('original')
      ax1.axis("off")
      idx +=1

      ax2 = fig.add_subplot(rows, cols, idx)
      ax2.imshow(k,cmap='gray')
      ax2.set_title('feature map')
      ax2.axis("off")
      idx += 1

      ax3 = fig.add_subplot(rows, cols, idx)
      ax3.imshow(out)
      ax3.set_title('output')
      ax3.axis("off")
      idx += 1


    plt.show()
