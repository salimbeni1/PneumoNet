import numpy as np
import keras.backend as K
import sound_processing as sp
from keras.utils import to_categorical


def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def prepare_samples( feat , cont , posi , pnbs , shuffle=True ):

  y_train = (cont == 'Ca')
  x_train = feat

  y_train_big = []
  x_train_big = []
  train_patientnb_big = []
  train_position_big = []

  for i, j, p,pos in zip(x_train, y_train, pnbs,posi):
    for subsample in i:
      y_train_big.append(j )
      x_train_big.append(sp.features_extraction(subsample,22050,'stft') )
      train_patientnb_big.append(p)
      train_position_big.append(pos)

  y_train = np.array(y_train_big)
  x_train = np.array(x_train_big)
  train_patientnb_big = np.array(train_patientnb_big)
  train_position_big = np.array(train_position_big)

  y_train = to_categorical(y_train.astype(int))
  x_train = np.array(x_train)[:,:,:,np.newaxis].astype('float32') # not sure why I do this but let's keep it for now

  if(shuffle):
    print('shuffleling')
    shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))
    y_train = y_train[shuffle_indices]  # rearranges the y_train based on the shuffled indices
    x_train = x_train[shuffle_indices]  # rearranges the x_train based on the shuffled indices
    train_patientnb_big = train_patientnb_big[shuffle_indices]
    train_position_big = train_position_big[shuffle_indices]

  return x_train, y_train, train_patientnb_big, train_position_big