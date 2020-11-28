import numpy as np
import keras.backend as K
import sound_processing as sp
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.stats import mode


def plot_history(history , title='model accuracy / loss'):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title(title)
  plt.ylabel('accuracy / loss')
  plt.xlabel('epoch')
  plt.legend(['train_acc', 'test_acc','train_loss','test_loss'], loc='lower left')
  plt.ylim(ymax = 1, ymin = 0)
  plt.show()



def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def prepare_samples( feat , cont , posi , pnbs , shuffle=True ,extractor = lambda s : sp.features_extraction(s,22050,'stft') ):

  y_train = (cont == 'Ca')
  x_train = feat

  y_train_big = []
  x_train_big = []
  train_patientnb_big = []
  train_position_big = []

  for i, j, p,pos in zip(x_train, y_train, pnbs,posi):
    for subsample in i:
      y_train_big.append(j )
      #x_train_big.append(sp.features_extraction(subsample,22050,'stft') )
      x_train_big.append(extractor(subsample))
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


def Stack_pos(patients, features, controls, positions, patientnbs):
    x = []
    y = []
    for patient in np.unique(patients):
        indx = patientnbs == patient

        x_data, y_data, train_patientnb_big, train_position_big = prepare_samples(features[indx], controls[indx],
                                                                              positions[indx], patientnbs[indx],
                                                                              shuffle=False, extractor=lambda s:
        sp.features_extraction(s, 22050, 'stft'))

        shape = x_data[0].shape
        freq = mode(train_position_big)  # returns the position with the most crops and the number of crops
        for i in range(0, freq[1][0]):  # for each patient we make the x number of data points where x is the frequency of the most reoccurring position
            sample = np.empty(shape)  # initialize empty array
            for p in range(1, 9):  # iterate through positions
                pos = 'P' + str(p)
                if pos != freq[0][0]:  # if position is not the most reoccurring position
                    indx = np.argwhere(train_position_big == pos).flatten()  # return the indices of these positions
                    try:  # if position is present , stack
                        try:  # append available crops first
                            sample = np.dstack([sample, x_data[indx[i]]])
                        except:  # once we run out of them take any random one
                            sample = np.dstack([sample, x_data[
                            np.random.choice(indx)]])  # randomly take any available crops for this position and stack depthwise
                    except:  # if position not available fill the layer with zeros
                        sample = np.dstack([sample, np.zeros(shape)])
                else:  # if the position is the most reoccuring one
                    indx = np.argwhere(
                    train_position_big == 'P' + str(p)).flatten()  # return the indicies of the most reocurring body position
                    sample = np.dstack([sample, x_data[indx[i]]])  # stack that body pos
                    y.append(controls[indx][0] == 'Ca')  # append the label of the patient
            sample = np.delete(sample, 0, axis=2)  # delete the first row since it was used for initializing
            x.append(sample)  # append into train array
    x = np.array(x)
    y = to_categorical(np.array(y).astype(int))
    shuffle_indices = np.random.permutation(np.arange(y.shape[0]))

    return x[shuffle_indices], y[shuffle_indices]



