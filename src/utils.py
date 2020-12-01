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
  plt.savefig('graph.png')
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


def stack_pos(patients, features, controls, positions, patientnbs):
    x = []
    y = []
    patient_id = []

    uniq_id = np.unique(np.dstack((controls.astype('|S4'), patientnbs.astype('|S4'))), axis=1)[0, :, :]
    for (caco, pnb) in uniq_id:
        indx_initial = (controls.astype('|S4') == caco) & (patientnbs.astype('|S4') == pnb)

        x_data, y_data, train_patientnb_big, train_position_big = prepare_samples(features[indx_initial],
                                                                                  controls[indx_initial],
                                                                                  positions[indx_initial],
                                                                                  patientnbs[indx_initial],
                                                                                  shuffle=False, extractor=lambda s:
            sp.features_extraction(s, 22050, 'stft'))


        shape = x_data[0].shape
        freq = mode(train_position_big)  # returns the position with the most crops and the number of crops
        for i in range(0, freq[1][0]):
            # for each patient we make the x number of data points where x is the frequency of the most reoccurring position
            sample = np.empty(shape)  # initialize empty array
            for p in range(1, 9):  # iterate through positions
                pos = 'P' + str(p)
                if pos != freq[0][0]:  # if position is not the most reoccurring position
                    indx = np.argwhere(train_position_big == pos).flatten()  # return the indices of these positions
                    try:  # if position is present , stack
                        try:  # append available crops first
                            sample = np.dstack([sample, x_data[indx[i]]])
                        except:  # once we run out of them take any random one
                            sample = np.dstack([sample, x_data[np.random.choice(indx)]])
                            # randomly take any available crops for this position and stack depthwise
                    except:  # if position not available fill the layer with zeros
                        sample = np.dstack([sample, np.zeros(shape)])
                else:  # if the position is the most reoccuring one
                    indx = np.argwhere(
                        train_position_big == 'P' + str(p))  # return the indicies of the most reocurring body position

                    indx = indx.flatten()
                    patient_id.append(caco.decode('UTF-8') + pnb.decode('UTF-8'))   # list of unique pat id
                    sample = np.dstack([sample, x_data[indx[i]]])  # stack that body pos
                    y.append(controls[indx_initial][0] == 'Ca')  # append the label of the patient
            sample = np.delete(sample, 0, axis=2)  # delete the first row since it was used for initializing
            x.append(sample)  # append into train array

    patient_id = np.array(patient_id)
    x = np.array(x)
    y = to_categorical(np.array(y).astype(int))
    shuffle_indices = np.random.permutation(np.arange(y.shape[0]))

    return x[shuffle_indices], y[shuffle_indices], patient_id


def load_from_npz( z_compressed_path ):
  print("Importing from: "+z_compressed_path)
  # export from z_compressed file
  dict_data = np.load(z_compressed_path,allow_pickle=True)

  feat, deas = dict_data['arr_0'] , dict_data['arr_1']
  pos, cont =  dict_data['arr_2'] , dict_data['arr_3']
  freq, pat = dict_data['arr_4'] , dict_data['arr_5']

  return feat, deas, pos, cont, freq, pat


def feature_loader( z_compressed_path , raw_data_path ):

  is_in_memory = glob.glob(z_compressed_path)

  if ( len(is_in_memory)==0 ):
    print("Importing RAW from: "+raw_data_path , flush=True)

    # export from raw data
    feat, deas, pos, cont, freq, pat = sp.get_feature_and_labels( raw_data_path ) # both control and case
    # save a z_compressed file
    print("\nExporting .npz at: "+z_compressed_path, flush=True)
    np.savez( z_compressed_path , feat, deas, pos, cont, freq, pat )
    return feat, deas, pos, cont, freq, pat
  
  else :
    print("Importing from: "+z_compressed_path)
    # export from z_compressed file
    dict_data = np.load(z_compressed_path,allow_pickle=True)

    feat, deas = dict_data['arr_0'] , dict_data['arr_1']
    pos, cont =  dict_data['arr_2'] , dict_data['arr_3']
    freq, pat = dict_data['arr_4'] , dict_data['arr_5']

    return feat, deas, pos, cont, freq, pat



