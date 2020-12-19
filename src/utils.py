import numpy as np
import keras.backend as K
import sound_processing as sp
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from scipy.stats import mode


import pandas as pd
import os
import random
from sklearn.utils import shuffle




def prepare_samples( feat , cont , posi , pnbs , shuffle=True ,extractor = lambda s : sp.features_extraction(s,22050,'stft') ):
  """Intake lists of data an prepares them into arrays of 5 second spectrograms and labels"""
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
      x_train_big.append(extractor(subsample))  # appends the spectrogram
      train_patientnb_big.append(p)  # appends the patient number
      train_position_big.append(pos)    # appends the position number of the crop

  y_train = np.array(y_train_big)
  x_train = np.array(x_train_big)
  train_patientnb_big = np.array(train_patientnb_big)
  train_position_big = np.array(train_position_big)

  y_train = to_categorical(y_train.astype(int))  # convert the label to categorical label
  x_train = np.array(x_train)[:,:,:,np.newaxis].astype('float32')  # not sure why I do this but let's keep it for now

  if(shuffle):
    print('shuffleling')
    shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))
    y_train = y_train[shuffle_indices]  # rearranges the y_train based on the shuffled indices
    x_train = x_train[shuffle_indices]  # rearranges the x_train based on the shuffled indices
    train_patientnb_big = train_patientnb_big[shuffle_indices]
    train_position_big = train_position_big[shuffle_indices]

  return x_train, y_train, train_patientnb_big, train_position_big


def stack_pos(patients, features, controls, positions, patientnbs, high_limit=0, lower_limit = 150):
    """Gets the lists of data and returns the array of spectrograms prepared for the model by patient, and label
    array y"""
    x = []
    y = []
    patient_id = []

    uniq_id = np.unique(np.dstack((controls.astype('|S4'), patientnbs.astype('|S4'))), axis=1)[0, :, :]
    # creates patients unique identifier
    for (caco, pnb) in uniq_id:
        indx_initial = (controls.astype('|S4') == caco) & (patientnbs.astype('|S4') == pnb)

        x_data, y_data, train_patientnb_big, train_position_big = prepare_samples(features[indx_initial],
                                                                                  controls[indx_initial],
                                                                                  positions[indx_initial],
                                                                                  patientnbs[indx_initial],
                                                                                  shuffle=False, extractor=lambda s:
            sp.features_extraction(s, 22050, 'stft', high_limit=high_limit, lower_limit = lower_limit))


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

    return x[shuffle_indices], y[shuffle_indices], patient_id[shuffle_indices]



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






def get_GVA():
  """Loads the data from compressed files, which is separate for controls and cases and returns lists for train and
  test separately"""
  # Train -------------------------------------
  feat_GVA_Ca, _, posi_GVA_Ca, ctrl_GVA_Ca, _, nbpa_GVA_Ca = load_from_npz( '../pneumoscope/npz_files/GVA/GVA_Ca_train_b1.npz' )
  feat_GVA_Co, _, posi_GVA_Co, ctrl_GVA_Co, _, nbpa_GVA_Co = load_from_npz( '../pneumoscope/npz_files/GVA/GVA_Co_train_b1.npz' )
  # Merge train batches -----------------------
  features   = np.append(feat_GVA_Ca, feat_GVA_Co)  # audio data
  positions  = np.append(posi_GVA_Ca, posi_GVA_Co)  # location number where the data was collected
  controls   = np.append(ctrl_GVA_Ca, ctrl_GVA_Co)  # label whether control or case
  patientnbs = np.append(nbpa_GVA_Ca, nbpa_GVA_Co)  # patients number

  # Test ---------------------------------------
  feat_GVA_test, _, posi_GVA_test, ctrl_GVA_test, _, nbpa_GVA_test = load_from_npz( '../pneumoscope/npz_files/GVA/GVA_Ca_Co_test.npz'  )
  featuresT   = feat_GVA_test
  positionsT  = posi_GVA_test
  controlsT   = ctrl_GVA_test
  patientnbsT = nbpa_GVA_test

  return (features, positions, controls, patientnbs), (featuresT, positionsT, controlsT, patientnbsT)


# @title Get POA
def get_POA(shrink = True, batch=1):

  #Train -------------------------------------
  feat_POA_Ca, _, posi_POA_Ca, ctrl_POA_Ca, _, nbpa_POA_Ca = load_from_npz( '../pneumoscope/npz_files/POA/POA_Ca_train_b' + str(batch) + '.npz' )
  try:
    feat_POA_Co, _, posi_POA_Co, ctrl_POA_Co, _, nbpa_POA_Co = load_from_npz( '../pneumoscope/npz_files/POA/POA_Co_train_b' + str(batch) + '.npz' )
  except:
    #feat_POA_Co, _, posi_POA_Co, ctrl_POA_Co, _, nbpa_POA_Co = load_from_npz( '../pneumoscope/npz_files/POA/POA_Ca_train_b' + str(batch) + '.npz' )
    feat_POA_Co = []
    posi_POA_Co = []
    nbpa_POA_Co = []
    ctrl_POA_Co = []


  # Merge train batches -----------------------
  features   = np.append(feat_POA_Ca, feat_POA_Co)
  positions  = np.append(posi_POA_Ca, posi_POA_Co)
  controls   = np.append(ctrl_POA_Ca, ctrl_POA_Co)
  patientnbs = np.append(nbpa_POA_Ca, nbpa_POA_Co)

  # Test ---------------------------------------
  feat_POA_test, _, posi_POA_test, ctrl_POA_test, _, nbpa_POA_test = load_from_npz( '../pneumoscope/npz_files/POA/POA_Ca_Co_test.npz'  )
  featuresT   = feat_POA_test
  positionsT  = posi_POA_test
  controlsT   = ctrl_POA_test
  patientnbsT = nbpa_POA_test

  # Shrink too big features --------------------
  if (shrink):
    for i in range(featuresT.shape[0]):
      size = featuresT[i].shape[0]
      if (size > 20):
        idx = np.random.randint(size, size=15)
        featuresT[i] = featuresT[i][idx,:]
        #print(featuresT[i].shape)

    for i in range(features.shape[0]):
      size = features[i].shape[0]
      if (size > 20):
        idx = np.random.randint(size, size=15)
        features[i] = features[i][idx,:]

  return  (features, positions, controls, patientnbs), (featuresT, positionsT, controlsT, patientnbsT)



def get_arrays_GVA():
    """Returns the data as arrays of spectrograms of x and labels y"""
    print('Importing Data')
    (features, positions, controls, patientnbs), (featuresT, positionsT, controlsT, patientnbsT) = get_GVA()

    print('Creating Training Data')
    x_train, y_train, _ = stack_pos([], features, controls, positions, patientnbs)
    print(x_train.shape)

    print('Creating Test Data')
    x_test, y_test, _ = stack_pos([], featuresT, controlsT, positionsT, patientnbsT)
    print(x_test.shape)

    return x_train, y_train, x_test, y_test

# @title Get Arrays POA
def get_arrays_POA(batch=1):
  print('Importing Data')
  (features, positions, controls, patientnbs), (featuresT, positionsT, controlsT, patientnbsT) = get_POA(batch=batch)

  print('Creating Training Data')
  x_train, y_train, _ = stack_pos([], features, controls, positions, patientnbs)
  print(x_train.shape)

  print('Creating Test Data')
  x_test, y_test, _ = stack_pos([], featuresT, controlsT, positionsT, patientnbsT)
  print(x_test.shape)

  return x_train, y_train, x_test, y_test



def confusion(y_pred, y_exp):
  """Prints the confusion matrix"""
  y_pred = y_pred*2 - 1
  conf = list(y_pred + y_exp)
  TP = conf.count(2)
  TN = conf.count(-1)
  FP = conf.count(1)
  FN = conf.count(0)
  print('TP: {}, TN: {}, FP: {}, FN: {}'.format(TP, TN, FP, FN))



def pneu_bronch_sets():
  #1: bacterial pneumonia and 2: viral pneumonia and 3: bronchiolitis
  clinical = pd.read_csv('../data/Pn_POA_clinical_database.csv')
  diagnosis = np.array(clinical['diagnosis'].tolist())
  patients = np.array(clinical['patient'].tolist())
  pn_bronch = diagnosis[(diagnosis == '1' )| (diagnosis == '3') | (diagnosis == '2')]
  pn_bronch_patients = patients[(diagnosis == '1' )| (diagnosis == '3') | (diagnosis == '2')]
  print('Bronchiolitis: '+ str(np.sum(pn_bronch == '3')))
  print('Pneumonia: ' + str(np.sum(pn_bronch == '1') + np.sum(pn_bronch == '2')))

  bronch = pn_bronch_patients[pn_bronch == '3']
  pneu = pn_bronch_patients[(pn_bronch == '1') | (pn_bronch == '2')]


  train1 = np.array(os.listdir('../data/train/Pn_POA_Cases/batch_1'))
  train2 = np.array(os.listdir('../data/train/Pn_POA_Cases/batch_2'))
  train3 = np.array(os.listdir('../data/train/Pn_POA_Cases/batch_3'))
  train4 = np.array(os.listdir('../data/train/Pn_POA_Cases/batch_4'))
  train5 = np.array(os.listdir('../data/test/POA_Ca_Co'))

  train = np.hstack((train1, train2, train3, train4, train5))

  keep_train = np.array([p for p in pn_bronch_patients if p in train])
  bronch_all = np.array([b for b in bronch if b in keep_train])
  pneu_all = np.array([pn for pn in pneu if pn in keep_train])

  print('Undersampling of bronchiolite')
  
  keep = np.array([83, 12, 26, 78, 96, 32, 82, 84,  5, 44, 97, 33, 86, 22, 71, 92, 91,  1, 52, 54, 85,  0, 57, 88, 7, 64, 73, 87, 17, 70])

  bronch_all = bronch_all[keep]
  bronch_train = bronch_all[:26]
  bronch_test = bronch_all[26:]
  pneu_train = pneu_all[:26]
  pneu_test = pneu_all[26:]


  keep_train = np.hstack((bronch_train, pneu_train))
  keep_test = np.hstack((bronch_test, pneu_test))
  print('Train: ' + str(len(keep_train)))
  print('Test: ' + str(len(keep_test)))

  sets = ['../data/train/Pn_POA_Cases/*/', '../data/test/POA_Ca_Co/']
  features = []; positions = []; patientnbs = []
  featuresT = []; positionsT = []; patientnbsT = []
  
  for i,p in enumerate(keep_train):
    feature, _, position, _, _, patientnb = sp.get_feature_and_labels(sets[1]+str(p)+'/*' if p in train5 else sets[0]+str(p)+'/*' ) # both control and case
    features = np.hstack((features, feature))
    positions = np.hstack((positions, position))
    patientnbs = np.hstack((patientnbs, patientnb))

  for i,p in enumerate(keep_test):
    featureT, _, positionT, _, _, patientnbT = sp.get_feature_and_labels(sets[1]+str(p)+'/*' if p in train5 else sets[0]+str(p)+'/*' ) # both control and case
    featuresT = np.hstack((featuresT, featureT))
    positionsT = np.hstack((positionsT, positionT))
    patientnbsT = np.hstack((patientnbsT, patientnbT))

  controls = []; controlsT = []
  for nb in patientnbs:
    control = 'Ca' if 'Pn_POA_Ca'+str(nb) in bronch_train else 'Co'
    controls = np.hstack((controls, control))
  for nb in patientnbsT:
    controlT = 'Ca' if 'Pn_POA_Ca'+str(nb) in bronch_test else 'Co'
    controlsT = np.hstack((controlsT, controlT))

  features, controls, positions, patientnbs = shuffle(features, controls, positions, patientnbs)
  featuresT, controlsT, positionsT, patientnbsT = shuffle(featuresT, controlsT, positionsT, patientnbsT)

  return (features, controls, positions, patientnbs), (featuresT, controlsT, positionsT, patientnbsT)

  



# BASIC UTILS


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
    """F1 metric callback for the model"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def load_from_npz(z_compressed_path ):
  """Loads the data from compressed file and returns the lists of data"""
  print("Importing from: "+z_compressed_path)
  # export from z_compressed file
  dict_data = np.load(z_compressed_path,allow_pickle=True)

  feat, deas = dict_data['arr_0'] , dict_data['arr_1']
  pos, cont =  dict_data['arr_2'] , dict_data['arr_3']
  freq, pat = dict_data['arr_4'] , dict_data['arr_5']

  return feat, deas, pos, cont, freq, pat