import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm

import nlpaug
import nlpaug.augmenter.audio as naa

import glob



def get_feature_and_labels( path ):
    '''
    crop raw audios from path
    
    Args:
        path (string): path of the folder
    
     Returns   
         (features, diseases , positions , controls , frequences , patientnbs) : features of the patients crops in the given path       
    '''
    sound_path = glob.glob(path+'/*.wav')
    size = len(sound_path) # number of sounds
    print('parsing '+str(size)+' audio files',flush=True)
    
    features = np.zeros(size, dtype=object)
    diseases = np.zeros(size, dtype=object)
    positions = np.zeros(size, dtype=object)
    controls = np.zeros(size, dtype=object)
    frequences = np.zeros(size, dtype=object)
    patientnbs = np.zeros(size, dtype=object) 
    
    too_short_signal = []
    
    for i in tqdm(range(size)): 
        try : 
            split_path = sound_path[i].split('_')
            
            data, rate = librosa.load(sound_path[i])
            #rate, data = wavfile.read(sound_path[i])
            
            features[i] = crop_sample( data , rate )
            
            try : 
                diseases[i] = split_path[-4].split('\\')[2] # this may depend on your OS ( here : Ca31\\audio\\Pn)
            except :
                try :
                    diseases[i] = split_path[-4].split('/')[2]
                except :
                    diseases[i] = 'XXX'
                    raise Exception('unable to split ur path synthax')
                
            positions[i] = split_path[-1].split('.')[0] # remove .wav
            controls[i] = split_path[-2][:2] # Ca - Co
            patientnbs[i] = split_path[-2][2:]
            frequences[i] = rate
            
            if(features[i].shape[0] == 0):
                too_short_signal.append(i)
            
        except Exception as e:
            print("problemm with -> ",sound_path[i])
            print("[Error] ",e)
            
    
    # remove too short samples
    features = np.delete(features, too_short_signal)
    diseases = np.delete(diseases, too_short_signal)
    positions = np.delete(positions, too_short_signal)
    controls = np.delete(controls, too_short_signal)
    frequences = np.delete(frequences, too_short_signal)
    patientnbs = np.delete(patientnbs, too_short_signal)
        
    return features, diseases , positions , controls , frequences , patientnbs





def crop_sample( samples, frequency, size_crop=5, step_crop=2.5):
    '''
    crop samples combining left and right overlays 
    
    Args:
        samples (numpy array): data to be croped
        frequency (int): rate of the sample
        size_crop (float): lengh if crop in seconds
        step_crop (float): lengh og overlay in seconds
        
    
     Returns   
         (N, size_crop*frequency) array : N sub samples of size_crop seconds        
    '''
    
    size = round(size_crop*frequency) # size of the crop in seconds
    step = round(step_crop*frequency) # overlay in seconds

    croped_samples_right = [samples[i : i + size] for i in range(0, len(samples), step)]
    croped_samples_right = [i for i in croped_samples_right if len(i) == size]

    croped_samples_left = [samples[ len(samples)-1-(i+size) :  len(samples)-1-i] for i in range(0, len(samples), step)]
    croped_samples_left = [i for i in croped_samples_left if len(i) == size]

    croped_samples = np.array(croped_samples_right + croped_samples_left)
    return croped_samples


def features_extraction(sample, rate, ft , high_limit=0 , lower_limit = 150):
    '''
    Extract interesting features from audio file 
    
    Args:
        sample (numpy array): original audio data
        rate (int): sampling frequency
        ft (char): type of feature to extract. It can be 'stft','mel_spect','mfcc'
        
    
     Returns   
         (2d numpy array) : feature      
    '''
    if ft == 'stft':
        stft = librosa.core.stft(y=sample)
        stft_db = librosa.power_to_db(abs(stft)**2)[high_limit:lower_limit,:]
        return stft_db
    
    if ft == 'mel_spect':
        mel = librosa.feature.melspectrogram(sample, sr=rate)
        mel_db = librosa.power_to_db(mel, ref=np.max)[:50,:]
        return mel_db
    
    if ft == 'mfcc':
        mfccs = librosa.feature.mfcc(sample, sr=rate, n_mfcc=32)
        return mfccs
    
    #if ft == 'filter_banks':
    #    filter_banks = _filter_banks(sample,rate)
    #    return filter_banks
    

def augmented(features,positions, controls, nbs):
  '''
    Data augmentation by changing loudness, pitch, adding noise to the audio wave 
    
    Args:
        features (numpy array): audio crops
        positions (numpy array): patient position
        controls (numpy array): label
        nbs (numpy array): patient number
          
     Returns   
         (four 2d numpy arrays) : augmented dataset    
  '''
  size = len(features)

  noise = np.zeros(size, dtype=object)
  loud  = np.zeros(size, dtype=object)
  pitch = np.zeros(size, dtype=object)
  
  #noise, loudness, pitch changing
  noise_aug = naa.NoiseAug()
  loud_aug = naa.LoudnessAug(factor=(0.5, 2))
  pitch_aug = naa.PitchAug(sampling_rate=22050, factor=(0.75,1.25))

  n_noise = []; n_pitch = []; n_loud = []
  for i, feat in enumerate(features):
    noise[i] = np.array(noise_aug.augment([x for x in feat], 1))
    pitch[i] = np.array(pitch_aug.augment([x for x in feat], 1))
    loud[i] = np.array(loud_aug.augment([x for x in feat], 1))
    n_noise.append(str(nbs[i])+'b')
    n_pitch.append(str(nbs[i])+'c')
    n_loud.append(str(nbs[i])+'d')
    
  
  features_aug = np.hstack((features, noise, pitch, loud))
  positions_aug = np.hstack(([positions]*4))
  controls_aug = np.hstack(([controls]*4))
  #nbs_aug = np.hstack(([nbs]*4))
  nbs_aug = np.hstack((nbs, np.array(n_noise),np.array(n_pitch),np.array(n_loud)))


  return features_aug, positions_aug, controls_aug, nbs_aug



# BASIC UTILS 

def show_spec_from_sample(sample , frequency):
    f, t, Sxx = signal.spectrogram(sample, frequency)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
def get_spect(path):
    sample_rate, samples = wavfile.read(path)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    return spectrogram
