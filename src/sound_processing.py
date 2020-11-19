import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

import glob




def get_feature_and_labels( path ):
    
    sound_path = glob.glob(path+'/*.wav')
    size = len(sound_path) # number of sounds
    
    features = np.zeros(size, dtype=object)
    diseases = np.zeros(size, dtype=object)
    positions = np.zeros(size, dtype=object)
    controls = np.zeros(size, dtype=object)
    frequences = np.zeros(size, dtype=object)
    
    for i in range(size):
        
        try : 
            split_path = sound_path[i].split('_')

            rate, data = wavfile.read(sound_path[i])

            features[i] = crop_sample( data , rate )
            diseases[i] = split_path[-4].split('\\')[2] # this may depend on your OS ( here : Ca31\\audio\\Pn )
            positions[i] = split_path[-1].split('.')[0] # remove .wav
            controls[i] = split_path[-2][:2] # Ca - Co 
            frequences[i] = rate
        except : 
            print("problem with -> ",sound_path[i])
        
    return features, diseases , positions , controls , frequences



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