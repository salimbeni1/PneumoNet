import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import keras.backend as K



import glob




def get_feature_and_labels( path ):
    
    sound_path = glob.glob(path+'/*.wav')
    size = len(sound_path) # number of sounds
    
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


def normalize(X, ty):
    '''
    Normalization of input features 
    
    Args:
        X (2d numpy array): matrix to be normalized
        ty (char): type of normalization. It can be "mean", "minmax" or "std"
        
    
     Returns   
         (2d numpy array) : normalized matrix        
    '''
    if ty == 'mean':
        return X - np.mean(X)
    if ty == 'minmax':
        return (X-np.min(X))/(np.max(X)-np.min(X))
    if ty == 'std':
        return (X-np.mean(X))/np.std(X)
    #if ty == 'whiten':
    #    X = X.reshape((-1, np.prod(X.shape[1:])))
    #    X_centered = X - np.mean(X, axis=0)
    #    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    #    U, Lambda, _ = np.linalg.svd(Sigma)
    #    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
    #    return np.dot(X_centered, W.T)

def _filter_banks(sample, rate):
    '''
    Compute filter banks 
    
    Args:
        sample (numpy array): original audio data
        rate (int): sampling frequency
        
    
     Returns   
         (2d numpy array) : filter banks of the audio track       
    '''
    #pre-emphasis
    emphasized_sample = np.append(sample[0], sample[1:] - 0.95 * sample[:-1])

    #framing
    frame_length, frame_step = 0.025 * rate, 0.01 * rate  # Convert from seconds to samples
    signal_length = len(emphasized_sample)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_sample, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    #hamming window
    frames *= np.hamming(frame_length)

    #fourier transform and power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, 512))  # Magnitude of the FFT
    pow_frames = ((1.0 / 512) * ((mag_frames) ** 2))  # Power Spectrum

    #filter banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, 40 + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((512 + 1) * hz_points / rate)

    fbank = np.zeros((40, int(np.floor(512 / 2 + 1))))
    for m in range(1, 40 + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    filter_banks -= (np.mean(filter_banks, axis=0))
    
    return filter_banks

def features_extraction(sample, rate, ft):
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
        stft_db = librosa.power_to_db(abs(stft)**2)[:150,:]
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
    
def audio_feature_augm(sample,rate):
    '''
    Data augmentation by changing loudness, pitch, adding noise or shift the audio wave 
    
    Args:
        sample (numpy array): original audio data
        rate (int): sampling frequency
        
        
    
     Returns   
         (four 2d numpy arrays) : modified audio waves    
    '''
    loud_aug = naa.LoudnessAug(factor=(0.5, 2))
    noise_aug = naa.NoiseAug()
    pitch_aug = naa.PitchAug(sampling_rate=rate, factor=(0.75,1.25))
    shift_aug = naa.ShiftAug(rate)
    s = shift_aug.augment(sample, 1)
    n = noise_aug.augment(sample, 1)
    l = loud_aug.augment(sample, 1)
    p = pitch_aug.augment(sample, 1)
    return s,n,l,p

def display_all(sample, rate):
    '''
    Visualize all the features that we can extract 
    
    Args:
        sample (numpy array): original audio data
        rate (int): sampling frequency
           
    '''
    #plot audio wave
    f, ax = plt.subplots()
    f.suptitle('Audio wave')

    librosa.display.waveplot(sample, sr=rate, x_axis = 's')
    ax.set_ylabel('Amplitude')
    
    #plot stft
    stft_db = features_extraction(sample,rate,'stft')
    #stft_db = normalize(stft_db, 'minmax')
    f, axs = plt.subplots(1,2,figsize=(14,5))
    f.suptitle('Short time Fuorier transform spectrogram')
    axs[0].imshow(stft_db)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Freq')
     
    librosa.display.specshow(stft_db, sr=rate, x_axis='time',y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    #plot mfcc
    mfcc = features_extraction(sample,rate,'mfcc')
    f, ax = plt.subplots()
    f.suptitle('MFCC')
    librosa.display.specshow(mfcc, sr=rate, x_axis='time')

    #plot mel spectrograms
    mel = features_extraction(sample,rate,'mel_spect')
    f, axs = plt.subplots(1,2,figsize=(14,5))
    f.suptitle('Mel spectrogram')
    axs[0].imshow(mel)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Freq')
     
    librosa.display.specshow(mel, sr=rate, x_axis='time',y_axis='log')
    plt.colorbar(format='%+2.0f dB')

    #mfcc = normalize(mfcc, 'minmax')



def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val