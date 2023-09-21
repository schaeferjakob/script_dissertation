#%% 
import os
import numpy as np
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
import functions.definitions as definitions
from scipy.signal import hilbert
from scipy import stats
from scipy.signal import get_window
from tqdm.notebook import tqdm  
from sklearn.metrics import r2_score
from scipy import ndimage

#%%
start_directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/sorted_elastography'

transducer = 'L15' # choose C3_large, L15_large or combined
tgc = 'no_tgc'
ref_parameter = 'CAP' # choose CAP or Median_E

if transducer == 'L15':
    min_depth = 687
    max_depth = 2704
    start_scanline = 0
    number_of_scanlines = 192
    window_size = 100
    hop_size = 1
if transducer == 'C3':
    min_depth = 196
    max_depth = 2320
    start_scanline = 57
    number_of_scanlines = 94
    window_size = 100
    hop_size = 3

end_scanline = start_scanline + number_of_scanlines

if transducer == 'combined1':
    window_size = 50
    hop_size = 2

# for spectral shift
spectral_shift_border = 5 # in dB
#%%
# attenuation slope
pearson_per_frequency = []
slopes_all_recordings = []
recording_and_slope = {}
for dir in tqdm(os.listdir(start_directory)):
    # for loop through all recordings:
    for file in os.listdir(start_directory+'/'+dir):
        if transducer in file and tgc in file:
            # get array without tgc from this recording
            array = np.load(start_directory+'/'+dir+'/'+file)

            # skip arrays with less pixels than max_depth
            array_too_short = False
            if len(array[0,:,0]) < max_depth:
                array_too_short = True
                break
                
            # apply min- and max_depth
            array = array[:,min_depth:max_depth,:]

            #remove the sides from array 
            array = array[start_scanline:end_scanline,:,:]

            # apply log to data (to perform linear regression)
            array = np.log(np.abs(array)+1e-10)

            # remove nan and inf values
            array = np.nan_to_num(array)

            # extract just the first frame of the array (change later on?)
            array = array[:,:,0]
            
            slopes_per_recording = []
            for scanline in range(len(array)):

                one_scanline = array[scanline,:]
                # get slope 
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(one_scanline)), one_scanline)
                slopes_per_recording.append(slope)
            slopes_all_recordings.append(slopes_per_recording)

            # get patient and recording number
            recording_and_slope[dir+'_'+file] = slope
    
slopes_all_recordings = np.array(slopes_all_recordings) 

# get mean slope per recording
mean_slopes_all_recordings = []
for recording in range(len(slopes_all_recordings)):
    mean_slopes_all_recordings.append(np.mean(slopes_all_recordings[recording]))

# get abs value from slopes 
mean_slopes_all_recordings = np.abs(mean_slopes_all_recordings)

print('Slope for recording: ', recording_and_slope)
# %%
# spectral downshift
recording_and_spectral_shift = {}
all_slopes_frequency_shift = []
redcap_value = []
for recording in tqdm(os.listdir(start_directory)):
    # load recording
    for file in os.listdir(start_directory+'/'+recording):
        if transducer in file and tgc in file:
            file_path = start_directory+'/'+recording+'/'+file

            data = np.load(file_path)

            # get sampling frequency from yaml file
            yaml_file = file_path[:-11]+'.yaml'
            sampling_rate, t = definitions.read_yaml_file(yaml_file, data)

            # apply min- and max_depth
            data = data[:,min_depth:max_depth,:]
            #remove the sides from array 
            data = data[start_scanline:end_scanline,:,:]

            # perform stFFT
            stfft = definitions.stFFT(data, window_size, hop_size, sampling_rate)
            stfft_square = np.square(stfft)
            stfft_mean = np.mean(stfft_square, axis = 0)

            # # calibrate stfft -> results become worse -> do not use?
            # if len(stfft_mean[:,0]) < len(phantom_stfft_mean[:,0]):
            #     shortened_phantom_stfft_mean = phantom_stfft_mean[:len(stfft_mean[:,0]),:]
            #     stfft_mean = stfft_mean/ shortened_phantom_stfft_mean
            # else:
            #     stfft_mean = stfft_mean/ phantom_stfft_mean

            mean_frequencies = []
            for depth in range(len(stfft_mean[:,0])):
                # plt.figure()
                # plt.plot(stfft_mean[depth,5:])
                # plt.show()

                blurred_spectrum = ndimage.uniform_filter(stfft_mean[depth,:], size = 3)
                # plt.figure()
                # plt.plot(blurred_spectrum)
                # plt.show()

                # find maximum
                max_index = np.argmax(blurred_spectrum)
                max_value = np.max(blurred_spectrum)
                border_value = max_value/np.power(10, spectral_shift_border/20)
                # define upper border of calibration_spectrum
                for frequency in range(max_index, len(blurred_spectrum)):
                    if blurred_spectrum[frequency] < border_value:
                        upper_index_border = frequency
                        break
                # define lower border of calibration_spectrum
                for frequency in range(max_index, 0, -1):
                    if blurred_spectrum[frequency] < border_value:
                        lower_index_border = frequency
                        break
                
                mean_frequency = int((upper_index_border+lower_index_border)/2)
                
                mean_frequencies.append(mean_frequency)
            mean_frequencies = np.array(mean_frequencies)

            # transform frequency sections in frequency shift into MHz
            # real_mean_frequencies = mean_frequencies*sampling_rate/window_size/1e6
            

            slope_frequency_shift = np.polyfit(mean_frequencies, np.arange(len(mean_frequencies))*(window_size/hop_size), 1)[0]
            all_slopes_frequency_shift = np.append(all_slopes_frequency_shift, slope_frequency_shift)

            # get patient and recording number
            recording_and_spectral_shift[recording+'_'+file] = slope_frequency_shift
all_slopes_frequency_shift = np.array(all_slopes_frequency_shift)

print('Spectral shift for recording: ', recording_and_spectral_shift)
# %%
