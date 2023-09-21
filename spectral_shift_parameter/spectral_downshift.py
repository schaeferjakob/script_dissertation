#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
import functions.definitions as definitions
from tqdm import tqdm
from scipy import ndimage
import functions.def_notion_API as notion_def
# %%
# frequency downshift
# calculate stFFT and evaluate the depth-wise frequency spectrum

# define directory
version = '230726'
directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version
transducer = 'L15_large'
dir_redcap_survey = '/Users/jakobschaefer/Documents/REDCap_Survey_'+version+'.csv'
ref_parameter = 'CAP'
spectral_shift_border = 5 # in dB

# choose calibration type
calibration_type = 'Wat_Phan' # 'Phantom' or 'Met_Phan' or 'AIR00w000' or 'Wat_Phan'

if transducer == 'L15_large':
    # define cutoff values for recording and phantom
    min_depth = 0
    max_depth = 2704
    start_scanline = 0
    number_of_scanlines = 192

    # define window size and hop size
    window_function = 'hamming'
    window_size = 100
    hop_size = 1
elif transducer == 'C3_large':
    # define cutoff values for recording and phantom
    min_depth = 196
    max_depth = 2320
    start_scanline = 57
    number_of_scanlines = 94

    # define window size and hop size
    window_function = 'hamming'
    window_size = 100
    hop_size = 3

end_scanline = start_scanline + number_of_scanlines

# define if results should be saved to notion
safe_to_notion = True # True or False
database_id = 'cd5e889239124dbbacb41d334652ceca'
# define notion parameters
notion_transducer = transducer
notion_window_func = str('Window function: '+window_function+'\nWindow size: '+str(window_size)+'\nHop size: '+str(hop_size)+'\n')
notion_cropping_properties  = str('Min. depth: '+str(min_depth)+'\nMax. depth: '+str(max_depth)+'\nStart scanline: '+str(start_scanline)+'\nEnd scanline: '+str(end_scanline))

#%% 
# load calibration data
phantom_file = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/'+calibration_type+'/'+transducer+'.rf_no_tgc.npy'
phantom_data = np.load(phantom_file)
# get sampling frequency from yaml file
phantom_yaml_file = phantom_file[:-11]+'.yaml'
phantom_sampling_rate, t = definitions.read_yaml_file(phantom_yaml_file, phantom_data)

# # shorten array axially
# phantom_data = phantom_data[:,min_depth:max_depth,:]
# # shorten array laterally
# phantom_data = phantom_data[start_scanline:end_scanline,:,:]
# # phantom stfft
# phantom_stfft = definitions.stFFT(phantom_data, window_size, hop_size, phantom_sampling_rate)
# phantom_stfft_square = np.square(phantom_stfft)
# phantom_stfft_mean = np.mean(phantom_stfft_square, axis = 0)

# get redcap values out of csv file
Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP, scan_conditions = definitions.get_fibroscan_values(dir_redcap_survey)

#%%
all_slopes_frequency_shift = []
redcap_value = []
for recording in tqdm(os.listdir(directory)):
    # load recording
    file = directory+'/'+recording+'/'+transducer+'.rf_no_tgc.npy'
    data = np.load(file)

    # get sampling frequency from yaml file
    yaml_file = file[:-11]+'.yaml'
    sampling_rate, t = definitions.read_yaml_file(yaml_file, data)

    # apply min- and max_depth
    data = data[:,min_depth:max_depth,:]
    #remove the sides from array 
    data = data[start_scanline:end_scanline,:,:]

    # perform stFFT
    stfft = definitions.stFFT(data, window_size, hop_size, sampling_rate, window_function)
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
        
    # get redcap value out of lists from csv file
    number_redcap_value = Patient_ID.index(recording) 
    #choose from which list to get redcap values
    if ref_parameter == 'Median_E':
        list_values = Median_E
    else: 
        list_values = Median_CAP
    redcap_value.append(list_values[number_redcap_value])

    slope_frequency_shift = np.polyfit(mean_frequencies, np.arange(len(mean_frequencies))*(window_size/hop_size), 1)[0]
    all_slopes_frequency_shift = np.append(all_slopes_frequency_shift, slope_frequency_shift)
all_slopes_frequency_shift = np.array(all_slopes_frequency_shift)

# %%
# transform frequency sections in frequency shift into MHz
real_mean_frequencies = mean_frequencies*sampling_rate/(window_size)/1e6

# define depth axis
depth_axis = np.arange(0, len(real_mean_frequencies))
depth_axis = depth_axis*(window_size/hop_size)

# plot the last recording as an example
plt.figure()
plt.scatter(depth_axis, real_mean_frequencies)
plt.ylabel('Mean frequency [MHz]')
plt.xlabel('Depth [px]')
plt.title('Mean frequency of '+transducer+' in '+recording)
plt.show()


# linear regression of slopes and redcap values
x_reg, y_reg, r, r_squared = definitions.regression(redcap_value, all_slopes_frequency_shift, polynomial_degree = 1)

print('Pearson correlation coefficient spectral shift vs. '+ref_parameter+': ', round(r, 3))

# plot
plt.figure()
plt.scatter(redcap_value,all_slopes_frequency_shift)
plt.plot(x_reg, y_reg, color = 'orange')
plt.xlabel(ref_parameter)
plt.ylabel('Spectral shift [Hz]')
plt.title('Correlation between CAP value and spectral shift')
plt.show()

# %%
# save to notion
notion_evaluation_parameter = 'Spectral downshift'
notion_results = str('Pearson coef.: '+str(round(np.abs(r), 3)))
notion_version = str(version+'\n('+str(len(redcap_value))+' Pat.)')
notion_def.safe_to_notion_def(notion_version, notion_evaluation_parameter, notion_results, notion_transducer, notion_cropping_properties,notion_window_func, safe_to_notion, database_id)

# %%
