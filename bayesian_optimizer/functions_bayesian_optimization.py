# black-box functions bayesian-optimization
def pearson_correlation(max_depth, min_depth,number_of_scanlines, start_scanline):
    import sys
    sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
    import functions.definitions as definitions
    import numpy as np
    import os
    from tqdm import tqdm
    from scipy import stats
    import bayesian_optimization as bayesian_optimization

    max_depth = int(round(max_depth))
    min_depth = int(round(min_depth))
    start_scanline = int(round(start_scanline))
    number_of_scanlines = int(round(number_of_scanlines))

    # get configuration parameters
    version, transducer, ref_parameter = bayesian_optimization.config_parameters()

    #  define directory (on SSH to save storage space)
    start_directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version

    if not os.path.isdir(start_directory):
        start_directory = r'C:\Users\JakobSchaefer\Documents\Patientenaufnahmen_vorverarbeitet\C3_and_L15_without_TGC_'+version
    elif not os.path.isdir(start_directory):
        print('Error: '+start_directory+' does not exist.')

    tgc = 'no_tgc'
    ref_parameter = 'CAP' # choose CAP or Median_E

    end_scanline = start_scanline + number_of_scanlines


    # Define window and frame parameters -> based on pulse_length (should be) 5µs and frame_rate (5µs*15MHz = 75 px)
    #if transducer == 'C3':
        #window_size = 50
        #hop_size = window_size // 2
    #if transducer == 'L15':
        #window_size = 50
        #hop_size = window_size // 20


    # get results out of RedCap-Survey (csv)
    
    path_redcap_survey = '/Users/jakobschaefer/Documents/REDCap_Survey_'+version+'.csv'

    if not os.path.isfile(path_redcap_survey):
        path_redcap_survey = r"C:\Users\JakobSchaefer\Documents\RedCap-Surveys\REDCap_Survey_"+version+".csv"
    elif not os.path.isfile(path_redcap_survey):
        print('Error: '+path_redcap_survey+' does not exist.')
        
    # calculations:
    # # calculate stFFT of phantom for normalization
    # # load phantom recording
    # phantom_array = np.load('/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Phantom/'+transducer+'_large.rf_no_tgc.npy')
    # # get sampling rate
    # if transducer == 'C3':
    #     fs = 15e6
    # if transducer == 'L15':
    #     fs = 30e6
    # # shorten array axially
    # phantom_array = phantom_array[:,min_depth:max_depth,:]
    # # shorten array laterally
    # phantom_array = phantom_array[start_scanline:end_scanline,:,:]
    # # get stFFT of phantom
    # phantom_stfft = definitions.stFFT(phantom_array, window_size, hop_size, fs)

    # remove dot-underscore files (they are created when copying files from/to SSH)
    definitions.remove_dot_underscore_files(start_directory)
    for dir in os.listdir(start_directory):
        definitions.remove_dot_underscore_files(start_directory+'/'+dir)


    # get redcap values out of csv file
    Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP, scan_conditions = definitions.get_fibroscan_values(path_redcap_survey)

    redcap_value = []
    slopes_all_recordings = []
    for dir in tqdm(os.listdir(start_directory)):
        # for loop through all recordings:
        for file in os.listdir(start_directory+'/'+dir):
            if transducer in file and tgc in file:
                # get array without tgc from this recording
                array = np.load(start_directory+'/'+dir+'/'+file)

                # skip arrays with less pixels than max_depth
                if len(array[0,:,0]) < max_depth:
                    break

                # get redcap value out of lists from csv file
                number_redcap_value = Patient_ID.index(dir) 
                #choose from which list to get redcap values
                if ref_parameter == 'Median_E':
                    list_values = Median_E
                else: 
                    list_values = Median_CAP
                redcap_value.append(list_values[number_redcap_value])
                    
                # apply min- and max_depth
                array = array[:,min_depth:max_depth,:]

                #remove the sides from array 
                array = array[start_scanline:end_scanline,:,:]

                # apply log10 to data (to perform linear regression)
                array = np.log10(np.abs(array)+1e-10)

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
        
    slopes_all_recordings = np.array(slopes_all_recordings) 

    # get mean slope per recording
    mean_slopes_all_recordings = []
    for recording in range(len(slopes_all_recordings)):
        mean_slopes_all_recordings.append(np.mean(slopes_all_recordings[recording]))

    # get abs value from slopes 
    mean_slopes_all_recordings = np.abs(mean_slopes_all_recordings)

    slope, intercept, r, p, std_err = stats.linregress(redcap_value, mean_slopes_all_recordings)

    return np.abs(r)

def frequency_specific_pearson_correlation(hop_size, max_depth, min_depth,number_of_scanlines, start_scanline, window_size, window_function):
    import sys
    sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory    
    import functions.definitions as definitions
    import numpy as np
    import os
    from tqdm import tqdm
    from scipy import stats
    import math
    import bayesian_optimization as bayesian_optimization

    max_depth = int(round(max_depth))
    min_depth = int(round(min_depth))
    start_scanline = int(round(start_scanline))
    number_of_scanlines = int(round(number_of_scanlines))
    window_size = int(round(window_size))
    hop_size = int(round(hop_size))
    window_function = int(round(window_function))

    if window_function == 0:
        window_function = 'none'
    elif window_function == 1:
        window_function = 'hanning'
    elif window_function == 2:
        window_function = 'hamming'
    elif window_function == 3:
        window_function = 'blackman'

    # get configuration parameters
    version, transducer, ref_parameter = bayesian_optimization.config_parameters()

    #  define directory (on SSH to save storage space)
    start_directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version
    tgc = 'no_tgc'


    if window_size > (max_depth-min_depth):
        window_size = max_depth-min_depth
    if hop_size > window_size:
        hop_size = window_size

    end_scanline = start_scanline + number_of_scanlines

    print('hop_size: ', hop_size)
    print('max_depth: ', max_depth)
    print('min_depth: ', min_depth)
    print('number_of_scanlines: ', number_of_scanlines)
    print('start_scanline: ', start_scanline)
    print('window_size: ', window_size)
    print('window_function: ', window_function)

    # get results out of RedCap-Survey (csv)
    path_redcap_survey = '/Users/jakobschaefer/Documents/REDCap_Survey_'+version+'.csv'

    # for use one windows change directory paths
    if not os.path.isdir(start_directory):
        start_directory = r'C:\Users\JakobSchaefer\Documents\Patientenaufnahmen_vorverarbeitet\C3_and_L15_without_TGC_'+version
    elif not os.path.isdir(start_directory):
        print('Error: '+start_directory+' does not exist.')

    if not os.path.isfile(path_redcap_survey):
        path_redcap_survey = r"C:\Users\JakobSchaefer\Documents\RedCap-Surveys\REDCap_Survey_"+version+".csv"
    elif not os.path.isfile(path_redcap_survey):
        print('Error: '+path_redcap_survey+' does not exist.')

    # remove dot-underscore files (they are created when copying files from/to SSH)
    definitions.remove_dot_underscore_files(start_directory)
    for dir in os.listdir(start_directory):
        definitions.remove_dot_underscore_files(start_directory+'/'+dir)

    # get redcap values out of csv file
    Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP, scan_conditions = definitions.get_fibroscan_values(path_redcap_survey)

    redcap_value = []
    slopes_all_recordings = []
    pearson_per_frequency = []
    stfft_all_recordings = []
    list_resorted_fibroscan_values = []
    recordings_count = 0
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

                # read in metadata from yaml file
                if transducer == 'C3':
                    yaml_file = start_directory+'/'+dir+'/'+file[:-11]+'.yaml'
                    sampling_rate,  t = definitions.read_yaml_file(yaml_file, array)
                else:
                    yaml_file = start_directory+'/'+dir+'/'+'L15_large.rf.yaml'
                    sampling_rate,  t = definitions.read_yaml_file(yaml_file, array)

                # get redcap value out of lists from csv file
                number_redcap_value = Patient_ID.index(dir) 
                #choose from which list to get redcap values
                if ref_parameter == 'Median_E':
                    list_values = Median_E
                else: 
                    list_values = Median_CAP
                redcap_value.append(list_values[number_redcap_value])
                    
                # apply min- and max_depth
                array = array[:,min_depth:max_depth,:]

                #remove the sides from array 
                array = array[start_scanline:end_scanline,:,:]
                
                #calculate stFFT of recording
                stfft = definitions.stFFT(array, window_size, hop_size, sampling_rate, window_function)

                stfft_all_recordings.append(stfft)

                '''
                if len(stfft_all_recordings) != len(redcap_value):
                    print('error')
                '''

    stfft_all_recordings = np.array(stfft_all_recordings)
    # shape stfft_all_recordings: (number_of_recordings, number_of_scanlines, number_of_timeframes, number_of_frequencies)

    

    # stfft_all_recordings has to be shortened in time, because the phantom is shorter than the recordings (idk why, shoudn't be the case)
    stfft_all_recordings = stfft_all_recordings[:,:,:-1,:]
    
    # calculate frequency-dependent slope of stFFT for each recording
    slopes_all_recordings = []
    normalized_stfft_all_recordings = np.zeros(stfft_all_recordings.shape)

    for recording in tqdm(range(len(stfft_all_recordings))):
        slopes_per_scanline = []

        # divide stFFT of recording by stFFT of phantom
        normalized_stfft_all_recordings[recording] = stfft_all_recordings[recording] #/ (phantom_stfft+1e-10) #since I know, the division by the phantom makes no difference 

        # for-loop through all scanlines
        for scanline in range(len(normalized_stfft_all_recordings[recording])):
            # for-loop through all frequencies
            slopes_per_frequency = []
            for frequency in range(len(normalized_stfft_all_recordings[recording,scanline,0,:])):
                # get stFFT of recording
                stfft = normalized_stfft_all_recordings[recording,scanline,:,frequency]

                # bring stFFT to log scale (to get linear regression)
                stfft = np.log10(stfft+1e-10)
                # remove nan values
                stfft = np.nan_to_num(stfft)

                # get slope of stFFT
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(stfft)), stfft)
                slopes_per_frequency.append(slope)

                if math.isnan(slope):
                    print('nan')
            slopes_per_scanline.append(slopes_per_frequency)

        slopes_all_recordings.append(slopes_per_scanline)
    slopes_all_recordings = np.array(slopes_all_recordings)

    # calculate mean slope value for all scanlines of one frequency of one recording
    mean_slopes_all_recordings = []
    for recording in range(len(slopes_all_recordings)):
        mean_slope_per_frequency = []
        for frequency in range(len(slopes_all_recordings[recording,0,:])):
            mean_slope_per_frequency.append(np.mean(slopes_all_recordings[recording,:,frequency]))
        mean_slopes_all_recordings.append(mean_slope_per_frequency)
    mean_slopes_all_recordings = np.array(mean_slopes_all_recordings)

    # calculate pearson correlation coefficient between redcap value and frequency slope
    pearson_per_frequency = []
    for frequency in range(len(mean_slopes_all_recordings[0,:])):
        slope, intercept, r_value, p_value, std_err = stats.linregress(redcap_value, np.array(mean_slopes_all_recordings)[:,frequency])
        pearson_per_frequency.append(r_value)  

    # get abs-values of pearson correlation coefficient
    pearson_per_frequency = np.abs(pearson_per_frequency)

    max_pearson = np.max(pearson_per_frequency)

    return max_pearson

def spectral_shift(spectral_shift_border, min_depth, max_depth, start_scanline, number_of_scanlines, window_size, hop_size, window_function):
    import sys
    sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import functions.definitions as definitions
    from scipy import stats
    from tqdm import tqdm
    from scipy import ndimage
    import bayesian_optimization as bayesian_optimization

    # frequency downshift
    # calculate stFFT and evaluate the depth-wise frequency spectrum

    # define directory
    version, transducer, ref_parameter = bayesian_optimization.config_parameters()

    start_directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version
    path_redcap_survey = '/Users/jakobschaefer/Documents/REDCap_Survey_'+version+'.csv'

    # for use one windows changee directory paths
    if not os.path.isdir(start_directory):
        start_directory = r'C:\Users\JakobSchaefer\Documents\Patientenaufnahmen_vorverarbeitet\C3_and_L15_without_TGC_'+version
    elif not os.path.isdir(start_directory):
        print('Error: '+start_directory+' does not exist.')

    if not os.path.isfile(path_redcap_survey):
        path_redcap_survey = r"C:\Users\JakobSchaefer\Documents\RedCap-Surveys\REDCap_Survey_"+version+".csv"
    elif not os.path.isfile(path_redcap_survey):
        print('Error: '+path_redcap_survey+' does not exist.')

    max_depth = int(round(max_depth))
    min_depth = int(round(min_depth))
    start_scanline = int(round(start_scanline))
    number_of_scanlines = int(round(number_of_scanlines))
    window_size = int(round(window_size))
    hop_size = int(round(hop_size))
    spectral_shift_border = int(round(spectral_shift_border))
    window_function = int(round(window_function))

    if window_function == 0:
        window_function = 'none'
    elif window_function == 1:
        window_function = 'hanning'
    elif window_function == 2:
        window_function = 'hamming'
    elif window_function == 3:
        window_function = 'blackman'

    if window_size > (max_depth-min_depth):
        window_size = max_depth-min_depth
    if hop_size > window_size:
        hop_size = window_size

    end_scanline = start_scanline + number_of_scanlines

    print('hop_size: ', hop_size)
    print('max_depth: ', max_depth)
    print('min_depth: ', min_depth)
    print('number_of_scanlines: ', number_of_scanlines)
    print('start_scanline: ', start_scanline)
    print('window_size: ', window_size)
    print('spectral_shift_border: ', spectral_shift_border)
    print('Window function:', window_function)

    # # load phantom data
    # phantom_file = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Phantom/'+transducer+'.rf_no_tgc.npy'
    # phantom_data = np.load(phantom_file)
    # # get sampling frequency from yaml file
    # phantom_yaml_file = phantom_file[:-11]+'.yaml'
    # phantom_sampling_rate, t = definitions.read_yaml_file(phantom_yaml_file, phantom_data)

    # # shorten array axially
    # phantom_data = phantom_data[:,min_depth:max_depth,:]
    # # shorten array laterally
    # phantom_data = phantom_data[start_scanline:end_scanline,:,:]
    # # phantom stfft
    # phantom_stfft = definitions.stFFT(phantom_data, window_size, hop_size, phantom_sampling_rate)
    # phantom_stfft_square = np.square(phantom_stfft)
    # phantom_stfft_mean = np.mean(phantom_stfft_square, axis = 0)

    # get redcap values out of csv file
    Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP, scan_conditions = definitions.get_fibroscan_values(path_redcap_survey)


    all_slopes_frequency_shift = []
    redcap_value = []
    for recording in tqdm(os.listdir(start_directory)):
        # load recording
        file = start_directory+'/'+recording+'/'+transducer+'.rf_no_tgc.npy'
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

            # sometimes the first values (lowest frequencies) are unrealisticly high -> probably due to electrical artefacts
            # -> set them to zero
            if blurred_spectrum[0] > blurred_spectrum[5]*2:
                blurred_spectrum[0] = 0
                         
            # find maximum
            max_index = np.argmax(blurred_spectrum)
            max_value = np.max(blurred_spectrum)
            border_value = max_value/np.power(10, spectral_shift_border/20)
            # define upper border of calibration_spectrum
            for frequency in range(max_index, len(blurred_spectrum)):
                if blurred_spectrum[frequency] < border_value:
                    upper_index_border = frequency
                    break
                else:
                    upper_index_border = len(blurred_spectrum)-1
            # define lower border of calibration_spectrum
            for frequency in range(max_index, 0, -1):
                if blurred_spectrum[frequency] < border_value:
                    lower_index_border = frequency
                    break
                else:
                    lower_index_border = 0
                    
            try:
                mean_frequency = int((upper_index_border+lower_index_border)/2)
            except:
                try:
                    mean_frequency = int(upper_index_border/2)
                except:
                    mean_frequency = int(lower_index_border+len(len(blurred_spectrum))/2)

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

        slope_frequency_shift, intercept, r_value, p_value, std_err = stats.linregress(mean_frequencies, np.arange(len(mean_frequencies))*(window_size/hop_size))

        all_slopes_frequency_shift = np.append(all_slopes_frequency_shift, slope_frequency_shift)
    all_slopes_frequency_shift = np.array(all_slopes_frequency_shift)


    # transform frequency sections in frequency shift into MHz
    real_mean_frequencies = mean_frequencies*sampling_rate/(window_size)/1e6

    # define depth axis
    depth_axis = np.arange(0, len(real_mean_frequencies))
    depth_axis = depth_axis*(window_size/hop_size)

    # linear regression of slopes and redcap values
    x_reg, y_reg, r, r_squared = definitions.regression(redcap_value, all_slopes_frequency_shift, polynomial_degree = 1)

    return r


def lizzi_feleppa(start_ROI, end_ROI, start_scanline, number_of_scanlines, calibration_border, reflection_max):#, freq_shift_comp):
    import numpy as np
    import sys
    sys.path.insert(0, '/Users/jakobschaefer/Documents/rf-ultrasound')  # necessary to import functions from other directory
    import functions.definitions as definitions
    from tqdm import tqdm
    import os
    import bayesian_optimization as bayesian_optimization
    
    # round parameters given by bayesian optimization (to handle them in this function)
    start_ROI = int(round(start_ROI))
    end_ROI = int(round(end_ROI))
    start_scanline = int(round(start_scanline))
    number_of_scanlines = int(round(number_of_scanlines))
    calibration_border = int(round(calibration_border))
    #freq_shift_comp = int(round(freq_shift_comp))
    reflection_max = int(round(reflection_max))

    # reflection_max has to be greater than half the difference between start_ROI and end_ROI
    if reflection_max < (end_ROI-start_ROI)//2:
        reflection_max = (end_ROI-start_ROI)//2


    window_function = 'hamming'
    Phantom = 'Wat_Phan'

    specific_ROI = [start_ROI, end_ROI]

    end_scanline = start_scanline + number_of_scanlines

    print('start_ROI: ', start_ROI)
    print('end_ROI: ', end_ROI)
    print('start_scanline: ', start_scanline)
    print('number_of_scanlines: ', number_of_scanlines)
    print('calibration_border: ', calibration_border)
    print('reflection_max: ', reflection_max)


    # get configuration parameters
    version, transducer, ref_parameter = bayesian_optimization.config_parameters()
    lizzi_feleppa_parameter, ROI_type, len_ROI, remove_outliers = bayesian_optimization.config_lizzi_feleppa()

    # define where is the middle of reflection of the calibration phantom
    # if transducer == 'L15_large':
    #     reflection_max = 1300
    # if transducer == 'C3_large':
    #     reflection_max = 1800


    directory = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version

    # get values from redcap survey
    path_redcap_survey = '/Users/jakobschaefer/Documents/REDCap_Survey_'+version+'.csv'

    # for use one windows change directory paths
    if not os.path.isdir(directory):
        directory = r'C:\Users\JakobSchaefer\Documents\Patientenaufnahmen_vorverarbeitet\C3_and_L15_without_TGC_'+version
    elif not os.path.isdir(directory):
        print('Error: '+directory+' does not exist.')

    if not os.path.isfile(path_redcap_survey):
        path_redcap_survey = r"C:\Users\JakobSchaefer\Documents\RedCap-Surveys\REDCap_Survey_"+version+".csv"
    elif not os.path.isfile(path_redcap_survey):
        print('Error: '+path_redcap_survey+' does not exist.')


    if Phantom != 'fft_all_recordings':   
        # calculate fft of phantom recording
        phantom_file_path = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/'+Phantom+'/'+transducer+'.rf_no_tgc.npy'

        # change path if on windows
        if not os.path.isfile(phantom_file_path):
            # somehow python detects ' not as the end of the string if it stands after \ -> therefore I have to write the path like this
            phantom_file_path = os.path.join("C:\\Users\\JakobSchaefer\\Documents\\Patientenaufnahmen_vorverarbeitet\\Calibration" ,Phantom,transducer+'.rf_no_tgc.npy')
            

        phantom_data = np.load(phantom_file_path)
        phantom_data = phantom_data[:,:,0]
        # cut out specific ROI from phantom_data
        if ROI_type == 1:
            phantom_data = phantom_data[:,len(phantom_data[0,:])//2-(len_ROI//2):len(phantom_data[0,:])//2+(len_ROI//2)]
        if ROI_type == 2:
            if Phantom == 'Wat_Phan':
                # if specific_ROI[1]+freq_shift_comp > len(phantom_data[0,:]):
                #     freq_shift_comp = len(phantom_data[0,:])-specific_ROI[1]
                # phantom_data = phantom_data[:,specific_ROI[0]+freq_shift_comp:specific_ROI[1]+freq_shift_comp]
                len_specific_ROI = specific_ROI[1]-specific_ROI[0]
                phantom_data = phantom_data[:,reflection_max-(len_specific_ROI//2):reflection_max+(len_specific_ROI//2)]
                if len(phantom_data[0,:]) > len_specific_ROI:
                    phantom_data = phantom_data[:,:-1]
            else:
                phantom_data = phantom_data[:,specific_ROI[0]:specific_ROI[1]]
                #print('freq_shift_comp: ', freq_shift_comp)


        # get sampling frequency from yaml file
        yaml_file = phantom_file_path[:-14]+'.rf.yaml'
        sampling_rate, t = definitions.read_yaml_file(yaml_file, phantom_data)

        phantom_y_fft = []
        try:
            for scanline in range(len(phantom_data[:,0])):
                phantom_x_fft, phantom_y_fft_scanline = definitions.amplitude_spectrum(sampling_rate, phantom_data[scanline,:], window_function)
                phantom_y_fft.append(phantom_y_fft_scanline)
            phantom_y_fft = np.array(phantom_y_fft)
        except:
            print('len:phantom_data: '+ len(phantom_data[:,0]))
        phantom_y_fft_square = np.square(phantom_y_fft)
        calibration_spectrum = np.mean(phantom_y_fft_square, axis = 0)
        

    elif Phantom == 'fft_all_recordings':
        try:
            try:
                phantom_fft_path = "/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/Calibration/mean_fft_all_recordings/mean_fft_all_recordings_" + str(specific_ROI[0])+'-'+str(specific_ROI[1])+'_'+version+'_'+transducer+'.npy'
            except:
                phantom_fft_path = r'C:\Users\JakobSchaefer\Documents\Patientenaufnahmen_vorverarbeitet\Calibration\mean_fft_all_recordings\mean_fft_all_recordings_'+str(specific_ROI[0])+'-'+str(specific_ROI[1])+'_'+version+'_'+transducer+'.npy'
            
            phantom_fft = np.load(phantom_fft_path)
        except:
            # calculate FFT for all recordings
            fft_all_recordings = []
            for recording in tqdm(os.listdir(directory)):
                # load file
                array = np.load(directory+'/'+recording+'/'+transducer+'.rf_no_tgc.npy')

                # cut off sides of array 
                array = array[start_scanline:end_scanline,:,:]

                # sort out recordings where array is shorter than a certain length
                if len(array[0,:,0]) >= 2704:
                
                    array = array[:,specific_ROI[0]:specific_ROI[1],:]

                    # get sampling rate out of yaml file
                    yaml_file = str(directory)+'/'+recording+'/'+transducer+'.rf.yaml'
                    sampling_rate, t = definitions.read_yaml_file(yaml_file, array)

                    yfft_all_scanlines = []
                    # get fft from every scanline and calculate mean
                    for scanline in range(len(array[:,0,0])):

                        # perform FFT on first scanline in first frame
                        x_fft,y_fft = definitions.amplitude_spectrum(sampling_rate, array[scanline,:,0], window_function)

                        yfft_all_scanlines.append(y_fft)
                    yfft_all_scanlines = np.array(yfft_all_scanlines)
                    mean_yfft_all_scanlines = np.mean(yfft_all_scanlines, axis=0)



                    # append FFT to list
                    fft_all_recordings.append(mean_yfft_all_scanlines)
            fft_all_recordings = np.array(fft_all_recordings)

            fft_all_recordings = np.square(fft_all_recordings)
            mean_fft_all_recordings = np.mean(fft_all_recordings, axis = 0)

            phantom_fft = mean_fft_all_recordings, x_fft

        phantom_y_fft = phantom_fft[0]
        phantom_x_fft = phantom_fft[1]
        calibration_spectrum = np.square(phantom_y_fft)
    

    # get max_value from calibration_spectrum and calculate -15dB border'
    max_value_calibration = np.max(calibration_spectrum)
    index_max_value_calibration = np.argmax(calibration_spectrum)
    border_value = max_value_calibration/np.power(10, calibration_border/20)
    # define upper border of calibration_spectrum
    for frequency in range(index_max_value_calibration, len(calibration_spectrum)):
        if calibration_spectrum[frequency] < border_value:
            upper_index_border = frequency
            break  
        else: 
            upper_index_border = len(calibration_spectrum)-1
    # define lower border of calibration_spectrum
    for frequency in range(index_max_value_calibration, 0, -1):
        if calibration_spectrum[frequency] < border_value:
            lower_index_border = frequency
            break
        else:
            lower_index_border = 0
    new_phantom_x_fft = phantom_x_fft[lower_index_border:upper_index_border]
    new_calibration_spectrum = calibration_spectrum[lower_index_border:upper_index_border]

    # get redcap values out of csv file
    Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP, scan_conditions = definitions.get_fibroscan_values(path_redcap_survey)

    
    # calculate fft of recording
    quotient_all_recordings = []
    MBF_all_recordings = []
    slopes_all_recordings = []
    intercepts_all_recordings = []
    redcap_value = []
    for recording in tqdm(os.listdir(directory)):

        file = directory+'/'+recording+'/'+transducer+'.rf_no_tgc.npy'
        data = np.load(file)

        # cut off sides of array 
        data = data[start_scanline:end_scanline,:,:]

        # define sampling frequency
        yaml_file = directory+'/'+recording+'/'+transducer+'.rf.yaml'
        sampling_rate, t = definitions.read_yaml_file(yaml_file, data)
        
        # get redcap value out of lists from csv file
        number_redcap_value = Patient_ID.index(recording) 
        #choose from which list to get redcap values
        if ref_parameter == 'E':
            list_values = Median_E
        elif ref_parameter == 'CAP': 
            list_values = Median_CAP
        else:
            print('ERROR: ref_parameter not defined correctly!')
        redcap_value.append(list_values[number_redcap_value])

        # just take first frame
        data = data[:,:,0]

        # cut out specific ROI from data
        if ROI_type == 1:
            data = data[:,len(data[0,:])//2-(len_ROI//2):len(data[0,:])//2+(len_ROI//2)]
        if ROI_type == 2:
            if Phantom == 'Wat_Phan':
                data = data[:,specific_ROI[0]:specific_ROI[1]]
            else:
                data = data[:,specific_ROI[0]:specific_ROI[1]]



        # # adapt the length of pahntom_data and data together (by zero-padding)
        # if len(data[0,:]) > len(phantom_data[0,:]):
        #     phantom_data = np.pad(phantom_data, ((0,0),(0,len(data[0,:])-len(phantom_data[0,:]))), 'constant', constant_values=0)
        # if len(data[0,:]) < len(phantom_data[0,:]):
        #     data = np.pad(data, ((0,0),(0,len(phantom_data[0,:])-len(data[0,:]))), 'constant', constant_values=0)

        # calculate fft of recording
        y_fft = []
        for scanline in range(len(data[:,0])):
            x_fft, y_fft_scanline = definitions.amplitude_spectrum(sampling_rate, data[scanline,:], window_function)
            y_fft.append(y_fft_scanline)
        y_fft = np.array(y_fft)

        # square of fft
        y_fft_square = np.square(y_fft)
        # average of fft (by that all scanlines are averaged together to one spectrum)
        average_power_spectrum = np.mean(y_fft_square, axis = 0)

        # # JUST TO TRY SOMETHING -> DELETE LATER!!
        # artificial_frequency_shift = np.where(x_fft > 0.05e6)[0][0]
        # average_power_spectrum = np.append(np.zeros(artificial_frequency_shift),average_power_spectrum[:-artificial_frequency_shift])    

        # define index of border
        new_x_fft = x_fft[lower_index_border:upper_index_border]
        new_average_power_spectrum = average_power_spectrum[lower_index_border:upper_index_border]

        # quotient of recording fft and phantom fft 
        try: # some shorter 
            quotient = new_average_power_spectrum/new_calibration_spectrum
        
            # Calculate Lizzi-Feleppa-Parameters

            x_quotient = new_x_fft

            # linear regression
            slope_quotient, intercept_quotient, r_value, p_value, mymodel = definitions.linear_regression(x_quotient, quotient)

            # calculate Midband Fit
            MBF = slope_quotient * (x_quotient[round(len(x_quotient)/2)]) + intercept_quotient

            # collect values for all recordings
            quotient_all_recordings.append(quotient)
            MBF_all_recordings.append(MBF)
            slopes_all_recordings.append(slope_quotient)
            intercepts_all_recordings.append(slope_quotient * 0 + intercept_quotient)

        except: # skip that recording
            redcap_value = redcap_value[:-1]
            pass
    quotient_all_recordings = np.array(quotient_all_recordings)
    slopes_all_recordings = np.array(slopes_all_recordings)
    intercepts_all_recordings = np.array(intercepts_all_recordings)
    MBF_all_recordings = np.array(MBF_all_recordings)

    # if lizzi_feleppa_parameter == 'MBF':
    #     evaluation_parameters = MBF_all_recordings
    # elif lizzi_feleppa_parameter == 'slope':
    #     evaluation_parameters = slopes_all_recordings
    # elif lizzi_feleppa_parameter == 'intercept':
    #     evaluation_parameters = intercepts_all_recordings

    # if not remove_outliers: 
    #     slope, intercept, r_value, p_value, mymodel = definitions.linear_regression(redcap_value, evaluation_parameters)
    # elif remove_outliers:
    #     # define borders for outliers
    #     lower_th = 0.05
    #     upper_th = 0.95
    #     quartile1 = np.quantile(evaluation_parameters, lower_th)
    #     quartile3 = np.quantile(evaluation_parameters, upper_th)
    #     iqr = quartile3 - quartile1
    #     upper_limit = quartile3 + 1.5 * iqr
    #     lower_limit = quartile1 - 1.5 * iqr

    #     # detect outliers
    #     list_outliers = []
    #     for element in range(len(evaluation_parameters)-1):
    #         if np.abs(evaluation_parameters[element]) > upper_limit or np.abs(evaluation_parameters[element]) < lower_limit:
    #             list_outliers.append(element)
        
    #     print('Number of outliers: ', len(list_outliers))

    #     # remove outliers
    #     evaluation_parameters_no_outliers = np.delete(evaluation_parameters, list_outliers)
    #     redcap_value_outlier_removal = np.delete(redcap_value, list_outliers)

    #     # linear regression of parameter and redcap values
    #     slope, intercept, r_value, p_value, mymodel = definitions.linear_regression(redcap_value_outlier_removal, evaluation_parameters_no_outliers)

    # try to find the best overall fit 
    # mean of squared pearson correlation coefficient
    slope, intercept, pearson_MBF, p_value, mymodel = definitions.linear_regression(redcap_value, MBF_all_recordings) 
    slope, intercept, pearson_slope, p_value, mymodel = definitions.linear_regression(redcap_value, slopes_all_recordings)
    slope, intercept, pearson_intercept, p_value, mymodel = definitions.linear_regression(redcap_value, intercepts_all_recordings)

    overall_pearson = pearson_MBF**2 + pearson_slope**2 + pearson_intercept**2


    return overall_pearson #np.abs(r_value)
 

# start_ROI:  118
# end_ROI:  2156
# start_scanline:  0
# number_of_scanlines:  49
# window_function:  blackman
# calibration_border:  30

start_ROI =  118
end_ROI =  2156
start_scanline =  0
number_of_scanlines =  49
window_function =  3
calibration_border =  30
Phantom =  0


# pearson_coeff =  lizzi_feleppa(start_ROI, end_ROI, start_scanline, number_of_scanlines, window_function, calibration_border, Phantom)

# print(pearson_coeff)