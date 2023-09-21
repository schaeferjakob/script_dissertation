
# unpack tar-files
def unpack_tar(directory):
    import tarfile
    import os
    for root, dirs, files in os.walk(directory):  
        for name in files:
             if name.endswith('.tar'):
                my_tar = tarfile.open(root+"/"+name)
                my_tar.extractall(root+'/'+name[:-4]) # specify which folder to extract to
                my_tar.close()
   
# decompress lzo data in directory
def lzo_decompressor(directory):
    import os
    for root, dirs, files in os.walk(directory):  
        for name in files:
            if name.endswith(('rf.raw.lzo','env.raw.lzo')):
                name_length = len(name)
                file_path_1 = os.path.abspath(name)
                file_path_2 = file_path_1[:-name_length]+ root +'/'+file_path_1[-name_length:]
                os.system('lzop -d {}'.format(file_path_2))

# copy rf.raw-files from start_directory into end_directory
def copy_rf_raw_files(start_directory, end_directory):
    import os
    import shutil
    
    for root, dirs, files in os.walk(start_directory):
        for name in files:
            if name.endswith('rf.raw'):
                name_length = len(name)
                file_path_1 = os.path.abspath(name)
                file_path_2 = file_path_1[:-name_length]+ root +'/'+file_path_1[-name_length:]
                shutil.copy(file_path_2, end_directory)
                   
# read rf data
def read_rf(filename):
    import numpy as np
    import matplotlib.pyplot as plt
    hdr_info = ('id', 'frames', 'lines', 'samples', 'samplesize')
    hdr, timestamps, data = {}, None, None
    with open(filename, 'rb') as raw_bytes:
        # read 4 bytes header
        for info in hdr_info:
            hdr[info] = int.from_bytes(raw_bytes.read(4), byteorder='little')
        # read timestamps and data
        timestamps = np.zeros(hdr['frames'], dtype='int64')
        sz = hdr['lines'] * hdr['samples'] * hdr['samplesize']
        data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='int16')
        for frame in range(hdr['frames']):
            # read 8 bytes of timestamp
            timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')
            # read each frame
            data[:, :, frame] = np.frombuffer(raw_bytes.read(sz), dtype='int16').reshape([hdr['lines'], hdr['samples']])
    print('Loaded {d[2]} raw frames of size, {d[0]} x {d[1]} (lines x samples)'.format(d=data.shape))

    return hdr, timestamps, data

# read env data
def read_env(filename):
    import numpy as np
    hdr_info = ('id', 'frames', 'lines', 'samples', 'samplesize')
    hdr, timestamps, data = {}, None, None
    with open(filename, 'rb') as raw_bytes:
        # read 4 bytes header 
        for info in hdr_info:
            hdr[info] = int.from_bytes(raw_bytes.read(4), byteorder='little')
        # read timestamps and data
        timestamps = np.zeros(hdr['frames'], dtype='int64')
        sz = hdr['lines'] * hdr['samples'] * hdr['samplesize']
        data = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='uint8')
        for frame in range(hdr['frames']):
            # read 8 bytes of timestamp
            timestamps[frame] = int.from_bytes(raw_bytes.read(8), byteorder='little')
            # read each frame
            data[:, :, frame] = np.frombuffer(raw_bytes.read(sz), dtype='uint8').reshape([hdr['lines'], hdr['samples']])
    print('Loaded {d[2]} raw frames of size, {d[0]} x {d[1]} (lines x samples)'.format(d=data.shape))
    return hdr, timestamps, data

# plots envelope image
def plot_env_image(filepath, root):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    # read b/envelope data
    hdr, timestamps, data = read_env(filepath)

    # display b data
    numframes = 1  # hdr['frames']
    for frame in range(numframes):
        plt.figure(figsize=(5, 5))
        plt.imshow(np.transpose(data[:, :, frame]), cmap=plt.cm.gray, aspect='auto', vmin=0, vmax=255)
        plt.title('Envelope ' + str(root))
        plt.show()
    
    return(data)

        
# plot rf data, convert it into watchable image and display it
def plot_rf_image(filepath, root, numframes, sampling_rate, soundspeed):
    import numpy as np
    from scipy.signal import hilbert
    import matplotlib.pyplot as plt
    import sys
    libPath = "../../common/python"
    if not libPath in sys.path: sys.path.append(libPath)

    time_per_pixel = 1/sampling_rate
    centimeter_per_one_pixel = time_per_pixel*(soundspeed*100)/2

    hdr, timestamps, data = read_rf(filepath) 

    bdata = np.zeros((hdr['lines'], hdr['samples'], hdr['frames']), dtype='float')
    for frame in range(numframes):
        bdata[:,:,frame] = 20 * np.log10(np.abs(1 + hilbert(data[:,:,frame])))
    
    one_line = np.linspace(0,len(data[0,:,0]), len(data[0,:,0]))
    yaxis_in_cm = np.linspace(0, len(one_line)*centimeter_per_one_pixel, len(one_line))
    
    # display images
    for frame in range(numframes):
        plt.figure(figsize=(5,5))
        #plt.subplot(1,2,1)
        #plt.imshow(np.transpose(data[:,:,frame]), cmap=plt.cm.plasma, aspect='auto', vmin=-1000, vmax=1000 ) # rf-picture 
        #plt.title('RF '+ str(root) + 'Frame ' + str(frame))
        #plt.subplot(1,2,2)
        plt.imshow(np.transpose(bdata[:,:,frame]), cmap=plt.cm.gray, aspect='auto', vmin=15, vmax=70 )
        plt.title('RF frame ' +str(root) +' '+ str(frame))
        plt.yticks(one_line[::int(1/centimeter_per_one_pixel)], np.around(yaxis_in_cm[::int(1/centimeter_per_one_pixel)]))
        plt.xlabel('Lines')
        plt.ylabel('Depth ($cm$)')
        plt.show()
    
    return data
           
# get rf-data 
def load_rf_image(filepath):
    import numpy as np
    from scipy.signal import hilbert
    import matplotlib.pyplot as plt
    import sys
    libPath = "../../common/python"
    if not libPath in sys.path: sys.path.append(libPath)

    hdr, timestamps, data = read_rf(filepath) 
    return data

# read out DICOM file
def read_DICOM(dicom_file):
    import pydicom
    import pydicom.data
    ds = pydicom.dcmread(dicom_file)
    return ds.pixel_array, ds

def get_a_scan(directory, sampling_rate, imaging_depth, soundspeed, line):
    # get sampling_rate and soundspeed out of rf.yml file (sometimes soundspeed is not provided)
    # sampling_rate in Hz (aus yml-file)
    # imaging_depth  in cm (aus yml-file)
    # soundspeed in cm/s (meist 14.5e4)
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    time_per_pixel = 1/sampling_rate
    #search rf-file in chosen directory
    for root, dirs, files in os.walk(directory):  
            for name in files:
                if name.endswith('rf.raw'):
                    name_length = len(name)
                    file_path_1 = os.path.abspath(name)
                    file_path_2 = file_path_1[:-name_length]+ root +'/'+file_path_1[-name_length:]
    rf_filename = file_path_2

    hdr, timestamps, data = read_rf(rf_filename) # read rf-file
    if line == {}: 
        middle_line = len(data[:,0,0])/2
    else:
        middle_line = line
    a_scan = data[int(middle_line)]
    a_scan = a_scan[:,0]
    a_scan = a_scan.tolist()

    x_axis = np.linspace(0, len(a_scan)*(time_per_pixel*soundspeed)/2, len(a_scan))

    plt.plot(x_axis, a_scan)
    plt.xlabel('Depth ($cm$)')
    plt.ylabel('Amplitude ($Unit$)')
    
    return a_scan



# plot fft and remove phase information
def amplitude_spectrum(sr, s, window_function, highpass_filter = True):  # sr =sampling rate #https://www.cbcity.de/die-fft-mit-python-einfach-erklaert
    import numpy as np
    import numpy.fft as fft
    import matplotlib.pyplot as plt
    from numpy import hanning 
    from scipy import signal

    if highpass_filter == True:
        sos = signal.butter(2, 1e6, 'hp', fs=sr, output='sos')
        s = signal.sosfilt(sos, s)

    if window_function == 'hanning':
        hann = np.hanning(len(s))
        y_fft = fft.fft(hann*s)
    elif window_function == 'blackman': 
        blackman = np.blackman(len(s))
        y_fft = fft.fft(blackman*s)
    elif window_function == 'hamming':
        hamming = np.hamming(len(s))
        y_fft = fft.fft(hamming*s)
    elif window_function == 'none':
        y_fft = fft.fft(s)
    else:
        print('ERROR: Window function not known!') 

    n_fft = int(len(y_fft)/2+1)
    
    x_fft = np.linspace(0, sr/2, n_fft, endpoint=True)
    y_fft_plot = 2.0*np.abs(y_fft[:n_fft])/n_fft # hierbei Umwandlung von komplexen in reele Zahlen über np.abs 

    #plt.plot(x_fft, y_fft_plot)
    #plt.xlabel('Frequency ($MHz$)')
    #plt.ylabel('Amplitude ($Unit$)')

    return x_fft, y_fft_plot

# plot fft and keep phase information
def amplitude_spectrum_with_phase_information(sr, s, window_function):  # sr =sampling rate #https://www.cbcity.de/die-fft-mit-python-einfach-erklaert
    import numpy as np
    import numpy.fft as fft
    import matplotlib.pyplot as plt
    from numpy import hanning 

    if window_function == 'hanning':
        hann = np.hanning(len(s))
        y_fft = fft.fft(hann*s)
    elif window_function == 'blackman': 
        blackman = np.blackman(len(s))
        y_fft = fft.fft(blackman*s)
    elif window_function == 'hamming':
        hamming = np.hamming(len(s))
        y_fft = fft.fft(hamming*s)
    elif window_function == 'none':
        y_fft = fft.fft(s)
    else:
        print('ERROR: Window function not known!') 

    n_fft = int(len(y_fft)/2+1)
    
    x_fft = np.linspace(0, sr/2, n_fft, endpoint=True)
    y_fft_plot = 2.0*(y_fft[:n_fft])/n_fft # hierbei Umwandlung von komplexen in reele Zahlen über np.abs --> haut es dann ohne np.abs hin? 

    #plt.plot(x_fft, y_fft_plot)
    #plt.xlabel('Frequency ($MHz$)')
    #plt.ylabel('Amplitude ($Unit$)')

    return x_fft, y_fft_plot



# remove specific characters of a yml-file so it can be loaded as yaml -- Danke Max :)
def change_and_load_yaml_pipeline(file, end_path, line_parameter='tgc', character1="}{", character2=";"):
    '''
    Parameters
    ----------
    file : str
        The name/path of the directory, the yaml file is in.

    line_parameter : str (optional)
        The beginning of the line, that 
        contains the invalid character.
    character : str (optional)
        The substring that should be replaced.
        Be aware to the setting of the spaces.
    '''
    import yaml
    import os
    import shutil
                
    # open file in read mode 
    stream = open(file, 'r') 
    # extract content as one string
    data = stream.read() 
    # seperate string in substrings (one substring per line)
    data_splitted = data.splitlines()
    # find line number of the specific line parameter
    for element in data_splitted:
        if line_parameter in element:
            line_number = data_splitted.index(element)
    # replace the invalid character in the requested line
    data = data.replace(data_splitted[line_number], data_splitted[line_number].replace(character1, ','))
    data = data.replace(data_splitted[line_number], data_splitted[line_number].replace(character2, ','))
    # close the input file to
    stream.close() 
    # open file in write mode
    stream = open(file, 'w') 
    # write changed data string into file (replace the whole text)
    stream.write(data)
    # close the file
    stream.close()

    # load changed yaml file
    with open(file, 'r') as stream:
        loaded_yaml = yaml.safe_load(stream)

    shutil.copy(file, end_path+".rf.yaml")

    return loaded_yaml

# remove specific characters of a yml-file so it can be loaded as yaml -- Danke Max :)
def change_and_load_yaml_display(directory, line_parameter='tgc', character1="}{", character2=";"):
    '''
    Parameters
    ----------
    file : str
        The name/path of the directory, the yaml file is in.

    line_parameter : str (optional)
        The beginning of the line, that 
        contains the invalid character.

    character : str (optional)
        The substring that should be replaced.
        Be aware to the setting of the spaces.
    '''
    import yaml
    import os
    import shutil

    for root, dirs, files in os.walk(directory):  
        for name in files:
            if name.endswith('rf.yml') or name.endswith('iq.yml') or name.endswith('rf.yaml'):
                name_length = len(name)
                file_path_1 = os.path.abspath(name)
                file = file_path_1[:-name_length]+ root +'/'+file_path_1[-name_length:]
                
    # open file in read mode 
    stream = open(file, 'r') 
    # extract content as one string
    data = stream.read() 
    # seperate string in substrings (one substring per line)
    data_splitted = data.splitlines()
    # find line number of the specific line parameter
    for element in data_splitted:
        if line_parameter in element:
            line_number = data_splitted.index(element)
    # replace the invalid character in the requested line
    data = data.replace(data_splitted[line_number], data_splitted[line_number].replace(character1, ','))
    data = data.replace(data_splitted[line_number], data_splitted[line_number].replace(character2, ','))
    # close the input file to
    stream.close() 
    # open file in write mode
    stream = open(file, 'w') 
    # write changed data string into file (replace the whole text)
    stream.write(data)
    # close the file
    stream.close()

    #load the changed yaml file
    with open(file, 'r') as stream:
        loaded_yaml = yaml.safe_load(stream)

    return loaded_yaml


def get_fibroscan_values(directory):
    import pandas as pd

    fibroscan_values = pd.read_csv(directory)
    Patient_ID = fibroscan_values['patients_id'].tolist()
    Median_E = fibroscan_values['median_e'].tolist()
    Median_CAP = fibroscan_values['median_cap'].tolist()
    IQR_E = fibroscan_values['iqr_e'].tolist()
    IQR_CAP = fibroscan_values['iqr_cap'].tolist()
    scan_conditions = fibroscan_values['conditions_of_examination'].tolist()


    return Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP, scan_conditions


def read_yaml_file(yaml_file, array):
    import yaml
    import numpy as np

    with open(yaml_file, 'r') as stream:
        yaml_file = yaml.safe_load(stream)  
    sampling_rate = float(yaml_file['sampling rate'][:-4])*1e6 # Hz
    fs = sampling_rate # Sample rate
    try:
        # if array is still 3d (scanlines, depth, frames)
        t = np.linspace(0, len(array[0,:,0]), len(array[0,:,0]))  
    except:
        # if array is already 2d (scanlines, depth)
        t = np.linspace(0, len(array[0,:]), len(array[0,:]))

    return sampling_rate, t

def stFFT(array, window_size, hop_size, fs, window_function):
    import numpy as np

    hop_size = round(window_size//hop_size)

    if hop_size == 0:
        hop_size = 1

    # add loop over all frames later on

    stft_array_all_scanlines = []

    for j in range(len(array[:,0,0])): # loop over all scanlines

        x_array = array[j,:,0]  # signal recording -> frame info = 0 -> should be changed later on

        stft_array_one_scanline = []

        # calculate stFFT for each horizontal frame and add them together -> for recording
        for i in range(0,round(len(x_array)/hop_size)):

            segment_array = x_array[i*hop_size:i*hop_size+window_size]
            if len(segment_array) == window_size:
                x_fft, y_fft_plot = amplitude_spectrum(fs, segment_array, window_function)
            
                y_fft_line = y_fft_plot.reshape(len(y_fft_plot),1)
                
                stft_array_one_scanline.append(y_fft_line)
            
        stft_array_one_scanline = np.array(stft_array_one_scanline)

        stft_array_all_scanlines.append(stft_array_one_scanline)
    
    stft_array_all_scanlines = np.array(stft_array_all_scanlines)
    stft_array_all_scanlines = np.squeeze(stft_array_all_scanlines)

    return stft_array_all_scanlines

def remove_dot_underscore_files(directory):
    import os
    for file in os.listdir(directory):
        if file.startswith('.'):
            try:
                os.remove(os.path.join(directory, file))
                os.rmdir(os.path.join(directory, file))
            except:
                print('Could not remove file/directory: ', file)

def stfft_for_all_files(start_directory, dir_redcap_survey, transducer, tgc, max_depth, min_depth, window_size, hop_size, ref_parameter, start_scanline, end_scanline):
    import os
    import numpy as np
    from tqdm import tqdm


    # calculate stFFT of all recordings 

    # get redcap values out of csv file
    Patient_ID, Median_E, Median_CAP, IQR_E, IQR_CAP = get_fibroscan_values(dir_redcap_survey)

    redcap_value = []
    frequency_slopes_all_recordings = []
    pearson_per_frequency = []
    stfft_all_recordings = []
    recordings_count = 0
    for dir in tqdm(os.listdir(start_directory)):
        for file in os.listdir(start_directory+'/'+dir):
            if transducer in file and tgc in file:
                # get array without tgc from this recording
                array = np.load(start_directory+'/'+dir+'/'+file)

                # skip arrays with less pixels than max_depth
                if len(array[0,:,0]) < max_depth:
                    break

                # read in metadata from yaml file
                yaml_file = start_directory+'/'+dir+'/'+file[:-11]+'.yaml'
                sampling_rate, fs, t = read_yaml_file(yaml_file, array)

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
                stfft = stFFT(array, window_size, hop_size, fs)
            
        stfft_all_recordings.append(stfft)

    stfft_all_recordings = np.array(stfft_all_recordings)
    # shape stfft_all_recordings: (number_of_recordings, number_of_scanlines, number_of_timeframes, number_of_frequencies)

    return stfft_all_recordings, redcap_value

# cut away unnecessary parts of DICOM arrays
def crop_dcm(data, transducer):
    import numpy as np

    try: 
        if len(data[3]>0):
            pass
    except: data = data[1]

    # L15
    if transducer.startswith('L15'):
        data = data[0,:,:,:] # [frames, x, y, rgb]
        # boundaries for image cropping
        x_min = 80
        x_max = 720
        y_min = 180
        y_max = 620
        # cut out unimportant image data
        data = data[y_min:y_max,x_min:x_max]

        # remove useless informations from clarius image
        # clarius logo => RGB-Values are different
        for y in range(0,y_max-y_min):
            for x in range(0,x_max-x_min):
                if data[y,x,0] != data[y,x,1] or data[y,x,0] != data[y,x,2]:
                    data[y,x,:] = 0.0 
        data[370:,:50] = 0.0 # sample frequency
        data[400:,610:] = 0.0 # deepth information
        data = data[:,:,0]

        #'''
        # remove black borders
        # x-axis
        x_lines_to_remove = []
        for x in range(0, len(data[:,0])):
            if np.sum(data[x,:]) == 0:
                x_lines_to_remove.append(x)
        x_lines_to_remove = np.array(x_lines_to_remove)
        if len(x_lines_to_remove) > 0:
            data = np.delete(data, x_lines_to_remove, 0)
                
        # y-axis
        y_lines_to_remove = []
        for y in range(0, len(data[0,:])):
            if np.sum(data[:,y]) == 0:
                y_lines_to_remove.append(y)
        y_lines_to_remove = np.array(y_lines_to_remove)
        if len(y_lines_to_remove) > 0:
            data = np.delete(data, y_lines_to_remove, 1)
        #'''

    # C3:
    if transducer.startswith('C3'):
        data = data[0,:,:,0] # [frames, x, y, rgb]

        # boundaries for image cropping (curved array)
        x_min = 80
        x_max = 720
        y_min = 180
        y_max = 620
        # crop important image data
        data = data[y_min:y_max,x_min:x_max]

        # remove useless informations from clarius image
        data[15:40,420:450] = 0.0 # clarius logo
        data[400:,:50] = 0.0 # sample frequency
        data[400:,610:] = 0.0 # deepth information

    return data

def linear_regression(ref_values, pred_values):
    from scipy import stats
    import numpy as np

    # calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(ref_values, pred_values)

    # calculate linear fit values
    def myfunc(x):
        return slope * x + intercept
    mymodel = list(map(myfunc, ref_values))

    return slope, intercept, r_value, p_value, mymodel

# linear or polynomial regression
def regression(mean_slopes_all_recordings, redcap_value, polynomial_degree=1):
    from sklearn.metrics import r2_score
    import numpy as np

    mean_slopes_all_recordings = np.abs(mean_slopes_all_recordings)

    model = np.polyfit(mean_slopes_all_recordings, redcap_value, polynomial_degree)
    predict = np.poly1d(model)
    x_reg = np.linspace(np.min(mean_slopes_all_recordings), np.max(mean_slopes_all_recordings), 100)
    y_reg = predict(x_reg)

    r_squared = r2_score(redcap_value, predict(mean_slopes_all_recordings))
    r = np.sqrt(r_squared)

    return x_reg, y_reg, r, r_squared

def get_attenuation_best_freq(transducer, version, data_directory, target_directory):
    import numpy as np
    from scipy import stats

    mean_slopes_all_recordings_freq = np.load(data_directory+'/mean_slopes_all_recordings_freqspec_'+transducer+'_'+version+'.npy')
    target_dict = np.load(target_directory+'/targets_CAP_freqspec_'+transducer+'_'+version+'.npy', allow_pickle='TRUE')

    redcap_value = list(target_dict.item().values())

    # calculate pearson correlation coefficient between redcap value and frequency slope
    pearson_per_frequency = []
    for frequency in range(len(mean_slopes_all_recordings_freq[0,:])):
        slope, intercept, r_value, p_value, std_err = stats.linregress(redcap_value, np.array(mean_slopes_all_recordings_freq)[:,frequency])
        pearson_per_frequency.append(r_value)  

    # attenuation values best freq. section
    attenuation_values_best_freq = mean_slopes_all_recordings_freq[:,np.argmax(np.abs(pearson_per_frequency))]

    # perform regression for best frequency section
    best_slope, best_intercept, best_r_value, best_p_value, best_mymodel = linear_regression(attenuation_values_best_freq,redcap_value)

    return pearson_per_frequency, attenuation_values_best_freq,redcap_value, best_slope, best_intercept, best_r_value, best_p_value, best_mymodel

def detect_and_remove_outliers(Patient_ID, array,redcap_value, removal_based_on, threshold):
    import numpy as np

    # define borders for outliers
    if removal_based_on == 'IQR':
        if threshold == 'lower':
            lower_th = 0.25
            upper_th = 0.75
        elif threshold == 'higher':
            lower_th = 0.1
            upper_th = 0.9
        quartile1 = np.quantile(array, lower_th)
        quartile3 = np.quantile(array, upper_th)
        iqr = quartile3 - quartile1
        upper_limit = quartile3 + 1.5 * iqr
        lower_limit = quartile1 - 1.5 * iqr
    elif removal_based_on == 'std':
        if threshold == 'lower':
            nr_stds = 2
        elif threshold == 'higher':
            nr_stds = 3
        std = np.std(array)
        upper_limit = np.mean(array) + nr_stds * std
        lower_limit = np.mean(array) - nr_stds * std
    # detect outliers
    list_outliers = []
    id_outliers = []
    for element in range(len(array)-1):
        if array[element] > upper_limit or array[element] < lower_limit:
            list_outliers.append(element)
            id_outliers.append(Patient_ID[element])

   
    n_outliers = len(list_outliers)

    # remove outliers
    evaluation_parameters_no_outliers = np.delete(array, list_outliers)
    redcap_value_outlier_removal = np.delete(redcap_value, list_outliers)

    return evaluation_parameters_no_outliers, redcap_value_outlier_removal, n_outliers, id_outliers

def plot_one_lizzi_feleppa_results(dict_results, lizzi_feleppa_parameter,ref_parameter, transducer, Phantom, safe_to_notion, database_id, version, notion_cropping_properties, notion_window_func, notion_transducer):
    import matplotlib.pyplot as plt
    import numpy as np
    import functions.def_notion_API as notion_def
    # get values out of dict_results
    redcap_value = dict_results[ref_parameter]
    lizzi_feleppa_value = dict_results[lizzi_feleppa_parameter]

    # linear regression of parameter and redcap values
    slope, intercept, r_value, p_value, mymodel = linear_regression(redcap_value,lizzi_feleppa_value)

    print('Pearson correlation coefficient '+ lizzi_feleppa_parameter +' vs. '+ref_parameter+': ', round(r_value,3))
    print('p-value: ', round(p_value,3))
    # calculate mean and std of lizzi_feleppa_value
    mean_lizzi_feleppa_value = np.mean(lizzi_feleppa_value)
    std_lizzi_feleppa_value = np.std(lizzi_feleppa_value)
    print('Mean '+lizzi_feleppa_parameter+': ', mean_lizzi_feleppa_value)
    print('Std '+lizzi_feleppa_parameter+': ',std_lizzi_feleppa_value)

   # plot
    plt.figure(figsize=(5,5))
    plt.scatter(lizzi_feleppa_value, redcap_value, color = '#00305D')
    plt.plot(mymodel,redcap_value, color = 'orange')
    plt.ylabel(ref_parameter, fontsize = 12)
    plt.xlabel('Lizzi-Feleppa '+lizzi_feleppa_parameter, fontsize = 12)
    plt.locator_params(axis='x', nbins=7)
    plt.title(transducer[:-6], fontweight = 'bold',fontsize = 13)#'Correlation between CAP value and '+lizzi_feleppa_parameter)
    plt.show()

    # safe results to notion
    if Phantom == 'Wat_Phan':
        notion_evaluation_parameter = lizzi_feleppa_parameter+' Lizzi Feleppa \nCalibration: '+Phantom
    else: 
        notion_evaluation_parameter = lizzi_feleppa_parameter+' Lizzi Feleppa \nCalibration: '+Phantom
    notion_results = str('Pearson coef.: '+str(round(np.abs(r_value), 3))+ '\n(p-value: '+ str(round(p_value, 3))+')')
    notion_version = str(version+'\n('+str(len(redcap_value))+' Pat.)')
    notion_def.safe_to_notion_def(notion_version, notion_evaluation_parameter, notion_results, notion_transducer, notion_cropping_properties,notion_window_func, safe_to_notion, database_id)

    return r_value, p_value, mean_lizzi_feleppa_value, std_lizzi_feleppa_value

def plot_all_lizzi_feleppa_results(result_directory, ref_parameter, transducer, Phantom, version):
    import matplotlib.pyplot as plt
    import numpy as np
    import functions.def_notion_API as notion_def

    # get C3 values 
    transducer = 'C3_large'
    dict_results_C3 = np.load(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+transducer+'.npy', allow_pickle = True).item()
    # get values out of dict_results
    redcap_value = dict_results_C3[ref_parameter]
    C3_lizzi_feleppa_MBF = dict_results_C3['MBF']
    C3_lizzi_feleppa_intercept = dict_results_C3['intercept']
    C3_lizzi_feleppa_slope = dict_results_C3['slope']

    # linear regression of parameters and redcap values
    slope, intercept, r_value_MBF_C3, p_value_MBF_C3, mymodel_MBF_C3 = linear_regression(redcap_value,C3_lizzi_feleppa_MBF)
    slope, intercept, r_value_intercept_C3, p_value_intercept_C3, mymodel_intercept_C3 = linear_regression(redcap_value,C3_lizzi_feleppa_intercept)
    slope, intercept, r_value_slope_C3, p_value_slope_C3, mymodel_slope_C3 = linear_regression(redcap_value,C3_lizzi_feleppa_slope)

    mean_MBF_C3 = np.mean(C3_lizzi_feleppa_MBF)
    std_MBF_C3 = np.std(C3_lizzi_feleppa_MBF)
    mean_intercept_C3 = np.mean(C3_lizzi_feleppa_intercept)
    std_intercept_C3 = np.std(C3_lizzi_feleppa_intercept)
    mean_slope_C3 = np.mean(C3_lizzi_feleppa_slope)
    std_slope_C3 = np.std(C3_lizzi_feleppa_slope)

    print('Pearson correlation coefficient MBF vs. '+ref_parameter+': ', round(r_value_MBF_C3,3), ' (p-value: ', round(p_value_MBF_C3,3),', mean MBF: ', round(mean_MBF_C3,3), ', std MBF: ', round(std_MBF_C3,3),')')
    print('Pearson correlation coefficient intercept vs. '+ref_parameter+': ', round(r_value_intercept_C3,3), ' (p-value: ', round(p_value_intercept_C3,3),', mean intercept: ', round(mean_intercept_C3,3), ', std intercept: ', round(std_intercept_C3,3),')')
    print('Pearson correlation coefficient slope vs. '+ref_parameter+': ', round(r_value_slope_C3,3), ' (p-value: ', round(p_value_slope_C3,3),', mean slope: ', round(mean_slope_C3,8), ', std slope: ', round(std_slope_C3,8),')')
    print('')

    # get L15 values
    transducer = 'L15_large'
    dict_results_L15 = np.load(result_directory+'/Lizzi_Feleppa_Results_'+version+'_'+transducer+'.npy', allow_pickle = True).item()
    # get values out of dict_results
    redcap_value = dict_results_L15[ref_parameter]
    L15_lizzi_feleppa_MBF = dict_results_L15['MBF']
    L15_lizzi_feleppa_intercept = dict_results_L15['intercept']
    L15_lizzi_feleppa_slope = dict_results_L15['slope']

    # linear regression of parameters and redcap values
    slope, intercept, r_value_MBF_L15, p_value_MBF_L15, mymodel_MBF_L15 = linear_regression(redcap_value,L15_lizzi_feleppa_MBF)
    slope, intercept, r_value_intercept_L15, p_value_intercept_L15, mymodel_intercept_L15 = linear_regression(redcap_value,L15_lizzi_feleppa_intercept)
    slope, intercept, r_value_slope_L15, p_value_slope_L15, mymodel_slope_L15 = linear_regression(redcap_value,L15_lizzi_feleppa_slope)

    mean_MBF_L15 = np.mean(L15_lizzi_feleppa_MBF)
    std_MBF_L15 = np.std(L15_lizzi_feleppa_MBF)
    mean_intercept_L15 = np.mean(L15_lizzi_feleppa_intercept)
    std_intercept_L15 = np.std(L15_lizzi_feleppa_intercept)
    mean_slope_L15 = np.mean(L15_lizzi_feleppa_slope)
    std_slope_L15 = np.std(L15_lizzi_feleppa_slope)

    print('Pearson correlation coefficient MBF vs. '+ref_parameter+': ', round(r_value_MBF_L15,3), ' (p-value: ', round(p_value_MBF_L15,3),', mean MBF: ', round(mean_MBF_L15,3), ', std MBF: ', round(std_MBF_L15,3),')')
    print('Pearson correlation coefficient intercept vs. '+ref_parameter+': ', round(r_value_intercept_L15,3), ' (p-value: ', round(p_value_intercept_L15,3),', mean intercept: ', round(mean_intercept_L15,3), ', std intercept: ', round(std_intercept_L15,3),')')
    print('Pearson correlation coefficient slope vs. '+ref_parameter+': ', round(r_value_slope_L15,3), ' (p-value: ', round(p_value_slope_L15,3),', mean slope: ', round(mean_slope_L15,8), ', std slope: ', round(std_slope_L15,8),')')

    # plot all results into one figure (3x2 subplots (C3 left, L15 right))
    fontsize_labels = 12
    fontdict_title = {'fontsize': 13, 'fontweight': 'bold'}

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle('Correlation between CAP value and Lizzi-Feleppa parameters', fontsize = 15, fontweight = 'bold')

    # C3
    axs[0, 0].scatter(C3_lizzi_feleppa_MBF, redcap_value, color = '#00305D')
    axs[0, 0].plot(mymodel_MBF_C3,redcap_value, color = 'orange')
    axs[0, 0].set_ylabel(ref_parameter, fontsize = fontsize_labels)
    axs[0, 0].set_xlabel('Lizzi-Feleppa MBF', fontsize = fontsize_labels)
    axs[0, 0].locator_params(axis='x', nbins=7)
    axs[0, 0].set_title('Lizzi-Feleppa MBF C3', fontweight = 'bold',fontdict = fontdict_title)
    axs[1, 0].scatter(C3_lizzi_feleppa_intercept, redcap_value, color = '#00305D')
    axs[1, 0].plot(mymodel_intercept_C3,redcap_value, color = 'orange')
    axs[1, 0].set_ylabel(ref_parameter, fontsize = fontsize_labels)
    axs[1, 0].set_xlabel('Lizzi-Feleppa intercept', fontsize = fontsize_labels)
    axs[1, 0].locator_params(axis='x', nbins=7)
    axs[1, 0].set_title('Lizzi-Feleppa intercept C3', fontweight = 'bold',fontdict = fontdict_title)
    axs[2, 0].scatter(C3_lizzi_feleppa_slope, redcap_value, color = '#00305D')
    axs[2, 0].plot(mymodel_slope_C3,redcap_value, color = 'orange')
    axs[2, 0].set_ylabel(ref_parameter, fontsize = fontsize_labels)
    axs[2, 0].set_xlabel('Lizzi-Feleppa slope', fontsize = fontsize_labels)
    axs[2, 0].locator_params(axis='x', nbins=7)
    axs[2, 0].set_title('Lizzi-Feleppa slope C3', fontweight = 'bold',fontdict = fontdict_title)

    # L15
    axs[0, 1].scatter(L15_lizzi_feleppa_MBF, redcap_value, color = '#00305D')
    axs[0, 1].plot(mymodel_MBF_L15,redcap_value, color = 'orange')
    axs[0, 1].set_ylabel(ref_parameter, fontsize = fontsize_labels)
    axs[0, 1].set_xlabel('Lizzi-Feleppa MBF', fontsize = fontsize_labels)
    axs[0, 1].locator_params(axis='x', nbins=7)
    axs[0, 1].set_title('Lizzi-Feleppa MBF L15', fontweight = 'bold',fontdict = fontdict_title)
    axs[1, 1].scatter(L15_lizzi_feleppa_intercept, redcap_value, color = '#00305D')
    axs[1, 1].plot(mymodel_intercept_L15,redcap_value, color = 'orange')
    axs[1, 1].set_ylabel(ref_parameter, fontsize = fontsize_labels)
    axs[1, 1].set_xlabel('Lizzi-Feleppa intercept', fontsize = fontsize_labels)
    axs[1, 1].locator_params(axis='x', nbins=7)
    axs[1, 1].set_title('Lizzi-Feleppa intercept L15', fontweight = 'bold',fontdict = fontdict_title)
    axs[2, 1].scatter(L15_lizzi_feleppa_slope, redcap_value, color = '#00305D')
    axs[2, 1].plot(mymodel_slope_L15,redcap_value, color = 'orange')
    axs[2, 1].set_ylabel(ref_parameter, fontsize = fontsize_labels)
    axs[2, 1].set_xlabel('Lizzi-Feleppa slope', fontsize = fontsize_labels)
    axs[2, 1].locator_params(axis='x', nbins=7)
    axs[2, 1].set_title('Lizzi-Feleppa slope L15', fontweight = 'bold',fontdict = fontdict_title)

    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

    plt.show()




