# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../') # necessary to import functions from other directory
import functions.definitions as definitions
import yaml 
import shutil
import tarfile
import re
import math
from tqdm.notebook import tqdm


# %%
# Chose start_directory and end_directory (conains unsorted ultrasound data) and end_directory (directory where sorted and converted data should be stored)

directory = '/Volumes/Extreme_SSD/Demo_6G/Start' 
# %%
# unpack all tar-files in the directory

for root, dirs, files in tqdm(os.walk(directory)):
    for name in files:
        try:
            if name.startswith('._'):
                os.remove(os.path.join(root, name))
                
            if name.endswith('.tar') and not name.startswith('._'):
                tar = tarfile.open(os.path.join(root, name))
                tar.extractall(path = root+'/'+name[:-4])
                tar.close()
                os.remove(os.path.join(root, name)) 
        except:
            print('Error in: ', os.path.join(root, name))

# %%
# decompress all lzo-files in the start_directory

for root, dirs, files in tqdm(os.walk(directory)):  
    for name in files:
        if name.endswith(('rf.raw.lzo','env.raw.lzo')):
            name_length = len(name)
            file_path = os.path.join(root, name)
            os.system('lzop -d {}'.format(file_path))

# %%
# convert data into arrays and safe them in extra directory
changed_directory = 'changed' # kannst du so lassen
for dir in os.listdir(directory):       

    os.mkdir(directory+"/"+changed_directory)
    #search, load and safe yaml
    for file in os.listdir(directory+"/"+dir):  
        if file.endswith('rf.yml'):
            # open file in read mode 
            stream = open(directory+"/"+dir+"/"+file, 'r') 
            # extract content as one string
            data = stream.read() 
            # seperate string in substrings (one substring per line)
            data_splitted = data.splitlines()
            # find line number of the specific line parameter
            line_parameter='tgc'
            character="}{"
            for element in data_splitted:
                if line_parameter in element:
                    line_number = data_splitted.index(element)
            # replace the invalid character in the requested line
            data = data.replace(data_splitted[line_number], data_splitted[line_number].replace(character, ';'))
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

            sampling_rate = int(loaded_yaml['sampling rate'][:-4])*1e6 #Hz
            imaging_depth = float(loaded_yaml['imaging depth'][:-3]) #mm
            if 'vsound'  in loaded_yaml:
                soundspeed = int(loaded_yaml['vsound'][:-4]) #m/s
            else:
                soundspeed = 1450 #m/s

            # copy changed yaml
            shutil.copy(directory+"/"+dir+"/"+file, directory+"/"+changed_directory+"/"+dir+'.rf.yaml')

    #search rf-file, extract rf-array and save it 
    for file in os.listdir(directory+"/"+dir):
        if file.endswith('rf.raw'):
            rf_filename = directory+"/"+dir+"/"+file
            hdr, timestamps, rf_array = definitions.read_rf(rf_filename)
            np.save(directory+"/"+changed_directory+"/"+dir+".rf.npy", rf_array)
    
    #search tgc-file and safe it in new directory
    for file in os.listdir(directory+"/"+dir):
        if file.endswith('tgc.yml'):   
            shutil.copy(directory+"/"+dir+"/"+file, directory+"/"+changed_directory+"/"+dir+'.rf.tgc')
    
    shutil.rmtree(directory+"/"+dir)

            
# %%
# rename files according to the roi-size and used scanner
for dir in os.listdir(directory):
    for file in os.listdir(directory+"/"+dir):
        yaml_files_with_same_name = 0
        if file.endswith(".yaml") and file.startswith("raw"):
            yaml_file = directory+"/"+dir+"/"+file
            with open(yaml_file, 'r') as stream:
                loaded_yaml = yaml.safe_load(stream)
            sampling_rate = int(loaded_yaml['sampling rate'][:-4])*1e6 #Hz
            imaging_depth = float(loaded_yaml['imaging depth'][:-3]) #mm
            delay_samples = int(loaded_yaml['delay samples'])
            number_of_lines = dict(loaded_yaml['size'])['number of lines'] #number of scan-lines horizontal
            samples_per_line = dict(loaded_yaml['size'])['samples per line'] #number of samples in one line
            iso_time_date = str(loaded_yaml['iso time/date'])

            if sampling_rate == 15e6:
                scanner = "C3"
            else: 
                scanner = "L15"
            
            if number_of_lines > 185:
                size = "large"
            elif number_of_lines == 24:
                size = "small"
            else:
                size = "medium"
            
            # check if there allready is a file with same recording modalities
            if os.path.isfile(directory+"/"+dir+"/"+scanner+"_"+size+".rf.yaml"):
                yaml_files_with_same_name += 1

            # rename yaml-file
            if yaml_files_with_same_name > 0:
                new_yaml_name = directory+"/"+dir+"/"+scanner+"_"+size+"_"+str(yaml_files_with_same_name)+".rf.yaml"
            else:
                new_yaml_name = directory+"/"+dir+"/"+scanner+"_"+size+".rf.yaml"
            os.rename(yaml_file,new_yaml_name)

            # rename according rf- and dcm-files
            recognition = file[:7]

            # rename rf-file
            try:
                os.rename(directory+"/"+dir+"/"+recognition+".rf.npy",new_yaml_name[:-5]+".npy")
            except:
                print('Could not rename rf-file for '+dir)
            
            #rename tgc-file
            try:
                os.rename(directory+"/"+dir+"/"+recognition+".rf.tgc",new_yaml_name[:-5]+".tgc")
            except:
                print('Could not rename tgc-file for '+dir)

            # rename dcm-file
            for dcm_file in os.listdir(directory+"/"+dir):
                if dcm_file.endswith(".dcm.npy"):
                    # load dcm data
                    dcm_data = np.load(directory+"/"+dir+"/"+dcm_file, allow_pickle=True)
                    # compare dcm-time and yaml-time
                    dcm_time = dcm_data[0]
                    yaml_time = int(iso_time_date[11:13]+iso_time_date[14:16]+iso_time_date[17:19])

                    if np.abs(dcm_time-yaml_time) <= 2:
                        os.rename(directory+"/"+dir+"/"+dcm_file,new_yaml_name[:-8]+".dcm.npy")

    # remove dcm-files without corresponding yaml-file
    for file in os.listdir(directory+"/"+dir):
        if file.startswith("1_") or file.startswith("0_"):
            os.remove(directory+"/"+dir+"/"+file)
    
    # remove tgc-files without corresponding yaml-file
    for file in os.listdir(directory+"/"+dir):
        if file.startswith('raw'):
            os.remove(directory+"/"+dir+"/"+file)

# %%
# cleansing of the signal from the tgc (for recordings with auto gain ON)

for dir in tqdm(os.listdir(directory)):
    if os.path.isdir(directory+'/'+dir):
        for file in  os.listdir(directory+'/'+dir):
            # try to remove files that start with "._"
            if file.startswith("._"):
                try:
                    os.remove(directory+'/'+dir+'/'+file)
                    print('File' + dir+'/'+file+' started with "._" and was removed.')
                except:
                    print('ERROR: File starts with "._": '+dir+'/'+file+ 'And could not be deleted.')
                    
            if file.endswith('rf.tgc') and 'large' in file and not os.path.isfile(directory+'/'+dir+'/'+file[:-4]+'_no_tgc.npy'):
                # create list with step-informations (just for plotting)
                step_list = []

                # get meta-information out of yaml and calculate pixel_per_millimeter
                yaml_file = directory+'/'+dir+'/'+file[:-3]+'yaml'
                with open(yaml_file, 'r') as stream:
                    yaml_file = yaml.safe_load(stream)  
                sampling_rate = float(yaml_file['sampling rate'][:-4])*1e6 # Hz
                imaging_depth = float(yaml_file['imaging depth'][:-3])
                if 'delay samples' in yaml_file: 
                    delay_samples = int(yaml_file['delay samples'])
                else: 
                    delay_samples = 0
                if 'vsound'  in yaml_file:
                    soundspeed = int(yaml_file['vsound'][:-4]) #m/s
                else:
                    soundspeed = 1540 #m/s
                time_per_pixel = 1/sampling_rate
                centimeter_per_one_pixel = time_per_pixel*(soundspeed*100)/2
                pixel_per_millimeter = 1/(centimeter_per_one_pixel*10)

                # define which array to change
                array = np.load(directory+"/"+dir+'/'+file[:-3]+'npy').astype(float)
                # if there are delay samples: change the array, so it gets changed at the right places
                #if delay_samples>0:
                #    delay_array = np.zeros((len(array[:,0,0]),delay_samples,len(array[0,0,:]))) # array, with zeros in front of normal array
                #    array = np.append(delay_array, array, axis=1)
                    #back_array = 1


                # define where to get tgc-values
                tgc = open(directory+'/'+dir+'/'+file,'r')
                lines = tgc.readlines()
                tgc.close()
                # check if the file has old or new tgc-sorting
                if len(lines)-4 == len(array[0,0,:]): #old sorting
                    important_lines = [idx for idx in lines if idx[0:9] == 'timestamp']
                    number_of_lines = len(important_lines)+4 # number_of_lines => number of frames in that recording
                    tgc_sorting = 0
                else: # new sorting
                    frame_number_count = 0
                    for line in lines:
                        if 'timestamp' in line:
                            frame_number_count += 1
                    number_of_lines = len(array[0,0,:])
                    tgc_sorting = 1
                

                frame_number = 0
                # take the tgc-values from one line out of the tgc-file and store it in 'matches'
                # do this for every line in the tgc-file 
                # one line in the tgc represents one frame in the matching array  
                for i in range(5,number_of_lines):
                    #define new_array (recalculated array will be safed here) bzw. set it back to 0
                    new_array = {}

                    if tgc_sorting == 0:
                        line = lines[i]
                        i += 1
                        matches = re.findall('[\d,\.]*(?=mm)|[\d,\.]*(?=dB)', line)[::2] # irgendnen scheiss, den Edgar sich überlegt hat (Reg Ex heißt die Methode-> regular expression)
                        # matches => List with alternating range (in cm) and matching amplification (in db) 
                    elif tgc_sorting == 1:
                        lines_with_timestamp =[]
                        for j,line in enumerate(lines):
                            if 'timestamp' in line:
                                lines_with_timestamp.append(j)
                        len_matches = lines_with_timestamp[i-4] - lines_with_timestamp[i-5]-1 # count how many tgc-informations are in the frame
                        line = str(lines[lines_with_timestamp[i-5]:lines_with_timestamp[i-4]])
                        matches = re.findall('[\d,\.]*(?=mm)|[\d,\.]*(?=dB)', line)[::2]
                    else: print('ERROR:'+dir+'/'+file)


                    # changed_pixels= quantity of pixels that where allready changed in this line (0 in the beginning)
                    changed_pixels = 0
                    matches_count = 0
                    k = 0

                    # INFOS: as far as I know, the tgc is allways in steps, that are a multiple of the first step and the last step is the same as the first one
                    # so one could calculate the step-width based on this assumption
                    # divide the given pixel-length of the frame by len(matches)
                    # len(matches)/2+1 = number_of_steps -> in matches you have 2 values per step (steplength(mm),amplification(dB)) so you have to divide by 2 but you have one step more, than matches (because array starts BEFORE the first step and ends AFTER the last one) so you have to add 1 again
                    len_matches = len(matches)
                    number_of_steps = (len_matches/2)+1                    
                    
                    # calculate how many pixels lay in the first tgc-frame (no tgc was made here)
                    # makes sense to define pixel-count of first and last step (=length of array/len_matches) because the other steps are just twize as big
                    # if you take the first and last step together the number of steps = matches/2 
                    # so to get the big_stepwidth you would take: length of arrray/(len_matches/2)
                    # because we want the small_stepwidth first, we take: length of arrray/len_matches
                    small_stepwidth = math.ceil(len(array[0,:,0])/len_matches) 
                    big_stepwidth = math.ceil((len(array[0,:,0])/len_matches)*2)

                    # define how many pixels lay in the first tgc step 
                    pixels_to_change = small_stepwidth
                    # define the array that have to be changed
                    array_to_change = array[:,changed_pixels:(pixels_to_change+changed_pixels),frame_number]
                    # remove the tgc 
                    changed_array = {}

                    if float(matches[0]) > 0:
                        # define the steps of the amplification in this hop 
                        amplification_change = float(matches[matches_count+1]) # difference between current amplification value and next amplification value => since this is the first value, there is no value to substract
                        if amplification_change > 0:
                            # linear increase from one to the next value in the tgc-file
                            amplification_steps =  np.arange(0,float(matches[matches_count+1]),amplification_change/small_stepwidth, dtype = float)
                        else: 
                            # if both steps are even, take one of them and set up an array with this value and len(stepwidth)
                            amplification_steps = np.zeros(small_stepwidth)
                            amplification_steps = np.place(amplification_steps, amplification_steps < 1, float(matches[matches_count+1]))                   
                    else:
                        # first step goes from 0mm to next step 
                        px_per_mm = int(round(len(array[0,:,0])/imaging_depth))
                        array_to_change = array[:,round(float((matches[0]))*px_per_mm):round(float((matches[2]))*px_per_mm),frame_number]

                        amplification_change = float(matches[3])-float(matches[1])
                        amplification_steps =  np.arange(float(matches[1]),float(matches[3]),amplification_change/len(array_to_change[0,:]), dtype = float)

                        k += 1
                        matches_count += 2
                        pixels_to_change = len(array_to_change[0,:])

                    for n in range(0,len(array_to_change[0,:])):

                        # amplification (in dB)= 20*log(amplificated value/original value)
                        changing_array = np.divide(array_to_change[:,n],10.0**((float(amplification_steps[n]))/20))
                        if len(changed_array) == 0:
                            changed_array =  changing_array
                        else:
                            changed_array = np.dstack((changed_array,changing_array))
                    changed_array = changed_array[0,:,:]
                    if len(new_array) < 1: 
                        new_array = changed_array

                    changed_pixels = changed_pixels+pixels_to_change
                    k += 1
                    matches_count += 2
                    
                    
                    # recalculation for pixels between first and last step (should all have big_stepwidth)
                    for k in range(k,int(len(matches)/2)):
                        # define how many pixels lay in the next tgc
                        pixels_to_change = big_stepwidth
                        # define the array that have to be changed
                        array_to_change = array[:,changed_pixels:(pixels_to_change+changed_pixels),frame_number]
                        # remove the tgc 
                        changed_array = {}

                        # define the steps of the amplification in this hop 
                        amplification_change = float(matches[matches_count+1])-float(matches[matches_count-1]) # difference between current amplification value and next amplification value
                        if amplification_change > 0:
                            # linear increase from one to the next value in the tgc-file
                            amplification_steps =  np.arange(float(matches[matches_count-1]),float(matches[matches_count+1]),amplification_change/big_stepwidth, dtype = float)
                        else: 
                            # if both steps are even, take one of them and set up an array with this value and len(stepwidth)
                            amplification_steps = np.zeros(big_stepwidth)
                            amplification_steps = amplification_steps.astype(np.float)
                            amplification_steps[amplification_steps == 0] = matches[matches_count+1]

                        # change values for every depth frame     
                        for n in range(0,len(array_to_change[0,:])):

                            # amplification (in dB)= 20*log(amplificated value/original value)
                            changing_array = np.divide(array_to_change[:,n],10.0**((float(amplification_steps[n]))/20))
                            if len(changed_array) == 0:
                                changed_array =  changing_array
                            else:
                                changed_array = np.dstack((changed_array,changing_array))
                        changed_array = changed_array[0,:,:]
                        # append array for the array with changed values
                        new_array = np.append(new_array,changed_array, axis = 1)

                        changed_pixels = changed_pixels+pixels_to_change
                        k += 1
                        matches_count += 2

                    # tgc recalculation for last step
                    # change the rest of the array
                    array_to_change = array[:,changed_pixels:,frame_number]
                    # remove the tgc 
                    # amplification (in dB)= 20*log(amplificated value/original value)
                    changed_array = np.divide(array_to_change,10.0**((float(matches[matches_count-1]))/20))
                    # append array for the array with changed values
                    new_array = np.append(new_array,changed_array, axis = 1)
                    
                    # change one frame in array to the values of new_array
                    array[:,:,frame_number] = new_array
                    
                    # change frame_number, so in the next loop the next frame will be calculated
                    frame_number += 1

                    # fill step_list with informations about steps
                    information_list = []
                    for l in range(0,k+1):
                        if l == 0:
                            information_list.append(small_stepwidth)
                        elif l == k:
                            information_list.append(len(new_array[0,:]))
                        else:
                            information_list.append(big_stepwidth+information_list[l-1])
                # insert information_list into tgc-file
                tgc = open(directory+'/'+dir+'/'+file,'r')
                textdata = tgc.read()
                tgc.close()
                if not 'list with step informations' in textdata:
                    try:
                        textdata = textdata + '\n\nlist with step informations for each frame\n' + str(information_list)
                        tgc = open(directory+'/'+dir+'/'+file,'w')
                        tgc.write(textdata)
                        tgc.close()
                    except:
                        print('could not write information_list into tgc-file')
                
                # save the array without tgc in new file
                np.save(directory+"/"+dir+'/'+file[:-4]+'_no_tgc.npy',array)



#%%
# QUS Evaluation
transducer = 'L15'
number_of_scanlines = 192
start_scanline = 0
min_depth = 687
max_depth = 2704
window_size = 100
hop_size = 1
frequency_section = 49

end_scanline = start_scanline+number_of_scanlines

for dir in os.listdir(directory): 
    for file in os.listdir(directory+'/'+dir):
        if transducer in file and 'no_tgc' in file:
            array = np.load(directory+'/'+dir+'/'+file)

            yaml_file = directory+'/'+dir+'/'+'L15_large.rf.yaml'
            sampling_rate, t = definitions.read_yaml_file(yaml_file, array)

            # apply min- and max_depth
            array = array[:,min_depth:max_depth,:]

            #remove the sides from array 
            array = array[start_scanline:end_scanline,:,:]
            
            #calculate stFFT of recording
            stfft = definitions.stFFT(array, window_size, hop_size, sampling_rate)

            # extract one frequency section
            scanline_one_frequency_section = stfft[:,:,frequency_section]

            # bring stFFT to log scale (to get linear regression)
            scanline_one_frequency_section = np.log(scanline_one_frequency_section+1e-10)
            # remove nan values
            scanline_one_frequency_section = np.nan_to_num(scanline_one_frequency_section)

            slopes_per_scanline = []
            for scanline in range(0,len(scanline_one_frequency_section[:,0])):
                # get slope of stFFT
                # slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(scanline_one_frequency_section[scanline,:])), scanline_one_frequency_section[scanline,:])
                slope = np.polyfit(np.arange(len(scanline_one_frequency_section[scanline,:])), scanline_one_frequency_section[scanline,:],1)[0]
                slopes_per_scanline.append(slope)
slopes_per_scanline = np.array(slopes_per_scanline)

# calculate mean slope over all scanlines
mean_slope = np.abs(np.mean(slopes_per_scanline))

# model function
model = np.array([1136.86503027,  -32.67724409])  

predict = np.poly1d(model)
x_lin_reg = np.linspace((100-model[1])/model[0], (400-model[1])/model[0], 100)
y_lin_reg = predict(x_lin_reg)

pred_CAP = predict(mean_slope)

# define borders for CAP (100 to 400)
if pred_CAP < 100:
    pred_CAP = 100
elif pred_CAP > 400:
    pred_CAP = 400

# plot
# plt.figure()
# plt.plot(x_lin_reg, y_lin_reg)
# plt.scatter(mean_slope, pred_CAP, color = 'red')
# plt.xlabel('Atteuation at best frequency')
# plt.ylabel('CAP')
# plt.show()

print('Predicted CAP value: ', pred_CAP)

# remove the recording in end_directory
shutil.rmtree(directory+'/'+dir)
# %%
