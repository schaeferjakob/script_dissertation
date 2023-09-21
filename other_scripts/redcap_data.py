#%%
import pandas as pd
import statistics
import sys
sys.path.append('../') # necessary to import functions from other directory
import functions.def_notion_API as notion_df
import numpy as np
import os
import matplotlib.pyplot as plt

#%%
# version of the survey
version = '230330'

# directory with data to evaluate
dir_eval = '/Volumes/Extreme_SSD/Aufnahmen_vorverarbeitet/C3_and_L15_without_TGC_'+version

# get results out of RedCap-Survey (csv)
dir_redcap_survey = '/Users/jakobschaefer/Documents/REDCap_Survey_'+version+'.csv'

redcap_survey = pd.read_csv(dir_redcap_survey)
redcap_survey = redcap_survey.set_index('patients_id')

# sort out recordings without reference data (without median_cap)
redcap_survey = redcap_survey.dropna(subset=['median_cap'])

#%%
# read out parameters from survey, calculate median and stdev and store in demographics_dict
# parameters to evaluate:
parameter_list = ['patient_age', 'patient_weight', 'patient_size', 'median_cap', 'iqr_cap']

# remove patients, that are not in the evaluation directory
for patient_id in redcap_survey.index:
    if not os.path.exists(dir_eval+'/'+patient_id):
        redcap_survey = redcap_survey.drop(patient_id)


# age 
list_patient_age = redcap_survey['patient_age']
median_age = statistics.median(list_patient_age)
mean_age = statistics.mean(list_patient_age)
sd_age = statistics.stdev(list_patient_age)

# sex
sex = redcap_survey['patient_sex'].tolist()
n_male = sex.count(1)
n_female = sex.count(2)

# split up male and female patients
male_patients = redcap_survey.loc[(redcap_survey['patient_sex']==1), parameter_list]
female_patients = redcap_survey.loc[(redcap_survey['patient_sex']==2), parameter_list]

heading_male = 'Men (n='+str(n_male)+')'
heading_female = 'Women (n='+str(n_female)+')'

demographics_df = pd.DataFrame()

def get_parameters(parameter, sub_variable, heading_male, heading_female, demographics_df):
    m_values = male_patients[sub_variable].tolist()
    m_values = [item for item in m_values if not(pd.isnull(item)) == True]
    m_mean = str(round(statistics.mean(m_values), 1))
    m_stdev = str(round(statistics.stdev(m_values), 1))

    f_values = female_patients[sub_variable].tolist()
    f_values = [item for item in f_values if not(pd.isnull(item)) == True]
    f_mean = str(round(statistics.mean(f_values), 1))
    f_stdev = str(round(statistics.stdev(f_values), 1))

    # create a demographics_df to store demographic data
    new_row = {'Parameter':[parameter], heading_male: [m_mean+' ± '+m_stdev], heading_female: [f_mean+' ± '+f_stdev]}
    demographics_df = demographics_df.append(pd.DataFrame(new_row), ignore_index=True)

    return demographics_df

# age
parameter = 'Age (y)'
sub_variable = 'patient_age'
demographics_df = get_parameters(parameter, sub_variable, heading_male, heading_female, demographics_df)

# weight
parameter = 'Weight (kg)'
sub_variable = 'patient_weight'
demographics_df = get_parameters(parameter, sub_variable, heading_male, heading_female, demographics_df)

# size
parameter = 'Size (cm)'
sub_variable = 'patient_size'
demographics_df = get_parameters(parameter, sub_variable, heading_male, heading_female, demographics_df)

# bmi (kg/m2)
parameter = 'BMI (kg/m2)'
sub_variable = 'bmi'
#male
m_size = male_patients['patient_size'].tolist()
m_size = [item for item in m_size if not(pd.isnull(item)) == True]
m_weight = male_patients['patient_weight'].tolist()
m_weight = [item for item in m_weight if not(pd.isnull(item)) == True]
m_bmi = [m_weight[i]/(m_size[i]/100)**2 for i in range(len(m_size))]
m_mean_bmi = str(round(statistics.mean(m_bmi), 1))
m_stdev_bmi = str(round(statistics.stdev(m_bmi), 1))
#female
f_size = female_patients['patient_size'].tolist()
f_size = [item for item in f_size if not(pd.isnull(item)) == True]
f_weight = female_patients['patient_weight'].tolist()
f_weight = [item for item in f_weight if not(pd.isnull(item)) == True]
f_bmi = [f_weight[i]/(f_size[i]/100)**2 for i in range(len(f_size))]
f_mean_bmi = str(round(statistics.mean(f_bmi), 1))
f_stdev_bmi = str(round(statistics.stdev(f_bmi), 1))
# insert bmi into demographics_df
new_row = {'Parameter':[parameter], heading_male: [m_mean_bmi+' ± '+m_stdev_bmi], heading_female: [f_mean_bmi+' ± '+f_stdev_bmi]}
demographics_df = demographics_df.append(pd.DataFrame(new_row), ignore_index=True)

# median_cap
parameter = 'CAP (dB/m)'
sub_variable = 'median_cap'
demographics_df = get_parameters(parameter, sub_variable, heading_male, heading_female, demographics_df)

# cap/iqr
try:
    parameter = 'Median/IQR CAP'
    # male 
    m_median_cap = male_patients['median_cap'].tolist()
    m_iqr = male_patients['iqr_cap'].tolist()
    m_median_cap = [item for item in m_median_cap if not(np.isnan(item)) == True]
    m_iqr = [item for item in m_iqr if not(np.isnan(item)) == True]
    # remove indices for m_median_cap and m_iqr where iqr is 0
    # len_m_median_cap = len(m_median_cap)
    # y = 0
    # for i in range(len(m_median_cap)-y):
    #     if  m_iqr == 0:
    #         m_median_cap.remove(m_median_cap[i])
    #         y +=1 
    m_median_cap = [m_median_cap[i] for i in range(len(m_median_cap)) if m_iqr[i] != 0]
    m_iqr = [m_iqr[i] for i in range(len(m_iqr)) if m_iqr[i] != 0]
    m_median_iqr = [m_median_cap[i]/m_iqr[i] for i in range(len(m_median_cap))]
    m_mean = str(round(statistics.mean(m_median_iqr), 1))
    m_stdev = str(round(statistics.stdev(m_median_iqr), 1))
    #female
    f_median_cap = female_patients['median_cap'].tolist()
    f_median_cap = [item for item in f_median_cap if not(pd.isnull(item)) == True]
    f_iqr = female_patients['iqr_cap'].tolist()
    f_iqr = [item for item in f_iqr if not(pd.isnull(item)) == True]
    # remove indices for f_median and f_iqr where iqr is 0
    f_median_cap = [f_median_cap[i] for i in range(len(f_median_cap)) if f_iqr[i] != 0]
    f_iqr = [f_iqr[i] for i in range(len(f_iqr)) if f_iqr[i] != 0]
    f_median_iqr = [f_median_cap[i]/f_iqr[i] for i in range(len(f_median_cap))]
    f_mean = str(round(statistics.mean(f_median_iqr), 1))
    f_stdev = str(round(statistics.stdev(f_median_iqr), 1))
    # insert median_iqr into demographics_df
    new_row = {'Parameter':[parameter], heading_male: [m_mean+' ± '+m_stdev], heading_female: [f_mean+' ± '+f_stdev]}
    demographics_df = demographics_df.append(pd.DataFrame(new_row), ignore_index=True)
except:
    print('Error in calculating median/iqr CAP (Probably no iqr available for some median_cap)')



print(demographics_df)
# %%
# upload to notion
notion_page_url = 'https://www.notion.so/Demographics-2dff6baa661c495fa2c2cc9093c137e3?pvs=4'
notion_token_230808 ='secret_gkAXWxlfrzEhCvOVRoBOk0BpVP4nKhfp4nb1y4cW13y'

page_title = str('Version: '+version)

notion_df.upload(demographics_df, notion_page_url, title=page_title, api_key=notion_token_230808)

# %%

plt.figure(figsize=(3,5))
plt.boxplot(redcap_survey['median_cap'].tolist())
plt.ylabel('CAP (dB/m)')
plt.show()

# %%
