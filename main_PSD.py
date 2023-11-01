import pandas as pd
import matplotlib.pyplot as plt
import methods
from scipy import signal
import numpy as np
from tqdm import tqdm 

def plt_multiple_psd(dfs, name, sampling_frequencies, gm=[], title='Power spectrum using Welch\'s method, Vov=0', max_freq=1000, harmonic_tolerance=2):

    plt.figure(figsize=(4,4))
    frequency_list = []
    psd_list = []
    
    for idx, df in enumerate(dfs):
        column_data = df['Id']
        # print(len(sampling_frequencies))
        fs = sampling_frequencies[idx]
        frequencies, psd = signal.welch(column_data, fs=fs, scaling='density', nperseg=fs)  # nperseg=fs/2
        # print("length of psd:", len(psd))
        # print("frequencies:", frequencies[0:10])

        psd = psd / gm[idx]**2
        
        # Limit frequencies and psd to max_freq
        max_freq_idx = np.searchsorted(frequencies, max_freq)
        frequencies = frequencies[:max_freq_idx]
        psd = psd[:max_freq_idx]
        # print("length of psd2:", len(psd))
        # print("length of frequencies:", frequencies[-10:-1])

        
        frequency_list.append(frequencies)
        psd_list.append(psd)

        # Create a mask for unwanted frequencies
        exclude_mask = np.any([
            np.abs(frequencies - 60 * harmonic) <= harmonic_tolerance
            for harmonic in range(1, max_freq // 60 + 1)
        ], axis=0)

        # Replace unwanted psd values with neighboring ones
        for index in np.where(exclude_mask)[0]:
            psd[index] = psd[index-1] if index > 0 else psd[index+1]
        
        
        # plt.loglog(frequencies, psd, label=f'DataFrame {idx+1}')
        plt.loglog(frequencies, psd, label=name[idx])

        psd_masked = psd.copy()
        psd_masked[exclude_mask] = 0
        # print("length of psd_masked:", len(psd_masked))
        
        integral = np.trapz(psd_masked[1:max_freq_idx], frequencies[1:max_freq_idx])
        integral = "{:.4e}".format(integral)
        print("the power of", " ", name[idx], "is", integral)

    
    # 1/f line # Normalize 1/f line to make it visible on the plot with actual psd data
    f_line = np.linspace(frequencies[1], max_freq, 100)  # Avoiding division by zero
    psd_line = 1 / f_line
    
    
    psd_line = psd_line * np.mean(psd) / np.mean(psd_line)
    
    plt.loglog(f_line, psd_line, linestyle='--', label=r'$\frac{1}{f}$', color='red')

    # Get the power of the signal

    # plt.figure(figsize=(12,6))
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Svg (V^2/Hz)')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.savefig(title+".png")

    return frequency_list, psd_list

def notch_filter_small(df, samp_freq):
    notch_freq = [60 * i for i in range(1, int(samp_freq / 120))]
    y_notched = df.copy()
    y_notched = y_notched['Id']

    for filter_f in tqdm(notch_freq):
        if filter_f > 1e4:
            quality_factor = filter_f / 3
        elif filter_f > 1e3:
            quality_factor = filter_f / 2
        else:
            quality_factor = filter_f
        b_notch, a_notch = signal.iirnotch(filter_f, quality_factor, samp_freq)
        freq, h = signal.freqz(b_notch, a_notch, fs = samp_freq)

        y_notched = signal.filtfilt(b_notch, a_notch, y_notched)
    
    df_denoised = pd.DataFrame()
    df_denoised['Time'] = df['Time']
    df_denoised['Id_original'] = df['Id']
    df_denoised['Id'] = y_notched

    return df_denoised

if __name__ == "__main__":
    df2K= pd.read_csv(r"C:\wgfmu_example\cena\nthin56_Vg(0.4, 0.5,0.6,0.65,0.7,0.8)_Vd0.05_50kHz_2K_3900000_1MA.csv")
    df2K.columns = ["Time", "Id"]
    # Remember to change the sampling frequency! 
    df_denoise2K = notch_filter_small(df2K, 50000)
    names = [0.4, 0.5,0.6,0.65,0.7,0.8]
    df_2K_list = []
    for i in range(6):
        df_2K_list.append(df_denoise2K.iloc[650000*i+10000:650000*(i+1) - 10000])

    gm_zero_to_100_1 = [0.000179, 0.000425, 0.000453, 0.000456, 0.000375]
    sampling_frequencies_zero_to_100 = [50000, 50000, 50000, 50000, 50000]
    name = ['0Vov', '0.1Vov', '0.15Vov', '0.2Vov', '0.3Vov']

    frequency_list, psd_list = plt_multiple_psd(dfs = df_2K_list[1:], name=name, sampling_frequencies=sampling_frequencies_zero_to_100, gm=gm_zero_to_100_1, title='Temp=2K, Vov=[0, 0.1, 0.15, 0.2, 0.3]', max_freq=1000, harmonic_tolerance=2)
