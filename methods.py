# %%
# Importing necessary libraries

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.stats import expon
from scipy.stats import poisson
import pywt



# %%
# This function filters out 60Hz and its harmonics. It will make the time domain waveform much cleaner for 
# visulization. The quality factor is 30, so it will filter out a lot of high frequency signals. 
def notch_filter(df, samp_freq):
    notch_freq = [60 * i for i in range(1, int(samp_freq / 120))]
    y_notched = df.copy()
    y_notched = y_notched['Id']
    quality_factor = 30
    for filter_f in tqdm(notch_freq):
        b_notch, a_notch = signal.iirnotch(filter_f, quality_factor, samp_freq)
        freq, h = signal.freqz(b_notch, a_notch, fs = samp_freq)

        # apply notch filter to signal
        y_notched = signal.filtfilt(b_notch, a_notch, y_notched)

    plt.subplot(211)
    plt.plot(df['Time'],df['Id'], color = 'r')
    # plot notch-filtered version of signal
    plt.subplot(212)
    plt.plot(df['Time'], y_notched, color = 'r')
    
    df_denoised = pd.DataFrame()
    df_denoised['Time'] = df['Time']
    df_denoised['Id_original'] = df['Id']
    df_denoised['Id'] = y_notched

    return df_denoised


# %% 
# This is also time domain filter for 60Hz and its harmonics, but it has a higher Q. 
# Utilized by PSD analysis. 
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

    plt.subplot(211)
    plt.plot(df['Time'],df['Id'], color = 'r')
    # plot notch-filtered version of signal
    plt.subplot(212)
    plt.plot(df['Time'], y_notched, color = 'r')
    
    df_denoised = pd.DataFrame()
    df_denoised['Time'] = df['Time']
    df_denoised['Id_original'] = df['Id']
    df_denoised['Id'] = y_notched

    return df_denoised

# %%
def plot_raw(df, y_column="Id", name = "plot", color="blue", save_to_file=True):
    # Create a line plot trace
    trace = go.Scatter(x=df['Time'], y=df[y_column], 
                       mode='lines', 
                       name=name,
                       line=dict(color=color))

    # Create a layout
    layout = go.Layout(title=name, 
                       xaxis=dict(title='Time'), 
                       yaxis=dict(title='Current'))

    # Create a Figure and add the trace
    fig = go.Figure(data=[trace], layout=layout)

    return fig

# %%
def plt_psd(df, fs=50000, max_freq=1000, harmonic_tolerance=2):
    
    column_data = df['Id']
    frequencies, psd = signal.welch(column_data, fs=fs, scaling='density', nperseg=fs/2)
    
    # Limit frequencies and psd to max_freq
    max_freq_idx = np.searchsorted(frequencies, max_freq)
    frequencies = frequencies[:max_freq_idx]
    psd = psd[:max_freq_idx]
    
    # Create a mask for unwanted frequencies
    exclude_mask = np.any([
        np.abs(frequencies - 60 * harmonic) <= harmonic_tolerance
        for harmonic in range(1, max_freq // 60 + 1)
    ], axis=0)
    
    # Replace unwanted psd values with neighboring ones
    for idx in np.where(exclude_mask)[0]:
        # If it's not the first element, take the previous value, otherwise take the next one
        psd[idx] = psd[idx-1] if idx > 0 else psd[idx+1]

    # Plotting
    plt.figure(figsize=(12,6))
    plt.loglog(frequencies, psd, label='Window: default')
    plt.title('Power spectrum using Welch\'s method')
    plt.xlabel('Frequency')
    plt.ylabel('PSD (A^2/Hz)')
    plt.grid(which='both', axis='both')
    plt.show()
    
    return frequencies, psd


# %%
def plt_multiple_psd(dfs, name, sampling_frequencies, gm=[], title='Power spectrum using Welch\'s method, Vov=0', max_freq=1000, harmonic_tolerance=2):

    plt.figure(figsize=(4,4))
    frequency_list = []
    psd_list = []
    
    for idx, df in enumerate(dfs):
        column_data = df['Id']
        fs = sampling_frequencies[idx]
        frequencies, psd = signal.welch(column_data, fs=fs, scaling='density', nperseg=fs/2)

        psd = psd / gm[idx]**2
        
        # Limit frequencies and psd to max_freq
        max_freq_idx = np.searchsorted(frequencies, max_freq)
        frequencies = frequencies[:max_freq_idx]
        psd = psd[:max_freq_idx]

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
    
    # 1/f line: 
    f_line = np.linspace(frequencies[1], max_freq, 100)  # Avoiding division by zero
    psd_line = 1 / f_line
    
    # Normalize 1/f line to make it visible on the plot with actual psd data
    # This can be adjusted depending on your actual psd values
    psd_line = psd_line * np.mean(psd) / np.mean(psd_line)
    
    plt.loglog(f_line, psd_line, linestyle='--', label=r'$\frac{1}{f}$', color='red')
    
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Svg (V^2/Hz)')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()

    return frequency_list, psd_list

# %%
def plt_psd_1overf(df, fs=10000, max_freq=1000, harmonic_tolerance=2):
    # fs=10000
    column_data = df['Id']
    frequencies, psd = signal.welch(column_data, fs=fs, scaling='density', nperseg=fs/2)
    
    # Limit frequencies and psd to max_freq
    max_freq_idx = np.searchsorted(frequencies, max_freq)
    frequencies = frequencies[:max_freq_idx]
    psd = psd[:max_freq_idx]
    
    # Create a mask for unwanted frequencies
    exclude_mask = np.any([
        np.abs(frequencies - 60 * harmonic) <= harmonic_tolerance
        for harmonic in range(1, max_freq // 60 + 1)
    ], axis=0)
    
    # Replace unwanted psd values with neighboring ones
    for idx in np.where(exclude_mask)[0]:
        psd[idx] = psd[idx-1] if idx > 0 else psd[idx+1]

    # 1/f line: 
    f_line = np.linspace(frequencies[1], max_freq, 100)  # Avoiding division by zero
    psd_line = 1 / f_line
    
    # Normalize 1/f line to make it visible on the plot with actual psd data
    # This can be adjusted depending on your actual psd values
    psd_line = psd_line * np.mean(psd) / np.mean(psd_line)
    
    # Plotting
    plt.figure(figsize=(12,6))
    plt.loglog(frequencies, psd, label='PSD')
    plt.loglog(f_line, psd_line, linestyle='--', label=r'$\frac{1}{f}$', color='red')
    plt.title('Power spectrum using Welch\'s method')
    plt.xlabel('Frequency')
    plt.ylabel('PSD (A^2/Hz)')
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.show()
    
    return frequencies, psd

# %%
def integrate_psd(frequencies, psd, gm , fs=10000, harmonic_tolerance=2, max_freq=1000):

    # Find the indices where frequency is within harmonic_tolerance of 60 Hz multiples
    exclude_mask = np.any([
        np.abs(frequencies - 60 * harmonic) < harmonic_tolerance
        for harmonic in range(1, fs // 120 + 1)
    ], axis=0)

    # Set the psd values at these frequencies to zero
    psd_masked = psd.copy()
    psd_masked[exclude_mask] = 0
    
    # Find the index of the maximum frequency to consider in the integration
    max_freq_idx = np.searchsorted(frequencies, max_freq)

    # Calculate the integral of the modified PSD using numpy's trapezoidal rule
    integral = np.trapz(psd_masked[1:max_freq_idx], frequencies[1:max_freq_idx])
    integral = "{:.4e}".format(integral)
    return integral

def Perform_KDE(df, column="Id", n_peaks=4, type="normal", save_to_fig=True,  filename="KDE_output.png"):
    data = df[column]
    print(data.head())

    # Calculate KDE
    KDE_model = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)  # You can adjust the number of points as needed
    y = KDE_model(x)

    # Debug prints
    print("len(y)=", len(y))
    print("len(x)=", len(x))

    # Find peaks in the KDE distribution
    peaks, _ = find_peaks(y)

    # Find local minima by inverting the KDE y-data and finding peaks
    inverted_y = -y
    local_minima, _ = find_peaks(inverted_y)
    local_minima = np.array([int(x) for x in local_minima])
    minima = local_minima[np.argsort(inverted_y[local_minima])[-len(local_minima):]]

    # Get the top peaks based on height
    top_peaks = peaks[np.argsort(y[peaks])[-n_peaks:]]

    print("minima: ", minima)
    print("peaks: ", top_peaks)

    # Create figure and axis
    plt.figure(figsize=(10, 5))  # You can adjust the figure size as needed
    plt.fill_between(x, y, alpha=0.5)
    plt.scatter(x[top_peaks], y[top_peaks], color='red', s=30, marker='o')
    plt.scatter(x[local_minima], y[local_minima], color='green', s=30, marker='o')
    
    if type == "log":
        plt.gca().set_yscale("log")
    # plt.show()
    if (save_to_fig):
        plt.savefig(filename, dpi=300)
    plt.close()

    # Retrieve the x-values corresponding to these peaks
    peak_values = x[top_peaks]
    min_values_rank_by_density = x[minima]

    # Safeguard against potential IndexError
    valid_minima = local_minima[local_minima < len(x)]
    min_values = x[valid_minima]

    peak_values_rank_by_density = peak_values.copy()
    peak_values = np.sort(peak_values)
    print(peak_values)
    print(peak_values_rank_by_density)
    print(min_values)
    print(min_values_rank_by_density)

    print("finish plotting!")
    return peak_values, peak_values_rank_by_density, min_values, min_values_rank_by_density

# %%
def record_time(df, cutoff, threshold, column):
    emission_time = []
    capture_time = []
    start_time = None
    end_time = None
    # iterate over dataframe rows
    for idx, row in df.iterrows():
        if row[column] < threshold and start_time is None:
            start_time = row['Time']
    
        # check if rolling average rises above 521, and an ongoing period is being tracked
        elif row[column] > threshold and start_time is not None:
            end_time = row['Time']
            duration = end_time - start_time
            if (duration >= cutoff):
                emission_time.append((start_time, end_time, duration))
                start_time = None 
                
    if start_time is not None:
        end_time = df.loc[idx, 'Time']
        duration = end_time - start_time
        if (duration >= cutoff):
                emission_time.append((start_time, end_time, duration))
                start_time = None

    for idx, row in df.iterrows():
        if row[column] > threshold and start_time is None:
            start_time = row['Time']
    
        # check if rolling average rises above 521, and an ongoing period is being tracked
        elif row[column] < threshold and start_time is not None:
            end_time = row['Time']
            duration = end_time - start_time
            if (duration >= cutoff):
                capture_time.append((start_time, end_time, duration))
                start_time = None 

    if start_time is not None:
        end_time = df.loc[idx, 'Time']
        duration = end_time - start_time
        if (duration >= cutoff):
                capture_time.append((start_time, end_time, duration))
                start_time = None
        
    return (emission_time, capture_time)

def plot_tau_hist(capture, emission):
    capture1 = [x[2] for x in capture]
    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    plt.hist(capture1, bins=10, edgecolor='black')  # 10 bins, adjust as necessary
    plt.xlabel('Capture_times')
    plt.ylabel('# of Occurance')
    plt.title('Tau_capture')
    plt.savefig("capture_time_dist.png")

    emission1 = [x[2] for x in emission]
    # Plotting the histogram
    plt.figure(figsize=(10, 5))
    plt.hist(emission1, bins=10, edgecolor='black')  # 10 bins, adjust as necessary
    plt.xlabel('Emission')
    plt.ylabel('# of Occurance')
    plt.title('Tau_emission')
    plt.savefig("emission_time_dist.png")

    print("tau_capture = ", np.mean(capture1))
    print("tau_emission = ", np.mean(emission1))
    print("# of tau_capture", len(capture1))
    print("# of tau_emission = ", len(emission1))

    return 


# %%
def plot_fft(df, fs):
    # Get the time and current data from the dataframe
    t = df['Time']
    current = df['Id']

    # Apply the FFT
    fft_result = np.fft.fft(current)

    # Compute the normalized magnitude (absolute value) in dB
    magnitude = 20 * np.log10(np.abs(fft_result) / len(t))

    # Generate the frequency axis
    freq = np.fft.fftfreq(len(t), 1/fs)

    # Take only the positive half of the spectrum
    mask = freq >= 0
    freq = freq[mask]
    magnitude = magnitude[mask]

    # Create a line plot trace
    trace = go.Scatter(x=freq, y=magnitude, 
                       mode='lines', 
                       name='FFT',
                       line=dict(color='blue'))

    # Create a layout
    layout = go.Layout(title='FFT of Drain Current', 
                       xaxis=dict(title='Frequency'), 
                       yaxis=dict(title='Magnitude'))

    # Create a Figure and add the trace
    fig = go.Figure(data=[trace], layout=layout)

    return fig

# %%
def wavelet_transform(df, threshold=1e-5, level_val=7, save_to_fig=True, filename="wavelet_transform.png"):
    df_denoise1 = df.copy()
    noisy_signal = df_denoise1["Id"]
    t = df_denoise1["Time"]
    
    coeffs = pywt.wavedec(noisy_signal, 'haar', level=7)
    
    
    # Set a threshold value and apply thresholding to the detail coefficients
    
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]
    
    # Reconstruct the signal from the thresholded coefficients
    denoised_signal = pywt.waverec(coeffs_thresholded, 'haar')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 2)
    plt.plot(t, noisy_signal, label='Noisy Signal')
    plt.title("Signal with Noise")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    if len(t) != len(denoised_signal):
        # plt.plot(t, denoised_signal[:-1], label='Denoised Signal')
        denoised_signal = denoised_signal[:-1]
    plt.plot(t, denoised_signal, label='Denoised Signal')
    plt.title("Denoised Signal")
    plt.legend()
    
    plt.tight_layout()
    
    if (save_to_fig):
        plt.savefig(filename, dpi=300)
    plt.close()

    trace = go.Scatter(x=t, y=denoised_signal, 
                       mode='lines', 
                       name="Wavelet Denoised Data")

    # Create a layout
    layout = go.Layout(title="Wavelet Denoised Data", 
                       xaxis=dict(title='Time'), 
                       yaxis=dict(title='Current'))

    # Create a Figure and add the trace
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
    
    df_denoise1["Id_denoised"] = denoised_signal
    return df_denoise1

# %%
def record_time(df, cutoff, threshold, column):
    emission_time = []
    capture_time = []
    emission_start_time = None
    emission_end_time = None
    capture_start_time = None
    capture_end_time = None
    # iterate over dataframe rows
    for idx, row in df.iterrows():
        if row[column] < threshold and emission_start_time is None:
            emission_start_time = row['Time']
            
        elif row[column] > threshold and capture_start_time is None:
            capture_start_time = row['Time']
    
        elif row[column] > threshold and emission_start_time is not None:
            emission_end_time = row['Time']
            duration = emission_end_time - emission_start_time
            if (duration >= cutoff):
                emission_time.append((emission_start_time, emission_end_time, duration))
                emission_start_time = None 

        elif row[column] < threshold and capture_start_time is not None:
            capture_end_time = row['Time']
            duration = capture_end_time - capture_start_time
            # print(duration)
            if (duration >= cutoff):
                capture_time.append((capture_start_time, capture_end_time, duration))
                capture_start_time = None 
                
    # When we reach the end of the data, but the capture/emission even has not been changed.             
    if emission_start_time is not None:
        emission_end_time = df.loc[idx, 'Time']
        duration = emission_end_time - emission_start_time
        if (duration >= cutoff):
                emission_time.append((emission_start_time, emission_end_time, duration))
                emission_start_time = None

    elif capture_start_time is not None:
        end_time = df.loc[idx, 'Time']
        duration = capture_end_time - capture_start_time
        if (duration >= cutoff):
                capture_time.append((capture_start_time, capture_end_time, duration))
                start_time = None
        
    return emission_time, capture_time

# %%
def find_magnitude(df, peak_values, peak_values_density):
    average = df["Id"].mean()
    
    if len(peak_values) <= 3:
        magnitude0 = abs(peak_values_density[-1] - peak_values_density[-2])
        ratio0= magnitude0 / average
        print("peak_values are not enough. Peak values are:", peak_values)
        print("average is: ", average)
        print("magnitude for a single trap is: ", [magnitude0])
        print("ratio for a single trap is: ", [ratio0])
        return peak_values

    else:
        magnitude0 = abs(peak_values_density[-1] - peak_values_density[-2])
        magnitude1 = peak_values[1] - peak_values[0]
        magnitude2 = peak_values[2] - peak_values[1]
        magnitude = [magnitude1, magnitude2]
    
    
        ratio0= magnitude0 / average
        ratio1 = magnitude1 / average
        ratio2 = magnitude2 / average
        ratio = [ratio1, ratio2]
    
        # print("average is: ", float(f"{average:.4e}"))
        # print("magnitude for a single trap is: ", float(f"{magnitude0:.4e}"))
        # print("ratio for a single trap is: ", float(f"{ratio0:.2e}"))
        # print("magnitudes are: ", [float(f"{number:.4e}") for number in magnitude])
        # print("ratios are: ", [float(f"{number:.2e}") for number in ratio])
        # print("peak values are: ", peak_values)

        print("average is: ", average)
        print("magnitude for a single trap is: ", magnitude0)
        print("ratio for a single trap is: ", ratio0)
        print("magnitudes are: ", [number for number in magnitude])
        print("ratios are: ", [number for number in ratio])
        print("peak values are: ", peak_values)
    return 
