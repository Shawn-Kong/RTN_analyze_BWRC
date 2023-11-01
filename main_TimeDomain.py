import methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# python 3.8.10

if __name__ == "__main__":
    # print("sns version", sns.__version__)
    # Define the parameters
    directory = r"C:\wgfmu_example\cena\nthin56_Vg(0.55,0.6,0.65,0.7,0.8,0.9)_Vd0.2_10kHz_26K_3900000_1MA.csv"
    sampling_frequency = 10000
    names = [0.32,0.42,0.52,0.57,0.62,0.72] # Name your 6 plots
    data_to_analyze = 5
    cutoff = 0.006

    # 1. Visualize the RTN data in time domain.
    df= pd.read_csv(directory)
    df.columns = ["Time", "Id"]
    df_denoise = methods.notch_filter(df, sampling_frequency)
    
    for i in range(6):
        fig = methods.plot_raw(df_denoise.iloc[650000*i+2000:650000*(i+1) - 2000], name = str(names[i]))
        fig.show()

    # 2. Extract the average current and magnitude of RTN
    data2 = df.iloc[650000*data_to_analyze+2000:650000*(data_to_analyze+1)-2000]
    print("mean is:", np.mean(data2["Id"]))
    peak_values, peak_values_rank_by_density, min_values, min_values_rank_by_density = methods.Perform_KDE(data2, filename="raw_KDE")
    methods.find_magnitude(data2, peak_values, peak_values_rank_by_density)

    # 3. Use Wavelet transform to denoise thermal noise(optional)
    # need to fine-tune the parameters if want to us wavelet transform. 
    # wavelet_denoise = methods.wavelet_transform(data2)
    # print("after wavelet_transform")
    # peak_values, peak_values_rank_by_density, min_values, min_values_rank_by_density = methods.Perform_KDE(wavelet_denoise,column="Id_denoised", filename="denoised_KDE")

    # 4. Find the threshold for Tau extraction
    values_in_range = min_values[(min_values > peak_values_rank_by_density[-1]) & (min_values < peak_values_rank_by_density[-2])]

    if (len(values_in_range)==0):
        values_in_range = min_values[(min_values > peak_values_rank_by_density[-2]) & (min_values < peak_values_rank_by_density[-1])]

    min_index = max([np.where(min_values_rank_by_density == value) for value in values_in_range])
    threshold = min_values_rank_by_density[min_index][0]
    print("threshold is: ", threshold)
    
    # 5. Extract the capture and emission time constant
    emission, capture = methods.record_time(data2, cutoff, threshold, "Id")
    methods.plot_tau_hist(capture, emission)
