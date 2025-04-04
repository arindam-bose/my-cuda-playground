import numpy as np
import matplotlib.pyplot as plt

# program1 = 'cufft 1D'
# program2 = 'fftw 1D'
# fft_ticks = [64, 128, 256, 512, 1024, 2048, 4096, 8092, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456]
# elapsed_time_1 = [0.069632, 0.071068, 0.069575, 0.069935, 0.071563, 0.066409, 0.065961, 0.070663, 0.082314, 0.079993, 0.088264, 0.085698, 0.094617, 0.091715, 0.092579, 0.105324, 0.118083, 0.149415, 0.180646, 0.288893, 0.478088, 1.051997, 3.852042]
# elapsed_time_2 = [0.000823, 0.000668, 0.000632, 0.000676, 0.000619, 0.000982, 0.000944, 0.001649, 0.002656, 0.005053, 0.007769, 0.014545, 0.026510, 0.051689, 0.093379, 0.190877, 0.359093, 0.797793, 1.441364, 3.412655, 8.320708, 17.294937, 46.470760]

# program1 = 'cufft 3D'
# program2 = 'fftw 3D'
# fft_ticks = ['8x8x8', '16x16x16', '32x32x32', '64x64x64', '128x128x128', '256x256x256', '512x512x512']
# elapsed_time_1 = [0.253089, 0.089724, 0.090278, 0.094996, 0.124108, 0.273933, 1.347114]
# elapsed_time_2 = [0.000939, 0.000889, 0.003164, 0.016617, 0.279231, 1.922418, 18.610394]

# x_axis = np.arange(len(fft_ticks))
# fig, ax = plt.subplots(1,1)
# ax.plot(x_axis, elapsed_time_1, marker='o', label=program1)
# ax.plot(x_axis, elapsed_time_2, marker='*', label=program2)
# ax.set_title(f'{program1} vs. {program2}')
# ax.set_xticks(x_axis, fft_ticks, rotation=90)
# ax.set_xlabel('FFT length')
# ax.set_ylabel('Elapsed time (s)')
# ax.legend(loc='upper left')
# ax.grid('on')
# plt.tight_layout()
# plt.show()



program1 = 'cufft 4D (4x1D)'
program2 = 'cufft 4D (3D+1D)'
program3 = 'fftw 4D'
fft_ticks = ['8x8x8x8', '16x16x16x16', '32x32x32x32', '64x64x64x64', '64x64x512x128', '128x128x128x128']
elapsed_time_1 = [0.077631, 0.076496, 0.093748, 0.211920, 4.434531, 4.107372]
elapsed_time_2 = [0.092756, 0.086410, 0.106949, 0.202140, 2.862685, 3.263932]
elapsed_time_3 = [0.001250, 0.005084, 0.072448, 1.053681, 29.177355, 64.327660]

x_axis = np.arange(len(fft_ticks))
fig, ax = plt.subplots(1,1)
ax.plot(x_axis, elapsed_time_1, marker='o', label=program1)
ax.plot(x_axis, elapsed_time_2, marker='*', label=program2)
ax.plot(x_axis, elapsed_time_3, marker='x', label=program3)
ax.set_title(f'{program1} vs. {program2} vs. {program3}')
ax.set_xticks(x_axis, fft_ticks, rotation=90)
ax.set_xlabel('FFT length')
ax.set_ylabel('Elapsed time (s)')
ax.legend(loc='upper left')
ax.grid('on')
plt.tight_layout()
plt.show()