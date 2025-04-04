import numpy as np
import matplotlib.pyplot as plt



program1 = 'cufft 1D'
program2 = 'fftw 1D'
fft_ticks = [64, 128, 256, 512, 1024, 2048, 4096, 8092, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456]
elapsed_time_1 = [0.073074, 0.070511, 0.073638, 0.065568, 0.069474, 0.065691, 0.071692, 0.067107, 0.086724, 0.085446, 0.087722, 0.086332, 0.087629, 0.094722, 0.103402, 0.103443, 0.115831, 0.117995, 0.283215, 0.477620, 0.412856, 1.043853, 5.961780]
elapsed_time_2 = [0.000801, 0.000516, 0.000535, 0.000561, 0.000608, 0.000649, 0.000857, 0.001928, 0.002109, 0.003686, 0.007009, 0.013366, 0.023912, 0.045863, 0.078879, 0.152276, 0.308957, 0.683762, 1.323854, 2.980493, 7.221835, 15.253605, 35.845345]

# program1 = 'cufft 2D'
# program2 = 'fftw 2D'
# fft_ticks = ['8x8', '16x16', '32x32', '64x64', '128x128', '256x256', '512x512', '1024x1024', '2048x2048', '4096x4096', '8192x8192', '16384x16384']
# elapsed_time_1 = [0.090490, 0.086141, 0.088205, 0.092628, 0.088218, 0.091242, 0.099452, 0.109544, 0.116009, 0.154541, 0.459426, 2.387575]
# elapsed_time_2 = [0.000807, 0.000578, 0.000573, 0.000738, 0.003663, 0.007290, 0.026743, 0.082993, 0.328167, 1.436665, 7.117310, 30.128214]

# program1 = 'cufft 3D'
# program2 = 'fftw 3D'
# fft_ticks = ['8x8x8', '16x16x16', '32x32x32', '64x64x64', '128x128x128', '256x256x256', '512x512x512']
# elapsed_time_1 = [0.084987, 0.086152, 0.086388, 0.094165, 0.111248, 0.217069, 1.065681]
# elapsed_time_2 = [0.000923, 0.000798, 0.002746, 0.016117, 0.253879, 1.962103, 16.849342]

x_axis = np.arange(len(fft_ticks))
fig, ax = plt.subplots(1,1)
ax.plot(x_axis, elapsed_time_1, marker='o', label=program1)
ax.plot(x_axis, elapsed_time_2, marker='*', label=program2)
ax.set_title(f'{program1} vs. {program2}')
ax.set_xticks(x_axis, fft_ticks, rotation=90)
ax.set_xlabel('FFT length')
ax.set_ylabel('Elapsed time (s)')
ax.legend(loc='upper left')
ax.grid('on')
plt.tight_layout()
plt.show()



# program1 = 'cufft 4D (4x1D)'
# program2 = 'cufft 4D (3D+1D)'
# program3 = 'fftw 4D'
# fft_ticks = ['8x8x8x8', '16x16x16x16', '32x32x32x32', '64x64x64x64', '64x64x512x128', '128x128x128x128']
# elapsed_time_1 = [0.077901, 0.076350, 0.091987, 0.159601, 3.864928, 3.909272]
# elapsed_time_2 = [0.109665, 0.094532, 0.109635, 0.236655, 2.866760, 2.838921]
# elapsed_time_3 = [0.001167, 0.004935, 0.067764, 1.011662, 26.533165, 61.367886]

# x_axis = np.arange(len(fft_ticks))
# fig, ax = plt.subplots(1,1)
# ax.plot(x_axis, elapsed_time_1, marker='o', label=program1)
# ax.plot(x_axis, elapsed_time_2, marker='*', label=program2)
# ax.plot(x_axis, elapsed_time_3, marker='x', label=program3)
# ax.set_title(f'{program1} vs. {program2} vs. {program3}')
# # ax.set_title(f'{program1} vs. {program2}')
# ax.set_xticks(x_axis, fft_ticks, rotation=90)
# ax.set_xlabel('FFT length')
# ax.set_ylabel('Elapsed time (s)')
# ax.legend(loc='upper left')
# ax.grid('on')
# plt.tight_layout()
# plt.show()