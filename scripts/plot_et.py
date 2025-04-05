import numpy as np
import matplotlib.pyplot as plt

program1 = 'cufft2d'
program2 = 'fftw2d'

cmap =  {
    'cufft1d':'cufft 1D',
    'cufft2d':'cufft 2D',
    'cufft3d':'cufft 3D',
    'cufft4d_3d1d':'cufft 4D (3D+1D)',
    'cufft4d_2d2d':'cufft 4D (2D+2D)',
    'cufft4d_4x1d':'cufft 4D (4x1D)',
    'fftw1d':'fftw 1D',
    'fftw2d':'fftw 2D',
    'fftw3d':'fftw 3D',
    'fftw4d':'fftw 4D',
}

filename1 = f'build/time_results_{program1}.txt'
filename2 = f'build/time_results_{program2}.txt'
fft_ticks1 = []
fft_ticks2 = []
elapsed_time_1 = []
elapsed_time_2 = []

with open(filename1, 'r', encoding='UTF-8') as file:
    # Discard the first line
    file.readline()
    while line := file.readline():
        data = line.rstrip().split()
        fft_ticks1.append(data[0])
        elapsed_time_1.append(float(data[1]))

with open(filename2, 'r', encoding='UTF-8') as file:
    # Discard the first line
    file.readline()
    while line := file.readline():
        data = line.rstrip().split()
        fft_ticks2.append(data[0])
        elapsed_time_2.append(float(data[1]))

if (fft_ticks1 != fft_ticks2):
    print('Error: FFT ticks are different')
    print(fft_ticks1)
    print(fft_ticks2)

x_axis = np.arange(len(fft_ticks1))
fig, ax = plt.subplots(1,1)
ax.plot(x_axis, elapsed_time_1, marker='o', label=cmap[program1])
ax.plot(x_axis, elapsed_time_2, marker='*', label=cmap[program2])
ax.set_title(f'{cmap[program1]} vs. {cmap[program2]}')
ax.set_xticks(x_axis, fft_ticks1, rotation=90)
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