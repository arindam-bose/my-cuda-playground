import numpy as np
import matplotlib.pyplot as plt
import click

# Dictionary of pretty names
cmap =  {
    'cufft1d':'cufft 1D',
    'cufft2d':'cufft 2D',
    'cufft3d':'cufft 3D',
    'cufft4d_31':'cufft 4D (3D+1D)',
    'cufft4d_22':'cufft 4D (2D+2D)',
    'cufft4d_1111':'cufft 4D (4x1D)',
    'fftw1d-aarch64':'fftw 1D: aarch64',
    'fftw2d-aarch64':'fftw 2D: aarch64',
    'fftw3d-aarch64':'fftw 3D: aarch64',
    'fftw4d-aarch64':'fftw 4D: aarch64',
    'fftw4d_3d1d-aarch64':'fftw 4D (3D+1D): aarch64',
    'fftw4d_2d2d-aarch64':'fftw 4D (2D+2D): aarch64',
    'fftw1d-x86_64':'fftw 1D: x86_64',
    'fftw2d-x86_64':'fftw 2D: x86_64',
    'fftw3d-x86_64':'fftw 3D: x86_64',
    'fftw4d-x86_64':'fftw 4D: x86_64',
}

@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.argument('prog1', type=click.STRING)
@click.argument('prog2', type=click.STRING, required=False)
@click.argument('prog3', type=click.STRING, required=False)
def cli(prog1, prog2, prog3):
    prog1 = prog1.split('-') if '-' in prog1 else [prog1]
    program1 = prog1[0]
    arch1 = prog1[1] if len(prog1) > 1 else ''
    filename1 = f'build/time_results_{program1}_{arch1}.txt' if arch1 != '' else f'build/time_results_{program1}.txt'
    filename2 = ''
    filename3 = ''

    if prog2 != None:
        prog2 = prog2.split('-') if '-' in prog2 else [prog2]
        program2 = prog2[0]
        arch2 = prog2[1] if len(prog2) > 1 else ''
        filename2 = f'build/time_results_{program2}_{arch2}.txt' if arch2 != '' else f'build/time_results_{program2}.txt'

    if prog3 != None:
        prog3 = prog3.split('-') if '-' in prog3 else [prog3]
        program3 = prog3[0]
        arch3 = prog3[1] if len(prog3) > 1 else ''
        filename3 = f'build/time_results_{program3}_{arch3}.txt' if arch3 != '' else f'build/time_results_{program3}.txt'
    
    fft_ticks1 = []
    fft_ticks2 = []
    fft_ticks3 = []
    elapsed_time_1 = []
    elapsed_time_2 = []
    elapsed_time_3 = []

    # Read from corresponding files
    with open(filename1, 'r', encoding='UTF-8') as file:
        # Discard the first line
        file.readline()
        while line := file.readline():
            tokens = line.rstrip().split()
            fft_ticks1.append(tokens[0])
            elapsed_time_1.append(float(tokens[1]))

    if filename2 != '':
        with open(filename2, 'r', encoding='UTF-8') as file:
            # Discard the first line
            file.readline()
            while line := file.readline():
                tokens = line.rstrip().split()
                fft_ticks2.append(tokens[0])
                elapsed_time_2.append(float(tokens[1]))

    if filename3 != '':
        with open(filename3, 'r', encoding='UTF-8') as file:
            # Discard the first line
            file.readline()
            while line := file.readline():
                tokens = line.rstrip().split()
                fft_ticks3.append(tokens[0])
                elapsed_time_3.append(float(tokens[1]))

    fft_ticks_error_filter = False
    if fft_ticks2 != [] and fft_ticks3 != []:
        if not (fft_ticks1 == fft_ticks2 == fft_ticks3):
            fft_ticks_error_filter = True
    elif fft_ticks2 != []:
        if not (fft_ticks1 == fft_ticks2):
            fft_ticks_error_filter = True

    if fft_ticks_error_filter:
        print('Error: At least one FFT ticks are different')
        print(f'Ticks for program1: {fft_ticks1} | length: {len(fft_ticks1)}')
        print(f'Ticks for program2: {fft_ticks2} | length: {len(fft_ticks2)}')
        print(f'Ticks for program3: {fft_ticks3} | length: {len(fft_ticks3)}')
        exit(1)

    ax_title = f'{program1}'
    cmap_key = f'{program1}-{arch1}' if arch1 != '' else program1
    x_axis = np.arange(len(fft_ticks1))
    fig, ax = plt.subplots(1,1)
    ax.plot(x_axis, elapsed_time_1, marker='o', label=cmap[cmap_key])
    if elapsed_time_2 != []: 
        cmap_key = f'{program2}-{arch2}' if arch2 != '' else program2
        ax.plot(x_axis, elapsed_time_2, marker='*', label=cmap[cmap_key])
        ax_title += f' vs. {program2}'
    if elapsed_time_3 != []:
        cmap_key = f'{program3}-{arch3}' if arch3 != '' else program3
        ax.plot(x_axis, elapsed_time_3, marker='x', label=cmap[cmap_key])
        ax_title += f' vs. {program3}'
    ax.set_title(ax_title)
    ax.set_xticks(x_axis, fft_ticks1, rotation=90)
    ax.set_xlabel('FFT length')
    ax.set_ylabel('Elapsed time (s)')
    ax.legend(loc='upper left')
    ax.grid('on')
    plt.tight_layout()
    plt.savefig(f'docs/img/{ax_title.replace(" vs. ", "_")}.png')
    plt.show()

if __name__ == '__main__':
    cli()