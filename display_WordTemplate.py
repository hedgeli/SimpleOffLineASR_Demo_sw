import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import matplotlib
# from PyQt5.Qt import PYQT_VERSION_STR
# import spectrogram_sw as spec

np.seterr(divide='ignore', invalid='ignore')
# 预设字体格式，并传给rc方法
font = {'family': 'SimHei', "size": 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.unicode_minus'] = False

np.set_printoptions(precision=2)

# print("PyQt5 Version is: {}".format(PYQT_VERSION_STR))


def disp_WordTemplate(file):
    pars_pc = np.load(file)
    for i in range(3):
        for j in range(4):
            plt.subplot(3,4,i*4 + j+1)
            plt.imshow(pars_pc['arr_0'][i*4 + j]['specgram'],
                      origin='lower')
            plt.title(pars_pc['arr_0'][i*4 + j]['word'])
    plt.show()


if __name__ == '__main__':
    WrodTmpt_file = "./ResFiles/Wav/Word_voice_tmptWordPara.npz"
    # disp_WordTemplate(WrodTmpt_file)




