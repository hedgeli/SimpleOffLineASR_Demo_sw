import numpy as np
import wave
import scipy.signal

def load_wav(file=" "):
    if file == " ":
        print("need the .wav file to load")
        return []
    else:
        f = wave.open(file, 'rb')
        paras = f.getparams()
        nchs, sampdepth, Fs, n_points = paras[:4]
        data_buff = f.readframes(n_points)
        f.close()
        wav_data = np.frombuffer(data_buff, dtype=np.short)
        return nchs, sampdepth, Fs, n_points, wav_data


def normalization_pn1(data):  # 归一化 范围正负1
    if len(data.shape) == 1:
        _range = np.max(abs(data))
        return data / _range
    else:
        print("normalization_pn1(data) input data error")
        return []


def normalization_array_pn1(data):
    if len(data.shape) == 1:
        _out = normalization_pn1(data)
    elif len(data.shape) == 2:
        _row = len(data)
        _len = len(data[0])
        _out = np.zeros((_row, _len))
        for i in range(_row):
            _out[i] = normalization_pn1(data[i])
    else:
        print("normalization_array_pn1(data) input data error")
        _out = []
    return _out



def normalization_01(data):  # 归一化 范围0到1
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_frame(datain, frame_center, lengh):
    start_ = int(frame_center - lengh / 2)
    if start_ < 0:
        start_ = 0
    stop_ = start_ + lengh
    frame = datain[start_:stop_]
    return frame




def calc_fft(datain, Fs):
    len_N = len(datain)
    ft = np.fft.fft(datain)
    nfft = np.array(abs(ft[range(int(len_N / 2))] / len_N))
    freq = Fs / 2 / len_N
    freq_y = np.rint(np.array(list(range(0, int(len_N / 2)))) * freq).astype(np.int)
    return nfft, freq_y


def calc_energy(datain):
    return np.sum(datain * datain)


# 数据取对数
def calc_log(data, snr=46, logmod="log10", dtype="float"):
    minval = 1
    # if dtype == "float":
    #     if np.max(data) < 1.0:          # NOT <= 1.0 !!!!
    #         data = data * 32767         # 把-1到1的浮点数转换为16位整型数
    #         minval = int(32767 / (10 ** (snr / 20)))

    # if minval < 1:
    #     minval = 1
    # data = np.where(abs(data) < minval, np.sign(data) * minval, data)
    data = np.where(abs(data) < minval, np.sign(data), data)

    if logmod == "log2":
        _data = np.sign(data) * np.log2(abs(data))
    elif logmod == "log10":
        _data = 20 * np.sign(data) * np.log10(abs(data))        # dB=20*log10(amp)
    else:
        _data = np.sign(data) * np.log(abs(data))
    return _data


def calc_delta_n(datain, n=1):
    _len = len(datain)
    _out = np.zeros(_len)
    _out[0:_len-n] = datain[n:_len]-datain[0:_len-n]
    return _out

def move_avg(datain, n=5, mode="same"):
    return (np.convolve(datain, np.ones((n,)) / n, mode=mode))


# def cal_frames_fft_log(data, Fs, t_start=0,
# t_window=0.032, frame_n=16, move_step=0.016):
def cal_frames_fft_log(data, Fs, t_start=0,
                       t_window=0.016, frame_n=10, move_step=0.008):
    idx_start = int(Fs * t_start)
    n_width = int(Fs * t_window)
    n_step = int(Fs * move_step)
    # 加窗函数
    mul_window = np.hamming(n_width)
    fft_out = np.zeros((frame_n, int((n_width + 1) / 2)))
    freq_out = np.zeros((frame_n, int((n_width + 1) / 2)))
    for i_frame in range(frame_n):
        frm_start = idx_start + i_frame * n_step
        if (frm_start + n_width) > len(data):
            break
        frame_data = data[frm_start:frm_start + n_width]
        # if len(frame_data) != len(mul_window):
        #     print("err: ",  len(frame_data), "----", len(mul_window))
        frame_data = frame_data * mul_window
        nfft, freq_y = calc_fft(frame_data, Fs)
        nfft_log = calc_log(nfft, 1, "log10")
        nfft_avg = move_avg(nfft_log, 5)
        len_fft = len(nfft_avg)
        fft_out[i_frame:, 0:len_fft] = nfft_avg
        freq_out[i_frame:, 0:len_fft] = freq_y
    return fft_out, freq_out


def conv_avg9_2d(dat2d,):
    _datout = np.copy(dat2d)
    line, row = dat2d.shape
    for i in range(2,line-1):
        for j in range(2,row-1):
            _datout[i,j] = (dat2d[i-1,j-1] + dat2d[i-1,j]
                            +dat2d[i-1,j+1] + dat2d[i,j+1]
                            +dat2d[i,j-1] + dat2d[i+1,j-1]
                            +dat2d[i+1,j] + dat2d[i+1,j+1]
                           + dat2d[i,j])/9
    return _datout


