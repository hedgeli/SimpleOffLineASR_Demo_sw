import sys
import numpy as np

g_Word_Matched_Rate = 0.4

WordVoicePara = np.dtype([('word', 'U10'),
                          ('_version', 'int16'),
                          ('create_time', 'f8'),
                          ('spec_dhash', 'f4', (16, 16)),
                          ('spec_zoom_out', 'f4', (16, 16)),
                          ("_zoom_width", 'int16'),
                          ('specgram', 'f4', (80, 80)),
                          ('spec_width', 'int16'),
                          ("wav_file", 'U60'),
                          ("samp_rate", "f4"),
                          ("start_idx", "int64"),
                          ("stop_idx", "int64"),
                          ("time_length", "f4"),
                          ("ext_info", 'U50')])


def save_word_voice_para(para, spec,
                         word='', file='',
                         s_idx=0, e_idx=0,
                         time=0.0,
                         ext_info=""):

    para['word'] = word

    if spec.shape[0] > 80:
        word_fft = spec[0:80, 0:80].T
    else:
        word_fft = spec[:, 0:80].T

    _v, _h = word_fft.shape

    para['specgram'][0:_v, 0:_h] = word_fft[0:_v, 0:_h]
    para['spec_width'] = _v
    para['wav_file'] = file
    para['start_idx'] = s_idx
    para['stop_idx'] = e_idx
    para['ext_info'] = ext_info
    para['create_time'] = time
    _zoom_out = arr_sum_zoom_out(para['specgram'], 4, 4, 16, 16)

    # move_time = 0.01
    # num_frms = int(len(_data) / (self.Fs * move_time))
    # _fft, _freq = sigana.cal_frames_fft_log(_data * 10000, self.Fs, t_start=0,
    #                                         t_window=0.02, frame_n=num_frms,
    #                                         move_step=move_time)
    # _fft = sigana.normalization_array_pn1(_fft)
    #
    # # print("specgram shape: ", _fft.shape)
    # if _fft.shape[0] > 80:
    #     self.word_fft = _fft[0:80, 0:80].T
    # else:
    #     self.word_fft = _fft[:, 0:80].T
    #
    # _zoom_out = voice_para_save.arr_sum_zoom_out(self.word_fft, 4, 4, 16, 16)

    _v, _h = _zoom_out.shape
    if _v > 16:
        _v = 16
    if _h > 16:
        _h = 16
    para['spec_zoom_out'][0:_v, 0:_h] = _zoom_out[0:_v, 0:_h]
    para['_zoom_width'] = _h

    _dhash = cal_dhash(_zoom_out)

    _sign_dhash = std_dhash(_dhash, 0.06)

    _v, _h = _sign_dhash.shape
    if _v > 16:
        _v = 16
    if _h > 16:
        _h = 16
    para['spec_dhash'][0:_v, 0:_h] = _sign_dhash[0:_v, 0:_h]


def arr_similar_rate(arr1, arr2, cut_rate=0.5, simi_rate=0.70):
    _arr1 = arr1 / np.max(abs(arr1))
    _arr2 = arr2 / np.max(abs(arr2))
    _sum = (_arr1 + _arr2) / 2
    _sum = np.where(_sum >= cut_rate, _sum, 1)
    _sub = _arr1 - _arr2
    _diff = abs(_sub / _sum)
    _sim = np.where(abs(_diff) <= (1 - simi_rate), 1.0, 0)
    _out = np.sum(_sim) / (arr1.shape[0] * arr1.shape[1])
    # _out = np.sum(_sim)
    return _out


def sum_sub_abs(arr1, arr2, cut_rate=0.2, avg_rate=0.2):
    arr = abs(arr1 - arr2)
    arr = np.where(arr < cut_rate, 0, arr)
    arr_avg = abs(arr1+arr2)/2
    arr_avg = np.where(arr_avg < avg_rate, avg_rate, arr_avg)
    # _v, _w = arr.shape
    # _sum = np.sum(abs(arr/arr_avg))/(_v*_w)
    _sum = np.sum(abs(arr / arr_avg))/10
    return _sum


def cal_abs_diff(arr1, arr2, cut_rate=0.65, avg_rate=0.5):
    _v1, _w1 = arr1.shape
    _v2, _w2 = arr2.shape
    arr1 = arr1/np.max(abs(arr1))
    arr2 = arr2/np.max(abs(arr2))
    if _v1 != _v2:
        print('vertical of cal_dhash_diff_rate(arr1, arr2) must be same')
        return []
    if _w1 == _w2:
        _out = sum_sub_abs(arr1, arr2, cut_rate, avg_rate)
        return _out
    if _w1 > _w2:
        _d = _w1 - _w2
        t_val = np.zeros(_d+1)
        for k in range(_d+1):
            t_val[k] = sum_sub_abs(arr1[:, k:k+_w2], arr2[:, 0:_w2], cut_rate, avg_rate)
        return np.min(t_val)
    else:
        _d = _w2 - _w1
        t_val = np.zeros(_d+1)
        for k in range(_d+1):
            t_val[k] = sum_sub_abs(arr1[:, 0:_w1], arr2[:, k:k+_w1], cut_rate, avg_rate)
        return np.min(t_val)


def sub_abs_dhash(arr1, arr2):
    arr = arr1 - arr2
    _v, _w = arr.shape
    if _v * _w > 0:
        _sum = np.sum(abs(arr))/(_v*_w)
        return _sum
    else:
        return 1.0


def cal_dhash_diff_rate(arr1, arr2, threshold=0.5, phoneme=2):
    _v1, _w1 = arr1.shape
    _v2, _w2 = arr2.shape
    if _v1 != _v2:
        print('vertical of cal_dhash_diff_rate(arr1, arr2) must be same')
        return []
    if _w1 == _w2:
        _phoneme_width = _w1//2
        _out1 = sub_abs_dhash(arr1[0:_phoneme_width],
                                   arr2[0:_phoneme_width])
        _out2 = sub_abs_dhash(arr1[_phoneme_width:],
                                   arr2[_phoneme_width:])

        if (_out1 <= threshold) and (_out2 <= threshold):
            return (_out1 + _out2) / 2
        else:
            return max(_out1, _out2)   # 不同度

    if _w1 > _w2:
        _d = _w1 - _w2
        t_val = np.zeros(_d+1)
        for k in range(_d+1):
            t_val[k] = cal_dhash_diff_rate(arr1[:, k:k+_w2], arr2[:, 0:_w2])
        return np.min(t_val)
    else:
        _d = _w2 - _w1
        t_val = np.zeros(_d+1)
        for k in range(_d+1):
            t_val[k] = cal_dhash_diff_rate(arr1[:, 0:_w1], arr2[:, k:k+_w1])
        return np.min(t_val)


# def sub_abs_dhash(arr1, arr2):
#     arr = arr1 - arr2
#     _v, _w = arr.shape
#     if _v * _w > 0:
#         _sum = np.sum(abs(arr))/(_v*_w)
#         return _sum
#     else:
#         return 0.0
#
#
# def cal_dhash_diff_rate(arr1, arr2):
#     _v1, _w1 = arr1.shape
#     _v2, _w2 = arr2.shape
#     if _v1 != _v2:
#         print('vertical of cal_dhash_diff_rate(arr1, arr2) must be same')
#         return []
#     if _w1 == _w2:
#         _out = sub_abs_dhash(arr1, arr2)
#         return _out
#     if _w1 > _w2:
#         _d = _w1 - _w2
#         t_val = np.zeros(_d+1)
#         for k in range(_d+1):
#             t_val[k] = sub_abs_dhash(arr1[:, k:k+_w2], arr2[:, 0:_w2])
#         return np.min(t_val)
#     else:
#         _d = _w2 - _w1
#         t_val = np.zeros(_d+1)
#         for k in range(_d+1):
#             t_val[k] = sub_abs_dhash(arr1[:, 0:_w1], arr2[:, k:k+_w1])
#         return np.min(t_val)


def arr_sum_zoom_out(arr, h_step, v_step, width, height):
    _v, _h = arr.shape
    _wid = int(min(width, _h // h_step))  # 输出宽度
    _hei = int(min(height, _v // v_step))  # 输出高度
    _out = np.zeros((_hei, _wid))
    for line in range(_hei):
        for row in range(_wid):
            roi_left = row * h_step
            roi_right = roi_left + h_step + 1
            roi_bottom = line * v_step
            roi_top = roi_bottom + v_step + 1
            _out[line, row] = np.sum(arr[roi_bottom:roi_top,
                                     roi_left:roi_right])
    _out = _out/np.max(abs(_out))
    return _out


def cal_dhash(arr):
    _v, _h = arr.shape
    if (_v > 16) or (_h > 16):
        print("dhash img max shape is 16x16")
        return []
    else:
        _out = np.zeros((_v, _h))
        _out[0:_v-1, :] = arr[0:_v-1, :] - arr[1:_v, :]
        return _out


def std_dhash(arr, cut_rate=0.1):
    _out = arr/np.max(np.abs(arr))
    _out = np.where(abs(_out) > cut_rate, _out, 0)
    _out = np.sign(_out)
    return _out


def subtract_dhash(arr1, arr2):
    arr = arr1*arr2
    _v, _w = arr.shape
    arr = np.where(arr < 0, 1, 0)
    _sum = np.sum(arr*10)/(_v*_w)
    return _sum


def subtract_zoom_out(arr1, arr2, cut_rate=0.15):
    arr = arr1 - arr2
    _v, _w = arr.shape
    _rate = np.max(abs(arr))*cut_rate
    arr = np.where(abs(arr) >= _rate, arr, 0)
    _sum = np.sum(abs(arr))/(_v*_w)
    return _sum


def arr_similar_rate(arr1, arr2, cut_rate=0.1, simi_rate=0.85):
    _arr1 = arr1/np.max(abs(arr1))
    _arr2 = arr2/np.max(abs(arr2))
    _sum = _arr1 + _arr2
    _sum = np.where(_sum >= cut_rate, _sum/2, 1)
    _sub = _arr1 - _arr2
    _diff = _sub / _sum
    _sim = np.where(_diff <= simi_rate, 1.0, 0)
    _out = np.sum(_sim)/(arr1.shape[0]*arr1.shape[1])
    return _out


def get_simi_rst(simiarr, threshold=0.9):
    _out = np.where(simiarr >= threshold, 1, 0)
    return _out




def pre_init_paras(para_arr, file, version, ext_info=''):
    _nlen = len(para_arr)
    for i in range(_nlen):
        para_arr[i]["_version"] = version
        para_arr[i]["wav_file"] = file
        para_arr[i]["ext_info"] = ext_info


if __name__ == '__main__':
    arr_test = np.random.random((80, 80))
    arr_zoom_out = arr_sum_zoom_out(arr_test, 4, 4, 16, 16)
    print(arr_zoom_out.shape)
    dhash = cal_dhash(arr_zoom_out)
    print(dhash.shape)

    sign_dhash = std_dhash(dhash, 0.1)

    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.imshow(arr_test)
    plt.subplot(222)
    plt.imshow(arr_zoom_out)
    plt.subplot(223)
    plt.imshow(dhash)
    plt.subplot(224)
    plt.imshow(sign_dhash)

    plt.show()
