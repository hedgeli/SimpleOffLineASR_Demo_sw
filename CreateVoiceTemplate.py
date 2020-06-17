import numpy as np
import matplotlib.pyplot as plt
# import pack_global_vars as pgv
import wave
import os
import time
import shutil

from scipy.io.wavfile import write as wav_write
import VoiceSingalAnalysis as sigana

import WordVoiceParaSave as voice_para_save

g_wordpara_arr = np.zeros(24, dtype=voice_para_save.WordVoicePara)
# print('g_wordpara_arr shape: ', g_wordpara_arr.shape)
# print('g_wordpara_arr dtype: ', g_wordpara_arr.dtype)

debug_data = []

_frm_idx_log = []
_frm_stat_log = []

g_word_arr = []
g_word_idx = 0

g_check_word_arr = []
g_check_word_idx = 0


class VoiceFileSegInfo(object):
    def __init__(self, file=''):
        self.file = file
        self.fs = 16000  # 采样率
        self.len = 0
        self.sta_poi = 0
        self.end_poi = 0


def create_new_voice_fileseg(_poi, wav_file=''):
    global g_word_arr
    global g_word_idx
    _new_vfseg = VoiceFileSegInfo(wav_file)
    _new_vfseg.sta_poi = _poi
    g_word_arr.append(_new_vfseg)
    # print("create a new word:", g_word_idx, "poi:", _poi)


def finish_of_voice_fileseg(_poi):
    global g_word_arr
    global g_word_idx
    g_word_arr[g_word_idx].end_poi = _poi
    # print("finish of word: ", g_word_idx, "poi:", _poi)
    g_word_idx += 1


# 去除数据中的尖峰值
def delete_peak(datain, _max_step):
    _out = datain
    _len = len(datain)
    for i in range(1, _len - 1):
        if abs(datain[i - 1] - datain[i]) > _max_step:
            if abs(datain[i + 1] - datain[i]) > _max_step:
                _out[i] = (datain[i - 1] + datain[i + 1]) / 2
    return _out


def add_frm_stat_log(idx, stat):
    global _frm_idx_log
    global _frm_stat_log
    _frm_idx_log.append(idx)
    _frm_stat_log.append(stat)


def add_sum_log(sum):
    global _sum_log
    _sum_log.append(sum)


_log_idx = []
_log_info = []
_sum_log = []
change_stat_cnt = 0


def add_voi_ana_log(idx, info):
    global _log_idx
    global _log_info
    global change_stat_cnt
    _log_idx.append(idx)
    _log_info.append(info)
    change_stat_cnt += 1


voiless_eng_thread = 2
voiced_eng_thread = 10


def judge_voice_energy(data):
    _eng = sigana.calc_energy(data)
    _eng = _eng * 5
    if _eng < 1:
        _eng = 1
    _eng = sigana.calc_log(_eng)

    if _eng <= voiless_eng_thread:
        return _eng, "noise"
    elif _eng >= voiced_eng_thread:
        return _eng, "voied"
    else:
        return _eng, "voiless"


def frm_active_detect(self, data, s_time):
    if self.noise_info.noise_state != "noise_inited":
        print("Please call get_noise_info(data) before frm_active_detect(). ")
        return

    _judge = np.max(data) - np.min(data)
    if _judge >= self.voiced_rate * self.noise_info.noise_amp_p_n:                              #self.threshold_amp:
        if self.voice_stage == 'voi_clear':
            self.begin_time = s_time
            # print("voice begin in voiced: ", max(data))
            # print("ana_stage: ", self.ana_stage)
        self.voice_stage = "voiced"
        return "frm_active"
    # _len = len(data)
    # _delta = data[1:_len] - data[0:_len-1]
    # if max(_delta) >= self.active_threshold_delta:
    elif _judge >= self.voiness_rate * self.noise_info.noise_amp_p_n:
        if self.voice_stage == 'voi_clear':
            # print("voice_stage: ", self.voice_stage, max(data))
            self.voice_stage = 'voiceless'
            # print("voice begin in voiceless:", max(_delta))
            # print("voice_stage: ",  self.voice_stage)
            self.begin_time = s_time
        return "frm_active"
    else:
        return "frm_clear"



voiless_delta_threshold = 0.0035


def judge_voiless_delta(data):
    _data = sigana.calc_delta_n(data)
    # _val = np.max(_data)
    _val = np.sum(abs(_data)) / len(_data)
    if _val >= voiless_delta_threshold:
        return _val, "voiless"
    else:
        return _val, "noise"


def create_words_template_fromwav(wav, wordlist=[], Fs=16000):
    frm_time_width = 0.02  # 20ms
    analy_sta = "noinit"
    ana_frm_cnt = 0

    wav_data = np.array(wav)
    ana_frm_width = int(frm_time_width * Fs)
    ana_frm_step = ana_frm_width // 2

    noise_tmpt_frms = 6

    noise_fft_tmpt = np.zeros((noise_tmpt_frms, ana_frm_step))
    noise_freq_tmpt = np.zeros((noise_tmpt_frms, ana_frm_step))

    # 保存旧参数文件
    wavfile = "./ResFiles/Wav/Word_voice_tmpt.wav"
    time_ymd_hms = time.strftime("_%Y%m%d_%H%M%S", time.localtime())
    new_wavfile = wavfile[0:-4] + time_ymd_hms + ".wav"
    para_file = wavfile[0:-4] + 'WordPara' + ".npz"
    new_para_file = para_file[0:-4] + time_ymd_hms + ".npz"

    shutil.copyfile(wavfile, new_wavfile)
    shutil.copyfile(para_file, new_para_file)

    write_data = np.array(wav_data, dtype='int16')
    wav_write(wavfile, Fs, write_data)

    wav_data = sigana.normalization_pn1(wav_data)

    # print(len(wav_data))
    # print(wav_data[0:200])

    while analy_sta == "noinit":
        ana_poi = ana_frm_cnt * ana_frm_step
        _frm = sigana.get_frame(wav_data, ana_poi, ana_frm_width)
        _nfft, _freq_y = sigana.calc_fft(_frm, Fs)
        noise_fft_tmpt[ana_frm_cnt] = _nfft
        noise_freq_tmpt[ana_frm_cnt] = _freq_y
        ana_frm_cnt += 1

        if ana_frm_cnt >= noise_tmpt_frms:
            analy_sta = "noise"
            #             add_voi_ana_log(ana_poi, analy_sta)
            break

        else:
            pass

    while True:
        ana_poi = ana_frm_cnt * ana_frm_step
        ana_frm_cnt += 1
        _end_poi = ana_poi + ana_frm_width
        if _end_poi > len(wav_data):
            analy_sta = "finished"
            break

        _frm = sigana.get_frame(wav_data, ana_poi, ana_frm_width)

        _sum, _judge = judge_voice_energy(_frm)

        if _judge == "voiless":
            # 根据短时能量判定为清音
            if analy_sta != "voiless":
                if analy_sta == "noise":
                    create_new_voice_fileseg(ana_poi)

                analy_sta = "voiless"
                add_voi_ana_log(ana_poi, analy_sta)
                # add_sum_log(_sum)
                add_sum_log(0.5)
        elif _judge == "voied":
            # 根据短时能量判定为浊音
            if analy_sta != "voied":
                if analy_sta == "noise":
                    create_new_voice_fileseg(ana_poi)

                analy_sta = "voied"
                add_voi_ana_log(ana_poi, analy_sta)
                # add_sum_log(_sum)
                add_sum_log(1)
        else:
            # 根据一阶差分值判别清音和噪音
            _sum, _judge = judge_voiless_delta(_frm)

            if _judge == "noise":
                if analy_sta != "noise":
                    # if ana_poi - g_word_arr[g_word_idx].sta_poi >= 400:
                    finish_of_voice_fileseg(ana_poi)
                    analy_sta = "noise"
                    add_voi_ana_log(ana_poi, analy_sta)
                    # add_sum_log(_sum)
                    add_sum_log(0.2)
            elif _judge == "voiless":
                if analy_sta != "voiless":
                    if analy_sta == "noise":
                        create_new_voice_fileseg(ana_poi)

                    analy_sta = "voiless"
                    add_voi_ana_log(ana_poi, analy_sta)
                    # add_sum_log(_sum)
                    add_sum_log(0.5)
            else:
                pass

        if analy_sta == "voiless":
            add_frm_stat_log(ana_poi, 0.25)
        elif analy_sta == "voied":
            add_frm_stat_log(ana_poi, 0.5)
        elif analy_sta == "noise":
            add_frm_stat_log(ana_poi, -0.15)

    print("len(g_word_arr): ", len(g_word_arr))
    print("g_word_idx: ", g_word_idx)

    MIN_WORD_SPACE = 2400
    global g_check_word_idx
    global g_check_word_arr

    for i in range(len(g_word_arr)):
        if i == 0:
            _new_vfseg = VoiceFileSegInfo(wavfile)
            _new_vfseg.sta_poi = g_word_arr[i].sta_poi
            _new_vfseg.end_poi = g_word_arr[i].end_poi
            g_check_word_arr.append(_new_vfseg)
            # g_check_word_idx = g_check_word_idx + 1
        # if (g_word_arr[i].end_poi - g_word_arr[i].sta_poi) >= 640:
        if i >= 1:
            if (g_word_arr[i].sta_poi - g_check_word_arr[g_check_word_idx].end_poi) >= MIN_WORD_SPACE:
                g_check_word_arr[g_check_word_idx].end_poi = g_word_arr[i - 1].end_poi
                # create a new vfseg for new word
                _new_vfseg2 = VoiceFileSegInfo(wavfile)
                _new_vfseg2.sta_poi = g_word_arr[i].sta_poi
                _new_vfseg2.end_poi = g_word_arr[i].end_poi
                g_check_word_arr.append(_new_vfseg2)
                g_check_word_idx = g_check_word_idx + 1


            else:
                g_check_word_arr[g_check_word_idx].end_poi = g_word_arr[i].end_poi

        if i == len(g_word_arr) - 1:
            g_check_word_arr[g_check_word_idx].end_poi = g_word_arr[i].end_poi

    print("len(g_word_arr): ", len(g_word_arr))
    print("len(g_check_word_arr): ", len(g_check_word_arr))

    save_pars = []

    _word_n = len(g_check_word_arr)
    if _word_n > len(wordlist):
        _word_n = len(wordlist)

    for i in range(_word_n):

        _disp_data = wav_data[g_check_word_arr[i].sta_poi:g_check_word_arr[i].end_poi]

        if len(_disp_data) >= 2400:
            move_time = 0.01
            num_frms = int(len(_disp_data) / (16000 * move_time))
            _fft, _freq = sigana.cal_frames_fft_log(_disp_data * 10000, 16000, t_start=0,
                                                    t_window=0.02, frame_n=num_frms,
                                                    move_step=move_time)
            _fft = sigana.normalization_array_pn1(_fft)

            save_pars.append(_fft)

            voice_para_save.save_word_voice_para(g_wordpara_arr[i], _fft,
                                                 word=wordlist[i], file=wavfile,
                                                 s_idx=g_check_word_arr[i].sta_poi,
                                                 e_idx=g_check_word_arr[i].end_poi,
                                                 time=0.0,
                                                 ext_info="")

    para_file_name = wavfile[0:-4] + 'WordPara' + ".npz"
    np.savez(para_file_name, g_wordpara_arr)



def create_voi_template_fromfile(wavfile, wordlist=[], paras_save=True):
    word_cnter = 0
    frm_time_width = 0.02  # 20ms
    voi_sta = "clr_or_noise"
    analy_sta = "noinit"
    ana_poi = 0
    ana_frm_cnt = 0

    nchs, sampdepth, Fs, npoints, wav_data = sigana.load_wav(wavfile)
    # utdisp.show_curve(wav_data)
    ana_frm_width = int(frm_time_width * Fs)
    ana_frm_step = ana_frm_width // 2

    noise_tmpt_frms = 6

    noise_fft_tmpt = np.zeros((noise_tmpt_frms, ana_frm_step))
    noise_freq_tmpt = np.zeros((noise_tmpt_frms, ana_frm_step))

    voiless_t_sta = 0
    voiless_t_end = 0

    voied_t_sta = 0
    voied_t_end = 0

    wav_data = sigana.normalization_pn1(wav_data)
    if nchs != 1:
        print("The Wav file must be single channel !")
        return []

    while analy_sta == "noinit":
        ana_poi = ana_frm_cnt * ana_frm_step
        _frm = sigana.get_frame(wav_data, ana_poi, ana_frm_width)
        _nfft, _freq_y = sigana.calc_fft(_frm, Fs)
        noise_fft_tmpt[ana_frm_cnt] = _nfft
        noise_freq_tmpt[ana_frm_cnt] = _freq_y
        ana_frm_cnt += 1

        if ana_frm_cnt >= noise_tmpt_frms:
            analy_sta = "noise"
            #             add_voi_ana_log(ana_poi, analy_sta)
            break

        else:
            pass

    while True:
        ana_poi = ana_frm_cnt * ana_frm_step
        ana_frm_cnt += 1
        _end_poi = ana_poi + ana_frm_width
        if _end_poi > len(wav_data):
            analy_sta = "finished"
            #             add_voi_ana_log(ana_poi, analy_sta)
            break

        _frm = sigana.get_frame(wav_data, ana_poi, ana_frm_width)

        #         _nfft, _freq_y = spec.calc_fft(_frm, Fs)
        #         _sum, _judge = judge_voice_fft(_nfft, _freq_y)

        _sum, _judge = judge_voice_energy(_frm)
        # debug_data.append(_sum)

        if _judge == "voiless":
            # 根据短时能量判定为清音
            if analy_sta != "voiless":
                if analy_sta == "noise":
                    create_new_voice_fileseg(ana_poi)

                analy_sta = "voiless"
                add_voi_ana_log(ana_poi, analy_sta)
                # add_sum_log(_sum)
                add_sum_log(0.5)
        elif _judge == "voied":
            # 根据短时能量判定为浊音
            if analy_sta != "voied":
                if analy_sta == "noise":
                    create_new_voice_fileseg(ana_poi)

                analy_sta = "voied"
                add_voi_ana_log(ana_poi, analy_sta)
                # add_sum_log(_sum)
                add_sum_log(1)
        else:
            # 根据一阶差分值判别清音和噪音
            _sum, _judge = judge_voiless_delta(_frm)

            if _judge == "noise":
                if analy_sta != "noise":
                    # if ana_poi - g_word_arr[g_word_idx].sta_poi >= 400:
                    finish_of_voice_fileseg(ana_poi)
                    analy_sta = "noise"
                    add_voi_ana_log(ana_poi, analy_sta)
                    # add_sum_log(_sum)
                    add_sum_log(0.2)
            elif _judge == "voiless":
                if analy_sta != "voiless":
                    if analy_sta == "noise":
                        create_new_voice_fileseg(ana_poi)

                    analy_sta = "voiless"
                    add_voi_ana_log(ana_poi, analy_sta)
                    # add_sum_log(_sum)
                    add_sum_log(0.5)
            else:
                pass

        if analy_sta == "voiless":
            add_frm_stat_log(ana_poi, 0.25)
        elif analy_sta == "voied":
            add_frm_stat_log(ana_poi, 0.5)
        elif analy_sta == "noise":
            add_frm_stat_log(ana_poi, -0.15)

    print("len(g_word_arr): ", len(g_word_arr))
    print("g_word_idx: ", g_word_idx)

    MIN_WORD_WIDTH = 2400  # 词最小语音长度0.2S

    MIN_WORD_SPACE = 2400
    global g_check_word_idx
    global g_check_word_arr

    for i in range(len(g_word_arr)):
        if i == 0:
            _new_vfseg = VoiceFileSegInfo(wavfile)
            _new_vfseg.sta_poi = g_word_arr[i].sta_poi
            _new_vfseg.end_poi = g_word_arr[i].end_poi
            g_check_word_arr.append(_new_vfseg)
            # g_check_word_idx = g_check_word_idx + 1
        # if (g_word_arr[i].end_poi - g_word_arr[i].sta_poi) >= 640:
        if i >= 1:
            if (g_word_arr[i].sta_poi - g_check_word_arr[g_check_word_idx].end_poi) >= MIN_WORD_SPACE:
                g_check_word_arr[g_check_word_idx].end_poi = g_word_arr[i - 1].end_poi
                # create a new vfseg for new word
                _new_vfseg2 = VoiceFileSegInfo(wavfile)
                _new_vfseg2.sta_poi = g_word_arr[i].sta_poi
                _new_vfseg2.end_poi = g_word_arr[i].end_poi
                g_check_word_arr.append(_new_vfseg2)
                g_check_word_idx = g_check_word_idx + 1


            else:
                g_check_word_arr[g_check_word_idx].end_poi = g_word_arr[i].end_poi

        if i == len(g_word_arr) - 1:
            g_check_word_arr[g_check_word_idx].end_poi = g_word_arr[i].end_poi

    print("len(g_word_arr): ", len(g_word_arr))
    print("len(g_check_word_arr): ", len(g_check_word_arr))

    # for i in range(len(g_check_word_arr)):
    #     # print("---------checked word idx: ---------", i)
    #     # print("word sta: ", g_check_word_arr[i].sta_poi)
    #     # print("word end ", g_check_word_arr[i].end_poi)
    #     print(i, "  word width: ", g_check_word_arr[i].end_poi - g_check_word_arr[i].sta_poi)

    save_pars = []

    plt.figure()
    plt.title(wavfile)
    for i in range(12):
        # plt.subplot(3, 4, i + 1)

        _disp_data = wav_data[g_check_word_arr[i].sta_poi:g_check_word_arr[i].end_poi]

        # # print("sta poi: ", g_word_arr[i].sta_poi)
        # # print("end poi: ", g_word_arr[i].end_poi)
        # #
        # plt.plot(_disp_data, label=str(i))
        #
        # _pitch, _pos = spec.calc_picths(_disp_data, 320, 160, 13)
        # _pitch = delete_peak(_pitch, 10)
        # plt.plot(_pos, _pitch, "g.-", label=str(i))
        # plt.ylim((0, 100))
        # plt.grid(True)
        #
        # plt.legend()

        # print("i:", i, "len(_disp_data): ", len(_disp_data))
        #
        if len(_disp_data) >= 2400:
            move_time = 0.01
            num_frms = int(len(_disp_data) / (16000 * move_time))
            _fft, _freq = sigana.cal_frames_fft_log(_disp_data * 10000, 16000, t_start=0,
                                                    t_window=0.02, frame_n=num_frms,
                                                    move_step=move_time)
            _fft = sigana.normalization_array_pn1(_fft)
            # plt.imshow(_fft.T, origin="lower", extent=(0, 50, 0, 50))
            # plt.imshow(_fft.T, origin="lower")
            # plt.title(wordlist[i])

            save_pars.append(_fft)
            # print(_fft.shape)

            voice_para_save.save_word_voice_para(g_wordpara_arr[i], _fft,
                                                 word=wordlist[i], file=wavfile,
                                                 s_idx=g_check_word_arr[i].sta_poi,
                                                 e_idx=g_check_word_arr[i].end_poi,
                                                 time=0.0,
                                                 ext_info="")

    # plt.show()

    # para_file_name = wavfile[0:-4] + ".npz"
    # print("save paras in file: ", para_file_name)
    # np.savez(para_file_name, save_pars)

    para_file_name = wavfile[0:-4] + 'WordPara' + ".npz"
    np.savez(para_file_name, g_wordpara_arr)

    # for i in range(12):
    #     plt.subplot(3, 4, i + 1)
    #     # plt.imshow(g_wordpara_arr[i]['specgram'], origin="lower")
    #     # print(g_wordpara_arr[i]['_zoom_width'])
    #     plt.imshow(g_wordpara_arr[i]['spec_zoom_out'].T, origin="lower")
    #     # print(g_wordpara_arr[i]['spec_zoom_out'].T[0, :])
    #     # plt.imshow(g_wordpara_arr[i]['spec_dhash'], origin="lower")
    #     plt.title(g_wordpara_arr[i]['word'])
    #
    # plt.show()

    subs = np.zeros((4, 12, 12))
    for k in range(4):
        for i in range(12):
            for j in range(12):
                # subs[i, j] = voice_para_save.subtract_dhash(g_wordpara_arr[i]['spec_dhash'],
                #                                             g_wordpara_arr[j]['spec_dhash'])
                _wid_i = g_wordpara_arr[i]['_zoom_width']
                _wid_j = g_wordpara_arr[j]['_zoom_width']
                _wid = min(_wid_j, _wid_i)
                # print(g_wordpara_arr[i]['spec_zoom_out'].shape)
                subs[k, i, j] = voice_para_save.subtract_zoom_out(
                    g_wordpara_arr[i]['spec_zoom_out'][:, k * 4:k * 4 + 4],
                    g_wordpara_arr[j]['spec_zoom_out'][:, k * 4:k * 4 + 4])

                # subs[i, j] = voice_para_save.subtract_dhash(g_wordpara_arr[i]['spec_dhash'][:, 0:_wid],
                #                                             g_wordpara_arr[j]['spec_dhash'][:, 0:_wid])

    # for i in range(4):
    #     np.set_printoptions(precision=2)
    #     print(subs[i, :, :])
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(subs[i])
    # plt.show()

    # word_division_line_x = []
    # word_division_line_y = []
    #
    # for i in range(len(g_check_word_arr)):
    #     word_division_line_x.append(g_check_word_arr[i].sta_poi)
    #     word_division_line_y.append(-0.5)
    #     word_division_line_x.append(g_check_word_arr[i].sta_poi+1)
    #     word_division_line_y.append(0.5)
    #
    #     word_division_line_x.append(g_check_word_arr[i].end_poi)
    #     word_division_line_y.append(0.5)
    #     word_division_line_x.append(g_check_word_arr[i].end_poi + 1)
    #     word_division_line_y.append(-0.5)
    #
    #
    # plt.figure(figsize=(10, 5))
    # # plt.subplot(211)
    # plt.plot(wav_data, "g-")
    # plt.xlim((0, npoints))
    # plt.grid(True)
    # # plt.subplot(212)
    # _sta_val = np.array(_frm_stat_log)
    # plt.plot(np.array(word_division_line_x), np.array(word_division_line_y), "b-")
    # plt.xlim((0, npoints))
    # plt.title(wavfile)
    # plt.grid(True)
    # plt.show()


def debug_self():
    # wav_file = "./ResFiles/Wav/cmd_01_06_启动_停止_前进_后退_左转_右转.wav"
    # cmd_list = ["启动", "启动", "停止", "停止", "前进", "前进", "后退", "后退",
    #             "左转", "左转", "右转", "右转"]

    # wav_file = "./ResFiles/Wav/cmd_07_12_开机_关机_开灯_关灯_开门_关门.wav"
    # cmd_list = ["开机", "开机", "关机", "关机", "开灯", "开灯", "关灯", "关灯",
    #             "开门", "开门", "关门", "关门"]

    # wav_file = "./ResFiles/Wav/启动停止前进后退左转右转_seqx2.wav"
    # wav_file = "./ResFiles/Wav/启动停止前进后退左转右转_seqx2_reduNoise.wav"
    # cmd_list = ["启动", "停止", "前进", "后退", "左转", "右转", "启动", "停止", "前进", "后退",
    #             "左转", "右转"]

    # wav_file = "./ResFiles/Wav/左转_右转x6_redu.wav"
    # cmd_list = ["左转", "左转", "左转", "左转", "左转", "左转",
    #             "右转", "右转", "右转", "右转", "右转", "右转"]

    wav_file = "./ResFiles/Wav/mic_开机关机启动停止前进后退左转右转提高降低加速减速.wav"
    cmd_list = ['开机',  '关机', '启动',  '停止', '前进',  '后退',
                '左转',  '右转', '提高',  '降低', '加速',  '减速']

    # wav_file = "./ResFiles/Wav/start_stop_forward_backup_turnleft_turnright.wav"
    # cmd_list = ["start", "start", "stop", "stop", "forward", "forward", "backup", "backup",
    #             "turn left", "turn left", "turn right", "turn right"]

    create_voi_template_fromfile(wav_file, cmd_list)


# def test():
#     print("Start CreateVoiceTemplate.py test...")


if __name__ == '__main__':
        debug_self()
