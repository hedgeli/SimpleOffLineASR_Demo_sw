import numpy as np
import VoiceSingalAnalysis as sigana
import WordVoiceParaSave as voice_para_save


class NoiseInfo(object):
    def __init__(self):
        self.noise_fft_arr = np.zeros((5, 80, 80))
        self.tmpt_noise_fft = np.zeros((80, 80))
        self.noise_state = 'noise_noinit'
        self.Fs = 16000
        self.noise_amp_p_n = 0.0
        self.noise_frame_cnt = 0
        self.noise_frame_len = int(self.Fs*0.02)

    def get_noise_info(self, data):
        if self.noise_state == 'noise_inited':
            print("noise info has inited. Please call noise_info_reset(data) if needed.")
            return
        _frm_n = len(data)//self.noise_frame_len
        noise_amp_p_n = np.zeros((5,), dtype='float')
        for i in range(_frm_n):
            _idx_0 = i * self.noise_frame_len
            _idx_1 = _idx_0 + self.noise_frame_len
            _frm_data = data[_idx_0:_idx_1]
            noise_amp_p_n[self.noise_frame_cnt] = np.max(_frm_data) - np.min(_frm_data)

            self.noise_frame_cnt = self.noise_frame_cnt + 1
            if self.noise_frame_cnt >= 5:
                self.noise_amp_p_n = np.sum(noise_amp_p_n)/len(noise_amp_p_n)
                self.noise_state = 'noise_inited'
                break

    def noise_info_reset(self, data):
        # self.noise_state = 'noise_noinit'
        # self.noise_frame_cnt = 0
        self.__init__()
        self.get_noise_info(data)


    def get_noise_fft(self):
        return

    def noise_reduce(self, wavin):
        return


class VoiceSlice(object):
    def __init__(self):
        self.time_start = 0
        self.npoints = 0
        self.data = np.zeros(1)

    def set_data(self, datain):
        self.data = np.array(datain)


class VoiRealTimeAnalysis(object):

    def __init__(self):
        self.Fs = 16000
        self.data = []
        self.min_word_len = 0.25*self.Fs
        self.begin_time = 0.0
        self.end_time = 0.0
        self.voice_stage = "voi_clear"
        self.voiless_cnt = 0
        self.voiclear_cnt = 0
        self.ana_time = 0.0
        self.ana_idx = 0

        self.frame_len = 320
        self.frame_step = 160

        self.noise_info = NoiseInfo()

        # self.threshold_amp = 250                # 浊音幅值门限
        # self.active_threshold_delta = 100       # 清音幅值门限,
        # self.noise_amp = 50                     # 噪音幅值门限
        self.voiced_rate = 6.0
        self.voiness_rate = 3.0

        self.max_voiclear_frm_len = 15

        self.active_idx_arr = np.zeros((5, 2), dtype=np.int)

        self.word_fft = np.zeros((80, 80))
        self.fft_zoom_out = np.zeros((16, 16))
        self.spec_dhash = np.zeros((16, 16))
        self.spec_dhash_width = 0

        self.ana_stage = 'voi_ana_finished'

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

    def get_voice_word_data(self, data, begin_time):
        if self.noise_info.noise_state != "noise_inited":
            print("Please call get_noise_info(data) before get_voice_word_data(). ")
            return
        if self.ana_stage != 'voi_ana_finished':
            return
        _len = len(data)
        _frm_n = (_len-self.frame_step)//self.frame_step
        for i in range(_frm_n):
            _idx_s = i * self.frame_step
            _frm_s_time = begin_time + _idx_s/self.Fs
            _idx_e = _idx_s+self.frame_len
            frm_data = data[_idx_s:_idx_e]
            if self.frm_active_detect(frm_data, _frm_s_time) == 'frm_active':
                self.data = np.hstack((self.data, frm_data[0:self.frame_step]))
            else:
                if self.voice_stage != 'voi_clear':
                    self.data = np.hstack((self.data, frm_data[0:self.frame_step]))
                self.voiclear_cnt += 1
                if self.voiclear_cnt >= self.max_voiclear_frm_len:
                    if self.voice_stage != 'voi_clear':
                        self.ana_stage = "voi_ana_get_data"
                        self.end_time = begin_time
                        self.voice_stage = 'voi_clear'
                        # print("voice stage change to voi_clear")
                        # print('get voice data, need to recognize(time S): %2.3f' % (self.end_time - self.begin_time))

                    self.voiclear_cnt = 0


    def cal_spec_info(self):
        if self.noise_info.noise_state != "noise_inited":
            print("Please call get_noise_info(data) before cal_spec_info(). ")
            return
        _data = np.array(self.data)
        _data = sigana.normalization_pn1(_data)

        if len(_data) < self.min_word_len:
            self.data = []
            # self.word_fft = np.zeros((80, 80))
            # self.fft_zoom_out = np.zeros((16, 16))
            # self.spec_dhash = np.zeros((16, 16))
        else:      # len(_data) > self.min_word_len:
            move_time = 0.01
            num_frms = int(len(_data) / (self.Fs * move_time))
            _fft, _freq = sigana.cal_frames_fft_log(_data * 10000, self.Fs, t_start=0,
                                                    t_window=0.02, frame_n=num_frms,
                                                    move_step=move_time)
            _fft = sigana.normalization_array_pn1(_fft)

            self.word_fft = np.zeros((80, 80))
            if _fft.shape[0] > 80:
                self.word_fft = _fft[0:80, 0:80].T
            else:
                _v = _fft.shape[0]
                self.word_fft[0:80, 0:_v] = _fft[0:_v, 0:80].T

            # print('self.word_fft.shape: ', self.word_fft.shape)

            _zoom_out = voice_para_save.arr_sum_zoom_out(self.word_fft, 4, 4, 16, 16)

            self.fft_zoom_out = np.zeros((16, 16))
            _v, _h = _zoom_out.shape
            self.spec_dhash_width = _h
            if _v > 16:
                _v = 16
            if _h > 16:
                _h = 16
            self.fft_zoom_out[0:_v, 0:_h] = _zoom_out[0:_v, 0:_h]

            _dhash = voice_para_save.cal_dhash(_zoom_out)

            _sign_dhash = voice_para_save.std_dhash(_dhash, 0.06)

            self.spec_dhash = np.zeros((16, 16))
            _v, _h = _sign_dhash.shape
            if _v > 16:
                _v = 16
            if _h > 16:
                _h = 16
            self.spec_dhash[0:_v, 0:_h] = _sign_dhash[0:_v, 0:_h]


if __name__ == '__main__':
    voi_ana = VoiRealTimeAnalysis()
    print(voi_ana.active_idx_arr)
    print(voi_ana.active_idx_arr.dtype)

    voi_slice = VoiceSlice()
    voi_slice.set_data(np.array(np.arange(1, 100)))
    print(voi_slice.data.dtype)
    # print(voi_slice.data)


