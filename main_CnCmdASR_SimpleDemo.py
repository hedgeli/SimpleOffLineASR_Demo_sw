# class taken from the SciPy 2015 Vispy talk opening example
# see https://github.com/vispy/vispy/pull/928
# https://flothesof.github.io/pyqt-microphone-fft-application.html
# https://gist.github.com/nagordon/933b2c835597972de29e06b68de72e44
import pyaudio
import threading
import atexit
import numpy as np
import time
import matplotlib

import voice_realtime_analysis as voi_ana
import WordVoiceParaSave as voice_para_save
import CreateVoiceTemplate as cre_voi_tmpt

# 预设字体格式，并传给rc方法
font = {'family': 'SimHei', "size": 10}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.unicode_minus'] = False
np.set_printoptions(precision=3)


class MicrophoneRecorder(object):
    def __init__(self, rate=16000, chunksize=4096):
        self.rate = rate
        self.chunksize = chunksize
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.new_frame)
        self.lock = threading.Lock()
        self.stop = False
        self.frames = []
        self.voi_record = []
        atexit.register(self.close)

    def new_frame(self, data, frame_count, time_info, status):
        # data = np.fromstring(data, 'int16')
        data = np.frombuffer(data, 'int16')
        with self.lock:
            self.frames.append(data)
            if self.stop:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def get_frames(self):
        with self.lock:
            frames = self.frames
            # Todo record voice frames ???
            # self.voi_record.extend(frames)
            # print(frames)
            # print(len(frames))
            # print(self.voi_record.shape)
            # if len(self.voi_record) >= 160:
            #     self.voi_record = []
            #     print("clear record voice")
            self.frames = []
            return frames

    def start(self):
        self.stream.start_stream()

    def close(self):
        with self.lock:
            self.stop = True
        self.stream.close()
        self.p.terminate()


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

plt.switch_backend('Qt5Agg')


class MplFigure(object):
    def __init__(self, parent):
        self.figure = plt.figure(facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, parent)


import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox


class LiveFFTWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        # init class data
        self.initData()

        # customize the UI
        self.initUI()

        # connect slots
        self.connectSlots()

        # init MPL widget
        self.initMplWidget()

        #
        self.get_data_time_stamp = time.time()
        self.start_time_stamp = time.time()

        self.voi_ana_01 = voi_ana.VoiRealTimeAnalysis()

    def clicked_btn_pause(self):
        if self.pause == "pause":
            self.pause = "run"
            self.btn_pause.setText("识别")
        else:
            self.pause = "pause"
            self.btn_pause.setText("暂停")

    def clicked_btn_create_tmpt(self):
        print("clicked_btn_create_tmpt")
        if self.record_voice_for_tmpt == "NO":
            reply = QMessageBox.question(self, 'Message', 'Sure to create new voice template?\n确认创建命令词语音模板？',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:

                self.record_voice_for_tmpt = "YES"
                self.btn_create_tmpt.setText("正在录音")

        else:
            if len(self.word_record_data) > self.voi_ana_01.Fs*3:
                print("Begin create words info template...")
                word_list = self.edit_cmd.text()
                # print("line_edit_cmd:", word_list)
                t_srt = word_list.replace(" ", "")
                # print("edit list: ", t_srt.strip(',').split(','))
                # print("tmpt_list: ", self.word_tmpt_list)
                self.word_tmpt_list = t_srt.strip(',').split(',')
                self.word_cmd_N = len(self.word_tmpt_list)
                cre_voi_tmpt.create_words_template_fromwav(self.word_record_data, self.word_tmpt_list)
                self.tmpt_voice_pars = np.load(self.tmpt_file)
                # for i in range(20):
                #     word_cmd = self.tmpt_voice_pars['arr_0'][i]['word']
                #     if word_cmd != "":
                #         self.word_tmpt_list.append(word_cmd)
                #     self.word_cmd_N = len(self.word_tmpt_list)
            self.btn_create_tmpt.setText("命令词\n录音")
            self.record_voice_for_tmpt = "NO"

    def initUI(self):
        hbox_gain = QtWidgets.QHBoxLayout()
        # autoGain = QtWidgets.QLabel('Auto gain')
        # autoGainCheckBox = QtWidgets.QCheckBox(checked=True)

        self.btn_create_tmpt = QtWidgets.QPushButton()
        self.btn_create_tmpt.setFont(QtGui.QFont("Arial Font", 10, QtGui.QFont.Bold))
        self.btn_create_tmpt.setText("命令词\n录音")
        self.btn_create_tmpt.setFixedSize(80, 60)
        self.btn_create_tmpt.clicked.connect(self.clicked_btn_create_tmpt)

        self.btn_pause = QtWidgets.QPushButton()
        self.btn_pause.setFont(QtGui.QFont("Arial Font", 16, QtGui.QFont.Bold))
        self.btn_pause.setText("暂停")
        self.btn_pause.setFixedSize(80, 60)
        self.btn_pause.clicked.connect(self.clicked_btn_pause)


        self.label_word = QtWidgets.QLabel()
        self.label_word.setFont(QtGui.QFont("Arial Font", 24, QtGui.QFont.Bold))
        self.label_word.setText("XX")
        # label_word.setGeometry(0, 0, 60, 150)
        self.label_word.setFixedSize(150, 60)

        self.word_cnt = 0
        self.label_cnt = QtWidgets.QLabel()
        self.label_cnt.setFont(QtGui.QFont("Arial Font", 24, QtGui.QFont.Bold))
        self.label_cnt.setText(str(self.word_cnt))
        # label_word.setGeometry(0, 0, 60, 150)
        self.label_cnt.setFixedSize(150, 60)

        label_discard = QtWidgets.QLabel("丢弃帧数: ")
        label_discard.setFixedSize(80, 40)
        self.label_discard_cnt = QtWidgets.QLabel("0")
        self.label_discard_cnt.setFont(QtGui.QFont("Arial Font", 16, QtGui.QFont.Bold))
        self.label_discard_cnt.setFixedSize(30, 40)

        label_err = QtWidgets.QLabel("未匹配次数: ")
        label_err.setFixedSize(80, 40)
        self.label_err_cnt = QtWidgets.QLabel("0")
        self.label_err_cnt.setFont(QtGui.QFont("Arial Font", 16, QtGui.QFont.Bold))
        self.label_err_cnt.setFixedSize(40, 40)

        # hbox_gain.addWidget(autoGain)
        hbox_gain.addWidget(self.btn_create_tmpt)
        hbox_gain.addWidget(self.btn_pause)
        hbox_gain.addWidget(self.label_word)
        hbox_gain.addWidget(self.label_cnt)
        hbox_gain.addWidget(label_err)
        hbox_gain.addWidget(self.label_err_cnt)
        hbox_gain.addWidget(label_discard)
        hbox_gain.addWidget(self.label_discard_cnt)


        # # reference to checkbox
        # self.autoGainCheckBox = autoGainCheckBox

        hbox_cmd_list = QtWidgets.QHBoxLayout()
        cmd_label = QtWidgets.QLabel('命令词列表: ')
        cmd_label.setFixedSize(120, 40)
        self.edit_cmd = QtWidgets.QLineEdit()
        self.edit_cmd.setText(", ".join(self.word_tmpt_list))
        self.edit_cmd.setFixedSize(800, 40)
        hbox_cmd_list.addWidget(cmd_label)
        hbox_cmd_list.addWidget(self.edit_cmd)

        label_simirate = QtWidgets.QLabel('相似系数: ')
        label_simirate.setFixedSize(120, 40)
        self.disp_simirate = QtWidgets.QLabel('0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00')
        self.disp_simirate.setFixedSize(800, 40)
        hbox_simirate = QtWidgets.QHBoxLayout()
        hbox_simirate.addWidget(label_simirate)
        hbox_simirate.addWidget(self.disp_simirate)

        vbox = QtWidgets.QVBoxLayout()

        vbox.addLayout(hbox_gain)
        vbox.addLayout(hbox_cmd_list)
        vbox.addLayout(hbox_simirate)

        # mpl figure
        self.main_figure = MplFigure(self)
        vbox.addWidget(self.main_figure.toolbar)
        vbox.addWidget(self.main_figure.canvas)

        self.setLayout(vbox)

        self.setGeometry(500, 100, 800, 600)
        self.setWindowTitle('汉语/孤立词特定人/离线语音识别/SimpleDemo')
        self.show()
        # timer for callbacks, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        timer.start(100)
        # keep reference to timer
        self.timer = timer

    def initData(self):
        mic = MicrophoneRecorder()
        mic.start()

        self.pause = "pause"

        # keeps reference to mic
        self.mic = mic

        self.record_voice_for_tmpt = "NO"

        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize,
                                         1. / mic.rate)
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000
        self.disp_data = []

        self.discard_cnt = 0
        self.err_cnt = 0

        self.word_record_data = []
        self.word_tmpt_list = []
        self.word_cmd_N = len(self.word_tmpt_list)

        # self.tmpt_file = "./ResFiles/Paras/mic_开机关机启动停止前进后退左转右转提高降低加速减速WordPara.npz"
        self.tmpt_file = "./ResFiles/Wav/Word_voice_tmptWordPara.npz"

        try:
            self.tmpt_voice_pars = np.load(self.tmpt_file)
            for i in range(20):
                word_cmd = self.tmpt_voice_pars['arr_0'][i]['word']
                if word_cmd != "":
                    self.word_tmpt_list.append(word_cmd)
                self.word_cmd_N = len(self.word_tmpt_list)
        except FileNotFoundError as e:
            self.tmpt_voice_pars = np.array([])
            print("Excetp : ", e)


    def connectSlots(self):
        pass

    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps
        references for further use"""
        # top plot
        self.ax_top = self.main_figure.figure.add_subplot(211)
        self.ax_top.set_ylim(-3000, 3000)
        # self.ax_top.set_xlim(0, self.time_vect.max())
        self.ax_top.set_xlim(0, 2006)
        self.ax_top.grid(True)
        self.ax_top.set_xlabel(u'time (ms)')

        # bottom plot 1
        self.ax_bottom = self.main_figure.figure.add_subplot(246)
        self.ax_bottom.set_ylim(0, 80)
        # self.ax_bottom.set_xlim(0, self.freq_vect.max())
        self.ax_bottom.set_xlim(0, 80)
        self.ax_bottom.grid(True)
        self.ax_bottom.set_xlabel(u'语谱图')            #, fontsize=8

        # bottom plot 2
        self.ax_voice = self.main_figure.figure.add_subplot(245)
        self.ax_voice.set_ylim(0, 1)
        self.ax_voice.set_xlim(0, 4000)
        self.ax_voice.grid(True)
        self.ax_voice.set_xlabel(u'语音波形')     #, fontsize=8

        # bottom plot 3
        self.ax_spec = self.main_figure.figure.add_subplot(247)
        self.ax_spec.set_ylim(-0.5, 15.5)
        self.img_spec = np.random.random((16, 16))
        self.ax_spec.set_xlim(-0.5, 15.5)
        self.ax_spec.imshow(self.img_spec)
        self.ax_spec.grid(True)
        self.ax_spec.set_xlabel(u'语音dhash')            #, fontsize=8

        # bottom plot 4
        self.ax_tmpt_spec = self.main_figure.figure.add_subplot(248)
        self.ax_tmpt_spec.set_ylim(-0.5, 15.5)
        self.ax_tmpt_spec.set_xlim(-0.5, 15.5)
        self.img_tmpt = np.random.random((16, 16))
        self.ax_tmpt_spec.imshow(self.img_tmpt)
        self.ax_tmpt_spec.grid(True)
        self.ax_tmpt_spec.set_xlabel(u'模板dhash')  # , fontsize=8


        # line objects
        self.line_top, = self.ax_top.plot(self.time_vect,
                                          np.ones_like(self.time_vect))

        # self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
        #                                         np.ones_like(self.freq_vect))

        self.line_voice, = self.ax_voice.plot(self.time_vect,
                                              np.ones_like(self.time_vect))

    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """
        # gets the latest frames
        frames = self.mic.get_frames()

        if len(frames) > 0:
            if len(frames) >= 2:
                # print("Discard sound frames: ", len(frames) - 1)
                self.discard_cnt = self.discard_cnt + len(frames) - 1
                self.label_discard_cnt.setText(str(self.discard_cnt))
                if self.record_voice_for_tmpt == "YES":
                    self.word_record_data = []
                    print("Words voice template record failed! ")
                self.main_figure.canvas.draw()
                return
            # keeps only the last frame
            current_frame = frames[-1]
            # print("time: %.3f" % (time.time() - self.get_data_time_stamp))
            self.get_data_time_stamp = time.time()

            if self.record_voice_for_tmpt == "YES":
                self.word_record_data = np.hstack((self.word_record_data, current_frame))
                if len(self.word_record_data) >= self.voi_ana_01.Fs*20:
                    print("Words voice template record time max is 20 second. ")
                    self.record_voice_for_tmpt = "RecordOK"
            if self.record_voice_for_tmpt == "RecordOK":
                print("Begin create words info template...")

                word_list = self.edit_cmd.text()
                t_srt = word_list.replace(" ", "")
                self.word_tmpt_list = t_srt.strip(',').split(',')
                self.word_cmd_N = len(self.word_tmpt_list)

                cre_voi_tmpt.create_words_template_fromwav(self.word_record_data, self.word_tmpt_list)
                self.tmpt_voice_pars = np.load(self.tmpt_file)
                self.record_voice_for_tmpt = "NO"
                self.btn_create_tmpt.setText("命令词\n录音")

            if len(self.disp_data) <= 100:
                self.disp_data = current_frame.copy()
            else:
                if len(self.disp_data) <= 32000:
                    self.disp_data = np.hstack((self.disp_data, current_frame))
                else:
                    self.disp_data = self.disp_data[-28000:-1]
                    self.disp_data = np.hstack((self.disp_data, current_frame))

            self.time_vect = np.arange(len(self.disp_data)//20, dtype=np.float32) / self.mic.rate * 1000 * 20
            _len_data = len(self.disp_data)
            _disp_data = self.disp_data[0:_len_data:20]
            _len_disp = min(len(self.time_vect), len(_disp_data))

            self.line_top.set_data(self.time_vect[0:_len_disp], _disp_data[0:_len_disp])

            if self.pause == 'pause':
                self.main_figure.canvas.draw()
                return

            if self.record_voice_for_tmpt != "NO":
                self.main_figure.canvas.draw()
                return

            if time.time() - self.start_time_stamp >= 1.5:
                if self.voi_ana_01.noise_info.noise_state == 'noise_noinit':
                    self.voi_ana_01.noise_info.get_noise_info(current_frame)

                self.voi_ana_01.get_voice_word_data(current_frame, time.time()-self.start_time_stamp)

                if self.voi_ana_01.ana_stage == "voi_ana_get_data":

                    _data = np.array(self.voi_ana_01.data)
                    if len(_data) >= self.voi_ana_01.min_word_len:

                        self.voi_ana_01.cal_spec_info()

                        cmd_N = self.word_cmd_N

                        if cmd_N > 19:
                            cmd_N = 19
                        simi_rate = np.ones((cmd_N,))
                        time_rate = np.ones((cmd_N,))
                        for i in range(cmd_N):
                            _wi = self.voi_ana_01.spec_dhash_width
                            _wi_tmpt = self.tmpt_voice_pars['arr_0'][i]['_zoom_width']
                            if abs(_wi - _wi_tmpt) > 4:
                                # print("_wi:", _wi, "_wtmpt", _wi_tmpt)
                                continue
                            simi_rate[i] = voice_para_save.cal_dhash_diff_rate(self.voi_ana_01.spec_dhash[:, 0:_wi],
                                               self.tmpt_voice_pars['arr_0'][i]['spec_dhash'][:, 0:_wi_tmpt])

                            _tdhash = voice_para_save.cal_dhash(self.voi_ana_01.fft_zoom_out[:, 0:_wi].T)
                            _tdhash = voice_para_save.std_dhash(_tdhash, 0.06)
                            _v, _h = _tdhash.shape
                            if _v > 16:
                                _v = 16
                            if _h > 16:
                                _h = 16
                            _tdhash = _tdhash[0:_v, 0:_h]
                            _tdhash = _tdhash.T

                            _tmpthash = voice_para_save.cal_dhash(self.tmpt_voice_pars['arr_0'][i]['spec_zoom_out'][:, 0:_wi_tmpt].T)
                            _tmpthash = voice_para_save.std_dhash(_tmpthash, 0.06)
                            _v, _h = _tmpthash.shape
                            if _v > 16:
                                _v = 16
                            if _h > 16:
                                _h = 16
                            _tmpthash = _tmpthash[0:_v, 0:_h]
                            _tmpthash = _tmpthash.T

                            time_rate[i] = voice_para_save.cal_dhash_diff_rate(_tdhash, _tmpthash, threshold=0.8)

                        # print("-----" * 10)
                        # print(simi_rate)
                        # # print("-----" * 5)
                        # print(time_rate)

                        self.disp_simirate.setText(", ".join('%4.2f'%F for F in (simi_rate)))

                        _x_time = np.arange(len(self.voi_ana_01.data)) / self.voi_ana_01.Fs
                        _x_time = _x_time + self.voi_ana_01.begin_time
                        self.line_voice.set_data(_x_time, np.array(self.voi_ana_01.data))
                        self.ax_voice.set_xlim(_x_time[0], _x_time[-1])
                        self.ax_voice.set_ylim(-4000, 4000)

                        _word_idx = np.argmin(simi_rate)
                        _abs_idx = np.argmin(time_rate)

                        # if _abs_idx != _word_idx:
                            # print("Not equal: ", _word_idx, _abs_idx)

                        if (((simi_rate[_word_idx]) <= voice_para_save.g_Word_Matched_Rate) and
                            (time_rate[_word_idx] <= 0.8)):
                            self.ax_tmpt_spec.imshow(self.tmpt_voice_pars['arr_0'][_word_idx]['spec_dhash'],
                                                     origin='lower')
                            self.label_word.setText(self.tmpt_voice_pars['arr_0'][_word_idx]['word'])

                            self.word_cnt += 1
                            self.label_cnt.setText(str(self.word_cnt))

                            self.ax_spec.imshow(self.voi_ana_01.spec_dhash, origin='lower')
                            self.ax_bottom.imshow(self.voi_ana_01.word_fft, origin='lower')


                        else:
                            self.err_cnt = self.err_cnt + 1
                            self.label_err_cnt.setText(str(self.err_cnt))

                    self.voi_ana_01.data = []
                    self.voi_ana_01.ana_stage = 'voi_ana_finished'

            # refreshes the plots
            self.main_figure.canvas.draw()


import sys

app = QtWidgets.QApplication(sys.argv)
window = LiveFFTWidget()

# app.show()

sys.exit(app.exec_())
