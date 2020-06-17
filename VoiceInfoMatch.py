import numpy as np


def subtract_dhash(arr1, arr2):
    arr = arr1 * arr2
    _v, _w = arr.shape
    arr = np.where(arr < 0, 1, 0)
    _sum = np.sum(arr * 10) / (_v * _w)
    return _sum


def sub_abs_dhash(arr1, arr2):
    arr = arr1 - arr2
    _v, _w = arr.shape
    _sum = np.sum(abs(arr)) / (_v * _w)
    return _sum


def cal_dhash_diff_rate(arr1, arr2):
    _v1, _w1 = arr1.shape
    _v2, _w2 = arr2.shape
    if _v1 != _v2:
        print('vertical of cal_dhash_diff_rate(arr1, arr2) must be same')
        return []
    if _w1 == _w2:
        _out = sub_abs_dhash(arr1, arr2)
        return _out
    if _w1 > _w2:
        _d = _w1 - _w2
        t_val = np.zeros(_d + 1)
        for k in range(_d + 1):
            t_val[k] = sub_abs_dhash(arr1[:, k:k + _w2], arr2[:, 0:_w2])
        return np.min(t_val)
    else:
        _d = _w2 - _w1
        t_val = np.zeros(_d + 1)
        for k in range(_d + 1):
            t_val[k] = sub_abs_dhash(arr1[:, 0:_w1], arr2[:, k:k + _w1])
        return np.min(t_val)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=3)

    para_file = "./ResFiles/Wav/左转_右转x6_reduWordPara.npz"

    pars_pc = np.load(para_file)
    print(pars_pc.files)
    print(pars_pc["arr_0"].shape)

    simi_arr = np.zeros((12, 12))

    for i in range(12):
        for j in range(12):
            _wi = pars_pc['arr_0'][i]['_zoom_width']
            _wj = pars_pc['arr_0'][j]['_zoom_width']
            simi_arr[i, j] = cal_dhash_diff_rate(pars_pc['arr_0'][i]['spec_dhash'][:, 0:_wi],
                                                 pars_pc['arr_0'][j]['spec_dhash'][:, 0:_wj])

    print(simi_arr)

    t1 = simi_arr[0:6, 0:6]
    print(np.max(t1))
    t2 = simi_arr[6:, 0:6]
    print(np.min(t2))

    t3 = simi_arr[6:, 0:6]
    print(np.min(t3))
    t4 = simi_arr[6:, 6:]
    print(np.max(t4))

    plt.imshow(simi_arr)
    plt.show()




