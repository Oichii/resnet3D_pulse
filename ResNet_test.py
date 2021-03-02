import os
import glob
import json
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample, find_peaks
from pulse_sampler import PulseSampler
from pulse_dataset import PulseDataset
from PhysNet import NegPearson, NegPeaLoss
from ResNet_model import generate_model
from scipy.stats import pearsonr
import heartpy as hp
from utils import butter_bandpass_filter, psnr

import pandas as pd


for i in range(3, 30):
    resume = 'save_temp/transfer_3d_{}.tar'.format(i)
    print("initialize model {} ...".format(i))

    seq_len = 32

    model = generate_model(34)

    model = torch.nn.DataParallel(model)
    model.cuda()
    ss = sum(p.numel() for p in model.parameters())
    print('num params: ', ss)
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    sequence_list = "sequence_test.txt"
    root_dir = 'E:/Datasets_PULSE/set_all/'
    seq_list = []
    end_indexes_test = []
    with open(sequence_list, 'r') as seq_list_file:
        for line in seq_list_file:
            seq_list.append(line.rstrip('\n'))

    # seq_list = ['test_static']
    for s in seq_list:
        sequence_dir = os.path.join(root_dir, s)
        if sequence_dir[-2:len(sequence_dir)] == '_1':
            fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
            fr_list = fr_list[0:len(fr_list) // 2]
        elif sequence_dir[-2:len(sequence_dir)] == '_2':
            fr_list = glob.glob(sequence_dir[0:-2] + '/cropped/*.png')
            fr_list = fr_list[len(fr_list) // 2: len(fr_list)]
        else:
            if os.path.exists(sequence_dir + '/cropped/'):
                fr_list = glob.glob(sequence_dir + '/cropped/*.png')
            else:
                fr_list = glob.glob(sequence_dir + '/*.png')
        # print(fr_list)
        end_indexes_test.append(len(fr_list))

    end_indexes_test = [0, *end_indexes_test]
    # print(end_indexes_test)

    sampler_test = PulseSampler(end_indexes_test, seq_len, False)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    pulse_test = PulseDataset(sequence_list, root_dir, seq_len=seq_len,
                                            length=len(sampler_test), transform=transforms.Compose([
                                                                                                transforms.ToTensor(),
                                                                                                normalize]))
    val_loader = torch.utils.data.DataLoader(pulse_test, batch_size=1, shuffle=False, sampler=sampler_test, pin_memory=True)

    model.eval()
    criterion = NegPearson()
    criterion2 = nn.MSELoss()
    criterion3 = NegPeaLoss()

    criterion = criterion.cuda()

    outputs = []
    reference_ = []
    loss_avg = []
    loss_avg2 = []
    import time
    start = time.time()
    for k, (net_input, target) in enumerate(val_loader):
        net_input = net_input.cuda(non_blocking=True)
        # target = target.squeeze()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(net_input)

            outputs.append(output[0])
            reference_.append(target[0])

    end = time.time()
    print(end-start, len(val_loader))

    outputs = torch.cat(outputs)

    outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)
    # outputs = (outputs - torch.min(outputs)) / (torch.max(outputs) - torch.min(outputs))
    outputs = outputs.tolist()

    reference_ = torch.cat(reference_)
    # reference_ = (reference_ - torch.min(reference_)) / (torch.max(reference_) - torch.min(reference_))
    reference_ = (reference_-torch.mean(reference_))/torch.std(reference_)
    reference_ = reference_.tolist()

    fs = 30
    lowcut = 1
    highcut = 3
    import pandas as pd

    yr = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
    yr = (yr - np.mean(yr)) / np.std(yr)
    # plt.plot(yr, alpha=0.5, label='filtered')

    # restored=[]
    # for i in range(len(outputs)-1):
    #     restored.append(outputs[i]+outputs[i+1])
    # # print(outputs)
    # import pandas as pd
    # ppg = pd.read_csv(root_dir+'01-01_orginal' + '.txt', sep='\t')
    # ref_ppg = ppg.loc[:, 'waveform']
    # ref_ppg = resample(ref_ppg, len(fr_list))
    # ref_ppg = (ref_ppg - np.mean(ref_ppg)) /np.std(ref_ppg)
    # print(outputs)
    # print(reference_)
    # plt.subplot(121)

    plt.subplots_adjust(right=0.7)
    plt.plot(outputs, alpha=0.7, label='wyjście\n sieci')
    # plt.plot(yr, label='wyjście\n sieci po filtracji')
    plt.plot(reference_, '--', label='referencja\n PPG')

    # plt.plot(ref_ppg)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='large')
    # plt.legend(bbox_to_anchor=(0.5, -0.20), loc='upper center', borderaxespad=0, fontsize='large', ncol=3)
    # plt.title('model ze splotem trójwymiarowym'.format(i))
    # plt.plot(restored)
    plt.ylabel('Amplituda', fontsize='large', fontweight='semibold')
    plt.xlabel('Czas [próbka]', fontsize='large', fontweight='semibold')
    plt.grid()
    plt.xlim([350, 550])
    plt.ylim([-2, 3])

    plt.savefig('3d.svg', bbox_inches='tight')
    plt.show()
    reference_ = np.array(reference_)
    outputs = np.array(outputs)
    res = pd.DataFrame({'output': yr, 'ref': reference_})
    print(res)
    res.to_csv('splot_spa-temp.csv')

    # outputs = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))

    bpm_ref = []
    bpm_out = []
    bmmp_filt = []
    bpm_out2 = []
    hrv_ref = []
    hrv_out = []

    win = 255
    for i in range(win, len(reference_), win):
        peaks, _ = find_peaks(reference_[i:i+win], distance=20, height=0.9)
        peaks_out, _ = find_peaks(yr[i:i + win], height=0.95)
        # plt.plot(outputs[i:i+win])
        # plt.plot(yr[i:i + win])
        # plt.plot(peaks_out, outputs[i:i+win][peaks_out], "x")
        # plt.show()
        # print(len(peaks_out), len(peaks))

        _, measures2 = hp.process(reference_[i:i+win], 30.0)
        bpm_ref.append(30/(win/len(peaks))*win)
        bmmp_filt.append(measures2['bpm'])
        # print(measures2)
        hrv_ref.append(measures2['rmssd'])
        # print(measures2['bpm'], 30/(256/len(peaks))*60, 30/(256/len(peaks_out))*60)
        # print(measures2)
        # _, mm = hp.process(yr[i:i+256], 30.0)
        _, mmm = hp.process(yr[i:i + win], 30.0)
        # print(mm)
        bpm_out.append(mmm['bpm'])
        bpm_out2.append(30/(win/len(peaks_out))*win)
        hrv_out.append(mmm['rmssd'])

    plt.plot(bpm_out, label='output')
    plt.plot(bpm_ref, label='referencja')
    plt.plot(bmmp_filt, label='ref2')
    plt.plot(bpm_out2, label='out2')

    plt.legend()
    plt.show()

    corr, _ = pearsonr(bmmp_filt, bpm_out)
    c = np.corrcoef(bmmp_filt, bpm_out)
    cc = np.corrcoef(bpm_ref, bpm_out2)
    ccc = np.corrcoef(bmmp_filt, bpm_out2)
    print('korelacja pulsu:', c,  cc, ccc)

    # corr_ = np.correlate(reference_, outputs, 'full')
    # przes = np.argmax(corr_)
    # # cc = pearsonr(reference_[:len(outputs[przes:])], outputs[przes:])
    # # ccc = pearsonr(reference_, yr)
    # # print('korelacja sygnalow po przesunięciu:', cc, ccc)
    # plt.plot(reference_[:len(outputs[przes:])])
    # plt.plot(outputs[przes:])
    # plt.show()
    # corr2, _ = pearsonr(reference_[:len(outputs[przes:])], outputs[przes:])
    # cc = np.corrcoef(reference_[:len(outputs[przes:])], outputs[przes:])
    # print('korelacja sygnalow bez przesunięcia', corr2, cc)

    plt.subplots_adjust(right=0.7)
    time = np.arange(0, 3, 1 / fs)
    fourier_transform = np.fft.rfft(outputs)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, fs / 2, len(power_spectrum))
    plt.semilogy(frequency, power_spectrum, label='wyjście\n sieci')

    fourier_transform = np.fft.rfft(reference_)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    plt.xlim(-0.1, 10)
    plt.ylim(10e-6, 10e6)
    plt.semilogy(frequency, power_spectrum, label='referencja\n PPG')
    plt.ylabel('|A(f)|', fontsize='large', fontweight='semibold')
    plt.xlabel('Częstotliwość f [Hz]', fontsize='large', fontweight='semibold')
    plt.title('Częstitliwościowe widmo mocy')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    reference_ = torch.tensor(reference_)
    outputs = torch.tensor(outputs)
    # yr = torch.tensor(yr)
    pp = psnr(reference_, outputs)
    print('psnr', pp)

    criterionMSE = nn.MSELoss()
    criterionMAE = nn.L1Loss()
    mse = criterionMSE(reference_, outputs)
    rmse = torch.sqrt(mse)
    mae = criterionMAE(reference_, outputs)
    print(outputs.shape)
    se = torch.std(outputs-reference_)/np.sqrt(outputs.shape[0])
    print(mae, mse, rmse, se)
    print(hrv_out, hrv_ref)
    o = np.mean(hrv_out)
    r = np.mean(hrv_ref)
    err = abs(o-r)/r
    print(err)
    print()
    o = np.mean(bpm_out2)
    r = np.mean(bpm_ref)
    print(bpm_out2, bpm_ref)
    err = abs(o - r) / r
    print(err)
