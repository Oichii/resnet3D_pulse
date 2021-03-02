import os
import cv2
import glob
import json
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from pulse_sampler import PulseSampler
from pulse_dataset import PulseDataset

from scipy.stats import pearsonr
import heartpy as hp

from PhysNet import NegPearson

from torchvision import utils

from utils import butter_bandpass_filter, psnr

resume = 'save_temp/transfer_3d_3.tar'

seq_len = 32
from ResNet_model import generate_model
model = generate_model(34)
model = torch.nn.DataParallel(model)

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
pulse_test = PulseDataset(sequence_list, root_dir, seq_len=seq_len,
                          length=len(sampler_test), transform=transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(pulse_test, batch_size=1, shuffle=False, sampler=sampler_test, pin_memory=True)

model.eval()
criterion = NegPearson()

criterion = criterion.cuda()

outputs = []
reference_ = []
for i, (net_input, target) in enumerate(val_loader):
    net_input = net_input.cuda(non_blocking=True)
    # target = target.squeeze()
    # print(target)
    target = target.cuda(non_blocking=True)

    # compute output
    # with torch.no_grad():

    output = model(net_input)
    print(output.size(), output[:, 0])

    output[:, 0].backward()
    gradient = model.module.get_activations_gradient()
    print('grad', gradient.size())
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3])
    print(pooled_gradients.shape)
    activations = model.module.get_activations(net_input).detach()

    # weight the channels by corresponding gradients
    for i in range(60):
        activations[:, i, :, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    print(heatmap.size(), heatmap.size()[0])
    hs = 32
    print('start plot ')
    fig, axs = plt.subplots(2, 4, figsize=(12, 12), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.00001, wspace=.01)

    axs = axs.ravel()

    for j in range(0, 8):
        heatmap = heatmap[j, :]
        print(heatmap)
        print(activations.size(), heatmap.size())
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf

        heatmap = np.maximum(heatmap.cpu(),  0)
        print(heatmap)
        # normalize the heatmap
        print(j)
        heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))
        # plt.matshow(heatmap.squeeze())
        # plt.show()


        # draw the heatmap
        # plt.subplot(hs//2, hs//2, i+1)
        print(heatmap)

        axs[j].matshow(heatmap)
        axs[j].axis('off')
        axs[j].title.set_text('Filtr {}'.format(j+1))

        heatmap = torch.mean(activations, dim=1).squeeze()
    plt.tight_layout()
    plt.show()

    heatmap = heatmap[6, :]
    # relu on top of the heatmap
    heatmap = np.maximum(heatmap.cpu(), 0)
    # normalize the heatmap
    heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))
    print(output, net_input.shape)
    # img = net_input[:,1,:,:].squeeze().cpu()
    net_input = (net_input-torch.min(net_input))/(torch.max(net_input)-torch.min(net_input))
    img = np.stack([net_input[:, 0, 0, :, :].squeeze().cpu(),
                    net_input[:, 1, 0, :, :].squeeze().cpu(),
                    net_input[:, 2, 0, :, :].squeeze().cpu()], axis=2)
    img = (np.array(img)*255).astype('uint8')
    plt.imshow(img)
    plt.show()

    heatmap = heatmap.squeeze().cpu()
    print('shape:', heatmap.shape)
    h = (np.array(heatmap)*255).astype('uint8')
    print('\n\n', h)
    print(np.max(h), np.mean(h))

    heatmap = cv2.resize(h, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cv2.imshow('h', heatmap)
    plt.show()
    superimposed_img = heatmap * 0.6 + img
    superimposed_img = ((superimposed_img-np.min(superimposed_img))/(np.max(superimposed_img)-np.min(superimposed_img))*255).astype('uint8')
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()
    plt.savefig('conv4-resnet.eps', bbox_inches='tight')
    outputs.append(output.squeeze())
    reference_.append(target)


def visTensor(tensor, ch=0, ft=0, allkernels=False, nrow=8, padding=1):
    print(tensor.shape)
    n, c, t, w, h = tensor.shape

    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, ft, :, :].unsqueeze(dim=1)
        print(tensor.shape)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))


if __name__ == "__main__":
    resume = 'save_temp/checkpoint_9.tar'
    print("initialize model {} ...".format(13))

    seq_len = 32

    model = generate_model(34)
    model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict['state_dict'])
    # model.cuda()

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
    print('hello')
    layer = 1
    net_filter = model.module.ConvBlock9[0].weight.data.clone()
    visTensor(net_filter, ch=0, ft=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    # plt.show()
    visTensor(net_filter, ch=0, ft=1, allkernels=False)

    plt.axis('off')
    plt.ioff()
    # plt.show()
    visTensor(net_filter, ch=0, ft=2, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()
