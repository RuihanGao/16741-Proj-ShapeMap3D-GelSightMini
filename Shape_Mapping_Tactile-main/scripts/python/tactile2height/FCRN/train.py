import torch
import torch.utils.data
import torchvision
from loader import *
import os
from os import path as osp
from fcrn import FCRN
from torch.autograd import Variable
from weights import load_weights
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

dtype = torch.cuda.FloatTensor
weights_file = "NYU_ResNet-UpProj.npy"

def main():
    batch_size = 16
    data_path = 'nyu_depth_v2_labeled.mat'
    learning_rate = 1.0e-5
    monentum = 0.9
    weight_decay = 0.0005
    num_epochs = 100
    resume_from_file = True#False

    # 1.Load data
    save_path = osp.join('/home','zsi','shape_mapping','dataset')
    train_data_file = osp.join(save_path,'train_data.txt')
    train_label_file = osp.join(save_path,'train_label.txt')

    dev_data_file = osp.join(save_path,'dev_data.txt')
    dev_label_file = osp.join(save_path,'dev_label.txt')

    print("Loading data......")
    train_loader = torch.utils.data.DataLoader(GelDataLoader(train_data_file, train_label_file),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(GelDataLoader(dev_data_file, dev_label_file),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    print(train_loader)
    # 2.Load model
    print("Loading model......")
    model = FCRN(batch_size)
    # # load pre-trained weights
    # #resnet = torchvision.models.resnet50(pretrained=True)
    # resnet = torchvision.models.resnet50()
    # resnet.load_state_dict(torch.load('/home/pengfei/data/nets/ResNet/resnet50-19c8e357.pth'))
    # print("resnet50 loaded.")
    # resnet50_pretrained_dict = resnet.state_dict()
    #
    # model.load_state_dict(load_weights(model, weights_file, dtype))
    model = model.cuda()

    # 3.Loss
    loss_fn = torch.nn.MSELoss().cuda()
    print("loss_fn set.")

    # 5.Train
    best_val_err = 1.0e3

    # validate
    model.eval()
    num_correct, num_samples = 0, 0
    loss_local = 0
    with torch.no_grad():
        for input, depth in val_loader:
            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)

            input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
            pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

            input_gt_depth_image /= np.max(input_gt_depth_image)
            pred_depth_image /= np.max(pred_depth_image)

            plot.imsave('../outputs/train/input/input_rgb_epoch_0.png', input_rgb_image)
            plot.imsave('../outputs/train/gt/gt_depth_epoch_0.png', input_gt_depth_image, cmap="viridis")
            plot.imsave('../outputs/train/pred/pred_depth_epoch_0.png', pred_depth_image, cmap="viridis")

            # depth_var = depth_var[:, 0, :, :]
            # loss_fn_local = torch.nn.MSELoss()

            loss_local += loss_fn(output, depth_var)

            num_samples += 1

    err = float(loss_local) / num_samples
    print('val_error before train:', err)

    start_epoch = 0

    resume_file = 'checkpoint.pth.tar'
    if resume_from_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))

    for epoch in range(num_epochs):

        # 4.Optim
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=monentum, weight_decay=weight_decay)
        print("optimizer set.")

        print('Starting train epoch %d / %d' % (start_epoch + epoch + 1, num_epochs))
        model.train()
        running_loss = 0
        count = 0
        epoch_loss = 0

        #for i, (input, depth) in enumerate(train_loader):
        for input, depth in train_loader:
            # input, depth = data
            #input_var = input.cuda()
            #depth_var = depth.cuda()
            input_var = Variable(input.type(dtype))
            depth_var = Variable(depth.type(dtype))

            output = model(input_var)
            loss = loss_fn(output, depth_var)
            print('loss:', loss.item())
            count += 1
            running_loss += loss.data.cpu().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / count
        print('epoch loss:', epoch_loss)

        # validate
        model.eval()
        num_correct, num_samples = 0, 0
        loss_local = 0
        with torch.no_grad():
            for input, depth in val_loader:
                input_var = Variable(input.type(dtype))
                depth_var = Variable(depth.type(dtype))

                output = model(input_var)

                input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)
                pred_depth_image = output[0].data.squeeze().cpu().numpy().astype(np.float32)

                input_gt_depth_image /= np.max(input_gt_depth_image)
                pred_depth_image /= np.max(pred_depth_image)

                plot.imsave('../outputs/train/input/input_rgb_epoch_{}.png'.format(start_epoch + epoch + 1), input_rgb_image)
                plot.imsave('../outputs/train/gt/gt_depth_epoch_{}.png'.format(start_epoch + epoch + 1), input_gt_depth_image, cmap="viridis")
                plot.imsave('../outputs/train/pred/pred_depth_epoch_{}.png'.format(start_epoch + epoch + 1), pred_depth_image, cmap="viridis")

                # depth_var = depth_var[:, 0, :, :]
                # loss_fn_local = torch.nn.MSELoss()

                loss_local += loss_fn(output, depth_var)

                num_samples += 1

        err = float(loss_local) / num_samples
        print('val_error:', err)

        if err < best_val_err:
            best_val_err = err
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoint.pth.tar')

        if epoch % 10 == 0:
            learning_rate = learning_rate * 0.6


if __name__ == '__main__':
    main()
