import torch

from utils.func import print_val_info
from arch.spectral_downsample import Spectral_downsample
from arch.spatial_downsample import Spatial_downsample

from arch.augmentor import *

# 设置固定的输入值
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2)     #seed is set to 2

class DownModel():
    def __init__(self,train_dataset):
        # config
        self.train_dataset = train_dataset
        self.sp_matrix = self.train_dataset.sp_matrix
        self.sp_range = self.train_dataset.sp_range
        self.psf = self.train_dataset.PSF
        self.hsi_channels = self.train_dataset.hsi_channels
        self.msi_channels = self.train_dataset.msi_channels
        self.lr = args.lr_stage1
        self.L1Loss =  nn.L1Loss(reduction='mean')
        self.MseLoss = nn.MSELoss(reduction='mean')

        # define train_data
        self.origin_lhsi = self.train_dataset[0]["lhsi"].unsqueeze(0).float().to(args.device)  # change 3-order to 4-order(add batch)，i.e., from C,H,W to B,C,H,W (Meet the input requirements of pytorch)
        self.origin_hmsi = self.train_dataset[0]['hmsi'].unsqueeze(0).to(args.device)
        self.origin_hhsi = self.train_dataset[0]['hhsi'].unsqueeze(0).to(args.device)
        self.origin_lrmsi_frommsi = self.train_dataset[0]['lrmsi_frommsi'].unsqueeze(0).to(args.device)
        self.origin_lrmsi_fromlrhsi = self.train_dataset[0]['lrmsi_fromlrhsi'].unsqueeze(0).to(args.device)

        self.hhsi_true = self.origin_hhsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_lrhsi_true = self.origin_lhsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_msi_true = self.origin_hmsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_frommsi_true = self.origin_lrmsi_frommsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_fromlrhsi_true = self.origin_lrmsi_fromlrhsi.detach().cpu().numpy().transpose(0, 2, 3, 1)

        # define down network
        self.Spectral_down_net = Spectral_downsample(args, self.hsi_channels, self.msi_channels, self.sp_matrix, self.sp_range,init_type='Gaussian', init_gain=0.02, initializer=True)
        self.Spatial_down_net = Spatial_downsample(args, self.psf, init_type='mean_space', init_gain=0.02, initializer=True)

        self.augmentor = Augmentor(args.hsi_channel).to(args.device)

        self.optimizer_Spectral_down = torch.optim.Adam(self.Spectral_down_net.parameters(), lr=args.lr_stage1)
        self.optimizer_Spatial_down = torch.optim.Adam(self.Spatial_down_net.parameters(), lr=args.lr_stage1)

        self.optimizer_augmentor = torch.optim.Adam(
            self.augmentor.parameters(),
            lr=args.learning_rate_a,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=args.decay_rate
        )

        self.lr_a = args.learning_rate_a

    def __call__(self):
        for epoch in range(1, args.epoch_stage1 + 1):
            j = 0
            # 开始
            self.augmentor = self.augmentor.train(mode=True)

            # 增强操作
            self.optimizer_augmentor.zero_grad()
            angle_lhsi = self.augmentor(self.origin_lhsi)
            aug_lhsi = rot_img(self.origin_lhsi,angle_lhsi,dtype=torch.cuda.FloatTensor)
            # angle_hmsi = angle_lhsi.clone()
            aug_hmsi = rot_img(self.origin_hmsi,angle_lhsi,dtype=torch.cuda.FloatTensor)


            # 对增强图像进行下采样
            out_lrmsi_fromaug_lrhsi=self.Spectral_down_net(aug_lhsi)#spectrally degraded from lrhsi
            out_lrmsi_fromaug_hmsi=self.Spatial_down_net(aug_hmsi)     #spatially degraded from hrmsi

            # 对真实图像进行旋转
            aug_out_lrmsi_fromlrhsi_true = rot_img(self.origin_lrmsi_fromlrhsi,angle_lhsi,dtype=torch.cuda.FloatTensor).detach().clone()
            aug_out_lrmsi_frommsi_true = rot_img(self.origin_lrmsi_frommsi,angle_lhsi,dtype=torch.cuda.FloatTensor).detach().clone()

            loss_aug_lhsi = torch.log(self.L1Loss(out_lrmsi_fromaug_lrhsi,aug_out_lrmsi_fromlrhsi_true)) * 2

            loss_aug_hmsi = torch.log(self.L1Loss(out_lrmsi_fromaug_hmsi , aug_out_lrmsi_frommsi_true)) * 2

            loss_all = loss_aug_lhsi + loss_aug_hmsi
            loss_all.backward(retain_graph=True)

            self.optimizer_augmentor.step()

            self.optimizer_Spatial_down.zero_grad()
            self.optimizer_Spectral_down.zero_grad()

            out_lrmsi_fromori_lrhsi = self.Spectral_down_net(self.origin_lhsi)
            out_lrmsi_fromori_hmsi = self.Spatial_down_net(self.origin_hmsi)
            new_out_lrmsi_fromaug_lrhsi = self.Spectral_down_net(aug_lhsi.detach().clone())  # spectrally degraded from lrhsi
            new_out_lrmsi_fromaug_hmsi = self.Spatial_down_net(aug_hmsi.detach().clone())  # spatially degraded from hrmsio


            d_loss = self.L1Loss(out_lrmsi_fromori_lrhsi, self.origin_lrmsi_fromlrhsi) + self.L1Loss(out_lrmsi_fromori_hmsi, self.origin_lrmsi_frommsi) + \
                    self.L1Loss(new_out_lrmsi_fromaug_lrhsi,aug_out_lrmsi_fromlrhsi_true) + self.L1Loss(new_out_lrmsi_fromaug_hmsi, aug_out_lrmsi_frommsi_true)

            d_loss.backward(retain_graph=True)

            self.optimizer_Spatial_down.step()
            self.optimizer_Spectral_down.step()

            if epoch % 100 ==0:  #print traning results in the screen every 100 epochs
                    with torch.no_grad():

                        out_fromlrhsi0=out_lrmsi_fromori_lrhsi.detach().cpu().numpy()[j].transpose(1,2,0) #spectrally degraded from lrhsi
                        out_frommsi0  =out_lrmsi_fromori_hmsi.detach().cpu().numpy()[j].transpose(1,2,0) #spatially degraded from hrmsi

                        print('estimated PSF:',self.Spatial_down_net.psf.weight.data)
                        print('true PSF:',self.psf)
                        print('************')

                        train_message = 'two generated images, train epoch:{} lr:{}\ntrain:L1loss:{}, sam_loss:{},ergas:{}, psnr:{}, CC:{}, rmse:{}'
                        print_val_info(args,epoch,self.lr,train_message,out_fromlrhsi0,out_frommsi0)

                        test_message_SRF='SRF: generated lrmsifromlhsi and true lrmsifromlhsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC{}, rmse:{}'
                        print_val_info(args,epoch,self.lr,test_message_SRF,self.out_fromlrhsi_true[j],out_fromlrhsi0)

                        test_message_PSF='PSF: generated lrmsifrommsi and true lrmsifrommsi  epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC{}, rmse:{}'
                        print_val_info(args,epoch,self.lr,test_message_PSF,self.out_frommsi_true[j],out_frommsi0)

                        if (epoch > args.decay_begin_epoch_stage1 - 1):
                            each_decay = args.lr_stage1 / (args.epoch_stage1 - args.decay_begin_epoch_stage1 + 1)
                            self.lr = self.lr - each_decay
                            for param_group in self.optimizer_Spectral_down.param_groups:
                                param_group['lr'] = self.lr
                            for param_group in self.optimizer_Spatial_down.param_groups:
                                param_group['lr'] = self.lr
                        continue

            if (epoch>args.decay_begin_epoch_stage1-1):
                        each_decay=args.lr_stage1/(args.epoch_stage1-args.decay_begin_epoch_stage1+1)
                        each_decay_a=args.learning_rate_a/(args.epoch_stage1-args.decay_begin_epoch_stage1+1)
                        self.lr = self.lr-each_decay
                        self.lr_a = self.lr_a - each_decay_a
                        for param_group in self.optimizer_Spectral_down.param_groups:
                            param_group['lr'] = self.lr
                        for param_group in self.optimizer_Spatial_down.param_groups:
                            param_group['lr'] = self.lr
                        for param_group in self.optimizer_augmentor.param_groups:
                            param_group['lr'] = self.lr_a

class UpModel():
    def __init__(self,Spectral_up_net,down_model):
        # 构建数据
        self.origin_lhsi = down_model.origin_lhsi.clone().detach()   # change 3-order to 4-order(add batch)，i.e., from C,H,W to B,C,H,W (Meet the input requirements of pytorch)
        self.origin_hmsi = down_model.origin_hmsi.clone().detach()
        self.origin_hhsi = down_model.origin_hhsi.clone().detach()
        self.origin_lrmsi_frommsi = down_model.origin_lrmsi_frommsi.clone().detach()
        self.origin_lrmsi_fromlrhsi = down_model.origin_lrmsi_fromlrhsi.clone().detach()

        # reference 3-order H,W,C
        self.hhsi_true = self.origin_hhsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_lrhsi_true = self.origin_lhsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_msi_true = self.origin_hmsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_frommsi_true = self.origin_lrmsi_frommsi.detach().cpu().numpy().transpose(0, 2, 3, 1)
        self.out_fromlrhsi_true = self.origin_lrmsi_fromlrhsi.detach().cpu().numpy().transpose(0, 2, 3, 1)

        self.out_lrmsi_frommsi = down_model.Spatial_down_net(self.origin_hmsi)
        self.out_lrmsi_frommsi_new = self.out_lrmsi_frommsi.clone().detach()

        # 构建上采样网络
        self.Spectral_up_net = Spectral_up_net
        self.lr = args.lr_stage2
        self.optimizer_Spectral_up = torch.optim.Adam(self.Spectral_up_net.parameters(), lr=args.lr_stage2)
        self.L1Loss = nn.L1Loss(reduction='mean')
        self.Spatial_down_net = down_model.Spatial_down_net
        self.Spectral_down_net = down_model.Spectral_down_net
        self.optimizer_Spectral_down = down_model.optimizer_Spectral_down
        #for param_group in down_model.optimizer_Spectral_down.param_groups:
        #    param_group['lr'] = self.lr

    def __call__(self):
        # setup_seed(2)  # seed is set to 2
        #for param_group in self.optimizer_Spectral_down.param_groups:
        #    param_group['lr'] = self.lr
        for epoch in range(1, args.epoch_stage2 + 1):
            k = 0
            self.optimizer_Spectral_up.zero_grad()
            # self.optimizer_Spectral_down.zero_grad()


            lrhsi = self.Spectral_up_net(self.out_lrmsi_frommsi_new)  # learn SpeUnet, the spectral inverse mapping from low MSI to low HSI
            hrhsi = self.Spectral_up_net(self.origin_hmsi)
            pre_hrmsi = self.Spectral_down_net(hrhsi)
            
            loss3 = self.L1Loss(pre_hrmsi,self.origin_hmsi)
            loss2 = self.L1Loss(lrhsi, self.origin_lhsi)
            loss_all = loss2 + loss3 * 0.3
            loss_all.backward()

            self.optimizer_Spectral_up.step()

            if epoch % 100 == 0:  # print traning results in the screen every 100 epochs

                with torch.no_grad():
                    lrhsi = self.Spectral_up_net(self.out_lrmsi_frommsi_new)

                    out_lrhsi = lrhsi.detach().cpu().numpy()[k].transpose(1, 2, 0)

                    est_hhsi = self.Spectral_up_net(self.origin_hmsi).detach().cpu().numpy()[k].transpose(1, 2, 0)  # use the learned SpeUnet to generate estimated HHSI in the second stage

                    train_message_specUp = 'genrated lrhsi and true lrhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC{}, rmse:{}'
                    print_val_info(args,epoch,self.lr,train_message_specUp,self.out_lrhsi_true[k],out_lrhsi)

                    test_message_specUp = 'generated hhsi and true hhsi epoch:{} lr:{}\ntest:L1loss:{}, sam_loss:{}, psnr:{}, ergas:{}, CC{}, rmse:{}'
                    print_val_info(args,epoch,self.lr,test_message_specUp,self.hhsi_true[k],est_hhsi)

            if (epoch > args.decay_begin_epoch_stage2 - 1):
                each_decay = args.lr_stage2 / (args.epoch_stage2 - args.decay_begin_epoch_stage2 + 1)
                self.lr = self.lr - each_decay
                for param_group in self.optimizer_Spectral_up.param_groups:
                    param_group['lr'] = self.lr