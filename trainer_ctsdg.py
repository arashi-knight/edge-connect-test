
import os
import time
from time import sleep

import numpy as np
import torch
from colorama import Fore
from lpips import lpips
from torch import nn, optim
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import torch.nn.functional as F

from main import load_config
from src.edge_connect import EdgeConnect
from support import myUtils
# from options.train_options import TrainOptions

from support.config import Config
from support.data import comic_dataloader
from support.data.dataloader_init import get_dataloader
from support.edge_detector.model_torch import res_skip

from support.myUtils import set_device, get_optimizer, get_optimizer_D, ssim_by_list, psnr_by_list, make_val_grid



class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.loss = lpips.LPIPS(net='vgg').cuda()

    def forward(self, x, y):
        return self.loss(x, y).mean()

class Trainer_Our():
    def __init__(self, config: Config, is_test = False, debug = False):
        self.config = config
        # self.device = 'cuda' if config.is_cuda else 'cpu'
        self.epoch = 0
        # 迭代次数
        self.iteration = 0

        self.debug = debug

        self.val_img_save_path_compare = config.val_img_save_path_compare
        self.val_img_save_path_single = config.val_img_save_path_single
        self.best_val_img_save_path_compare = config.best_val_img_save_path_compare
        self.best_val_img_save_path_single = config.best_val_img_save_path_single
        self.val_from_train_img_save_path_compare = config.val_from_train_img_save_path_compare
        self.log_path = config.log_path
        self.debug_log_path = config.debug_log_path
        self.model_path = config.model_path

        self.val_test_img_save_path = os.path.join(config.result_path, 'val_test', 'val')
        self.val_test_from_train_img_save_path = os.path.join(config.result_path, 'val_test', 'val_from_train')

        self.best_psnr = 0
        self.best_ssim = 0
        self.data_range = config.data_range

        print('正在初始化数据集')
        self.classes, self.train_dataloader, self.test_dataloader, self.val_dataloader, self.val_from_train_dataloader = get_dataloader(config)
        print('数据集初始化完成')
        # self.classes_num = len(self.classes)

        # 损失函数
        self.lpips_loss = LPIPSLoss()

        self.avg_loss_hole = 0
        self.avg_loss_valid = 0
        self.avg_loss_l1 = 0
        self.avg_loss_adversarial = 0
        self.avg_loss_perceptual = 0
        self.avg_loss_style = 0
        # 创新点损失
        self.avg_loss_texture = 0
        self.avg_loss_mask_edge = 0
        self.avg_loss_structural = 0


        print('正在初始化生成模型')
        # 模型
        # self.g_model = set_device(ComicNet(in_channels=config.input_channels, out_channels=config.output_channels,
        #                                    context_channels=config.context_channels, base_block_num=config.base_block_num))
        opt = load_config(1)
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner

        self.model = EdgeConnect(opt)
        self.model.load()



        # 初始化边缘提取模型
        self.edge_model = set_device(self.get_edge_model())
        print('生成模型初始化完成')

        # 设置学习率衰减
        # self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, step_size=config.step_size, gamma=config.gamma)

        # 如果保存路径不存在则创建
        if not os.path.exists(self.val_img_save_path_compare):
            os.makedirs(self.val_img_save_path_compare)
        if not os.path.exists(self.val_img_save_path_single):
            os.makedirs(self.val_img_save_path_single)
        if not os.path.exists(self.best_val_img_save_path_compare):
            os.makedirs(self.best_val_img_save_path_compare)
        if not os.path.exists(self.best_val_img_save_path_single):
            os.makedirs(self.best_val_img_save_path_single)
        if not os.path.exists(self.val_from_train_img_save_path_compare):
            os.makedirs(self.val_from_train_img_save_path_compare)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.val_test_img_save_path):
            os.makedirs(self.val_test_img_save_path)
        if not os.path.exists(self.val_test_from_train_img_save_path):
            os.makedirs(self.val_test_from_train_img_save_path)

        # 加载模型
        self.load_model()


    # 保存模型
    def save_model(self):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model.pth')
        d_model_path = os.path.join(self.model_path, 'd_model.pth')
        torch.save(self.generator.state_dict(), g_model_path)
        torch.save(self.discriminator.state_dict(), d_model_path)

    # 保存第x个epoch的模型
    def save_model_epoch(self, epoch):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_{}.pth'.format(epoch))
        d_model_path = os.path.join(self.model_path, 'd_model_{}.pth'.format(epoch))
        torch.save(self.generator.state_dict(), g_model_path)
        torch.save(self.discriminator.state_dict(), d_model_path)
    # 保存最后一个epoch的模型
    def save_model_last(self):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_last.pth')
        d_model_path = os.path.join(self.model_path, 'd_model_last.pth')
        torch.save(self.generator.state_dict(), g_model_path)
        torch.save(self.discriminator.state_dict(), d_model_path)

    def load_model_epoch(self, epoch):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_{}.pth'.format(epoch))
        d_model_path = os.path.join(self.model_path, 'd_model_{}.pth'.format(epoch))
        self.generator.load_state_dict(torch.load(g_model_path))
        self.discriminator.load_state_dict(torch.load(d_model_path))

    def load_model_last(self):
        # 保存路径是否存在
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        g_model_path = os.path.join(self.model_path, 'g_model_last.pth')
        d_model_path = os.path.join(self.model_path, 'd_model_last.pth')
        self.generator.load_state_dict(torch.load(g_model_path))
        self.discriminator.load_state_dict(torch.load(d_model_path))


    # 一轮训练
    def train_epoch(self, mode = 1):

        self.model.edge_model.train()
        self.model.inpaint_model.train()

        # 平均l1损失
        self.avg_loss_l1 = 0
        batch_count = 0

        # 清空debug_log
        myUtils.clear_log(self.debug_log_path)
        # 当前epoch
        myUtils.write_log('epoch:{}'.format(self.epoch), self.debug_log_path, print_log=False)
        # 每个epoch的进度条
        with tqdm(total=len(self.train_dataloader),
                  bar_format=Fore.BLACK + '|{bar:30}|正在进行训练|当前batch为:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|剩余时间:{remaining}|{desc}') as train_pbar:
            for i, (imgs, structures, masks, labels, tags) in enumerate(self.train_dataloader):
                if self.debug:
                    myUtils.write_log('batch:{}'.format(i), self.debug_log_path, bar=train_pbar, print_log=False)
                # 设置cuda
                imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])

                input_image = imgs * masks
                input_structure = structures * masks

                images_gray = imgs

                if mode == 1:
                    outputs, gen_loss, dis_loss, logs = self.model.edge_model.process(images_gray, structures, masks)
                    precision, recall = self.model.edgeacc(structures * masks, outputs * masks)
                    logs.append(('precision', precision.item()))
                    logs.append(('recall', recall.item()))
                    # backward
                    self.model.edge_model.backward(gen_loss, dis_loss)

                # inpaint model
                elif mode == 2:
                    # train
                    outputs, gen_loss, dis_loss, logs = self.model.inpaint_model.process(imgs, structures, masks)
                    outputs_merged = (outputs * masks) + (imgs * (1 - masks))

                    # metrics
                    psnr = self.model.psnr(self.model.postprocess(imgs), self.model.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(imgs)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.model.inpaint_model.backward(gen_loss, dis_loss)

                elif mode == 3:
                    # train
                    if True or np.random.binomial(1, 0.5) > 0:
                        outputs = self.edge_model(images_gray, structures, masks)
                        outputs = outputs * masks + structures * (1 - masks)
                    else:
                        outputs = structures

                    outputs, gen_loss, dis_loss, logs = self.model.inpaint_model.process(imgs, outputs.detach(), masks)
                    outputs_merged = (outputs * masks) + (imgs * (1 - masks))

                    # metrics
                    psnr = self.model.psnr(self.model.postprocess(imgs), self.model.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(imgs)).float()
                    logs.append(('psnr', psnr.item()))
                    logs.append(('mae', mae.item()))

                    # backward
                    self.model.inpaint_model.backward(gen_loss, dis_loss)

                else:
                    e_outputs, e_gen_loss, e_dis_loss, e_logs = self.edge_model.process(images_gray, structures, masks)
                    e_outputs = e_outputs * masks + structures * (1 - masks)
                    i_outputs, i_gen_loss, i_dis_loss, i_logs = self.model.inpaint_model.process(imgs, e_outputs, masks)
                    outputs_merged = (i_outputs * masks) + (imgs * (1 - masks))

                    # metrics
                    psnr = self.model.psnr(self.model.postprocess(imgs), self.model.postprocess(outputs_merged))
                    mae = (torch.sum(torch.abs(imgs - outputs_merged)) / torch.sum(imgs)).float()
                    precision, recall = self.model.edgeacc(structures * masks, e_outputs * masks)
                    e_logs.append(('pre', precision.item()))
                    e_logs.append(('rec', recall.item()))
                    i_logs.append(('psnr', psnr.item()))
                    i_logs.append(('mae', mae.item()))
                    logs = e_logs + i_logs

                    # backward
                    self.model.inpaint_model.backward(i_gen_loss, i_dis_loss)
                    self.model.edge_model.backward(e_gen_loss, e_dis_loss)

                batch_count += 1
                train_pbar.update(1)

            self.avg_loss_l1 = self.avg_loss_l1 / batch_count


    # 训练
    def train(self):
        with tqdm(total=self.config.epochs,
                  bar_format=Fore.MAGENTA + '|{bar:30}|当前epoch:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|{desc}') as epoch_pbar:

            psnr_list = []
            ssim_list = []
            lpips_list = []
            train_l1_list = []
            test_l1_list = []


            # self.generator.train()
            start_time = time.time()

            for epoch in range(self.epoch, 20):
                epoch_pbar.update(1)
                # 进行一轮训练
                self.train_epoch(mode=1)
                epoch_pbar.display()

            for epoch in range(self.epoch, 20):
                epoch_pbar.update(1)
                # 进行一轮训练
                self.train_epoch(mode=2)
                epoch_pbar.display()

            for epoch in range(self.epoch, 30):
                epoch_pbar.update(1)
                # 进行一轮训练
                self.train_epoch(mode=3)

                # 记录l1损失
                train_l1_list.append(self.avg_loss_l1)
                # 保存train_l1_loss折线图
                myUtils.draw_by_list(train_l1_list, 'TRAIN_L1', save_path=self.config.train_l1_loss_img_save_path, show_min=True)


                epoch_pbar.display()
                # 是否进行测试
                if (epoch+1) % self.config.test_interval == 0:
                    psnr, ssim, l1, lpips = self.test()
                    # 保存psnr和ssim
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    lpips_list.append(lpips)
                    test_l1_list.append(l1)
                    # 打印psnr和ssim
                    myUtils.draw_by_list(psnr_list, 'PSNR', save_path=self.config.psnr_img_save_path, show_max=True)
                    myUtils.draw_by_list(ssim_list, 'SSIM', save_path=self.config.ssim_img_save_path, show_max=True)
                    myUtils.draw_by_list(lpips_list, 'LPIPS', save_path=self.config.lpips_img_save_path, show_min=True)
                    myUtils.draw_by_list(test_l1_list, 'TEST_L1', save_path=self.config.test_l1_loss_img_save_path, show_min=True)

                    # 判断psnr和ssim是否大于之前的最优值
                    if psnr > self.best_psnr and ssim > self.best_ssim:
                        myUtils.write_log('获得最优，当前epoch:{},psnr:{},ssim:{},l1:{},lpips:{}'.format(epoch, psnr, ssim, l1, lpips), self.log_path,
                                        bar=epoch_pbar)
                        self.best_psnr = psnr
                        self.best_ssim = ssim
                        # 保存模型
                        self.save_model()
                        # 保存验证图片
                        myUtils.write_log('正在保存验证图片', self.log_path, bar=epoch_pbar)
                        self.val(best=True)
                    else:
                        myUtils.write_log('未提高，当前epoch:{},psnr:{},ssim:{}，l1:{},lpips:{}'.format(epoch, psnr, ssim, l1, lpips), self.log_path,
                                        bar=epoch_pbar)
                        # 保存验证图片
                        myUtils.write_log('正在保存验证图片', self.log_path, bar=epoch_pbar)
                        self.val()



                # 保存模型
                if (epoch+1) % self.config.save_interval == 0:
                    self.save_model_epoch(epoch)
                # 更新进度条
                # 获取时间数据
                _, time_left, time_end = myUtils.get_time(start_time, epoch, self.config.epochs)
                epoch_pbar.set_description('预计剩余时间:{}|预计结束时间:{}'.format(time_left, time_end))

        self.save_model_last()

    # 测试
    def test(self):

        with tqdm(total=len(self.test_dataloader),
                  bar_format=Fore.BLUE + '|{bar:30}|正在进行测试|当前batch为:{n_fmt}/{total_fmt}|已运行时间:{elapsed}|剩余时间:{remaining}|{desc}') as test_pbar:
            avg_psnr = 0
            avg_ssim = 0
            avg_l1 = 0
            avg_lpips = 0
            batch_count = 0
            # 设置评估模式
            with torch.no_grad():
                self.model.inpaint_model.eval()
                self.model.edge_model.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.test_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                    images_gray = imgs
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])

                    input_image = imgs * masks
                    input_structure = structures * masks

                    edges = self.model.edge_model(images_gray, structures, masks).detach()
                    comp_structure = edges * masks + structures * (1 - masks)
                    outputs = self.model.inpaint_model(imgs, edges, masks)
                    comp_imgs = (outputs * masks) + (imgs * (1 - masks))


                    # 计算PSNR和SSIM
                    # 将图片转换为numpy
                    imgs_cpu = ((imgs - 0.5) / 0.5).detach().cpu().numpy()
                    comp_imgs_cpu = ((comp_imgs-0.5)/0.5).detach().cpu().numpy()

                    # 计算PSNR和SSIM
                    batch_psnr = psnr_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_ssim = ssim_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_l1 = F.l1_loss(imgs, comp_imgs).item()
                    batch_lpips = self.lpips_loss((imgs - 0.5) / 0.5, (comp_imgs-0.5)/0.5).item()


                    # 计算平均PSNR和SSIM
                    avg_psnr += batch_psnr
                    avg_ssim += batch_ssim
                    avg_l1 += batch_l1
                    avg_lpips += batch_lpips
                    batch_count += 1

                    val_grid = self.get_grid(imgs, structures, masks, comp_imgs, comp_structure)

                    compare_img_path = os.path.join(self.val_img_save_path_compare, f"{i:0>3d}" + '.jpg')
                    single_img_path = os.path.join(self.val_img_save_path_single, f"{i:0>3d}" + '.jpg')

                    # 保存图片
                    save_image(val_grid, compare_img_path)
                    save_image(make_grid(comp_imgs, nrow=1, normalize=True, scale_each=True), single_img_path)



                    test_pbar.update(1)

        if batch_count == 0:
            batch_count = 1
        avg_psnr = avg_psnr / batch_count
        avg_ssim = avg_ssim / batch_count
        avg_l1 = avg_l1 / batch_count
        avg_lpips = avg_lpips / batch_count
        # print('avg_psnr:{}, avg_ssim:{}, avg_l1:{}'.format(avg_psnr, avg_ssim, avg_l1))
        return avg_psnr, avg_ssim, avg_l1, avg_lpips

    # 验证（保存验证图像）
    def val(self, best = False):
        # print('val')
        # with tqdm(total=len(self.val_dataloader),
        #           bar_format=Fore.GREEN + '|{bar:30}|正在进行验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
        #     with torch.no_grad():
        #         # 设置评估模式
        #         self.generator.eval()
        #         for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_dataloader):
        #             # 设置cuda
        #             imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
        #
        #             input_image = imgs * masks
        #             input_structure = structures * masks
        #
        #             # ---------
        #             # Generator
        #             # ---------
        #
        #             output, projected_image, projected_edge = \
        #                 self.generator(input_image, torch.cat((input_structure, input_image), dim=1), masks)
        #             comp_imgs = imgs * masks + output * (1 - masks)
        #
        #             # 横向拼接图片（原图，mask，原图叠mask图，生成图）
        #             val_grid = self.get_grid(imgs, structures, masks, comp_imgs)
        #             if best:
        #                 compare_img_path = os.path.join(self.best_val_img_save_path_compare, f"{i:0>3d}" + '.jpg')
        #                 single_img_path = os.path.join(self.best_val_img_save_path_single, f"{i:0>3d}" + '.jpg')
        #             else:
        #                 compare_img_path = os.path.join(self.val_img_save_path_compare, f"{i:0>3d}" + '.jpg')
        #                 single_img_path = os.path.join(self.val_img_save_path_single, f"{i:0>3d}" + '.jpg')
        #
        #             # 保存图片
        #             save_image(val_grid, compare_img_path)
        #             save_image(make_grid(comp_imgs, nrow=1, normalize=True, scale_each=True), single_img_path)
        #
        #             val_pbar.update(1)

        with tqdm(total=len(self.val_from_train_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行测试集验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            with torch.no_grad():
                # 设置评估模式
                self.model.edge_model.eval()
                self.model.inpaint_model.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_from_train_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])
                    images_gray = imgs
                    input_image = imgs * masks
                    input_structure = structures * masks


                    edges = self.model.edge_model(images_gray, structures, masks).detach()
                    comp_structure = edges * masks + structures * (1 - masks)
                    outputs = self.model.inpaint_model(imgs, edges, masks)
                    comp_imgs = (outputs * masks) + (imgs * (1 - masks))


                    # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                    val_grid = self.get_grid(imgs, structures, masks, comp_imgs, comp_structure)
                    compare_img_path = os.path.join(self.val_from_train_img_save_path_compare, f"{i:0>3d}" + '.jpg')
                    # 保存图片
                    save_image(val_grid, compare_img_path)

                    val_pbar.update(1)


    def val_test(self, best = False):

        avg_psnr = 0
        avg_ssim = 0
        avg_l1 = 0
        avg_lpips = 0
        batch_count = 0
        # print('val')
        with tqdm(total=len(self.val_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            with torch.no_grad():
                # 设置评估模式
                self.generator.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])

                    input_image = imgs * masks
                    input_structure = structures * masks

                    # ---------
                    # Generator
                    # ---------

                    output, projected_image, projected_edge = \
                        self.generator(input_image, torch.cat((input_structure, input_image), dim=1), masks)
                    comp_imgs = imgs * masks + output * (1 - masks)

                    # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                    val_grid = self.get_grid(imgs, structures, masks, comp_imgs)

                    compare_img_path = os.path.join(self.val_test_img_save_path, f"{i:0>3d}" + '.jpg')

                    # 保存图片
                    save_image(val_grid, compare_img_path)


                    # 計算psnr和ssim,lpips
                    # 将图片转换为numpy
                    imgs_cpu = imgs.detach().cpu().numpy()
                    comp_imgs_cpu = comp_imgs.detach().cpu().numpy()

                    # 计算PSNR和SSIM
                    batch_psnr = psnr_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_ssim = ssim_by_list(imgs_cpu, comp_imgs_cpu, data_range=self.data_range)
                    batch_l1 = F.l1_loss(imgs, comp_imgs).item()
                    batch_lpips = self.lpips_loss(imgs, comp_imgs).item()


                    # 计算平均PSNR和SSIM
                    avg_psnr += batch_psnr
                    avg_ssim += batch_ssim
                    avg_l1 += batch_l1
                    avg_lpips += batch_lpips
                    batch_count += 1

                    val_pbar.update(1)
        avg_psnr = avg_psnr / batch_count
        avg_ssim = avg_ssim / batch_count
        avg_l1 = avg_l1 / batch_count
        avg_lpips = avg_lpips / batch_count

        myUtils.write_log('val_test:psnr:{},ssim:{},l1:{},lpips:{}'.format(avg_psnr, avg_ssim, avg_l1, avg_lpips), self.log_path, print_log=True)


        with tqdm(total=len(self.val_from_train_dataloader),
                  bar_format=Fore.GREEN + '|{bar:30}|正在进行测试集验证|当前batch为:{n_fmt}/{total_fmt}|{desc}') as val_pbar:
            with torch.no_grad():
                # 设置评估模式
                self.generator.eval()
                for i, (imgs, structures, masks, labels, tags) in enumerate(self.val_from_train_dataloader):
                    # 设置cuda
                    imgs, structures, masks, labels = set_device([imgs, structures, masks, labels])

                    input_image = imgs * masks
                    input_structure = structures * masks

                    # ---------
                    # Generator
                    # ---------

                    output, projected_image, projected_edge = \
                        self.generator(input_image, torch.cat((input_structure, input_image), dim=1), masks)
                    comp_imgs = imgs * masks + output * (1 - masks)

                    # 横向拼接图片（原图，mask，原图叠mask图，生成图）
                    val_grid = self.get_grid(imgs, structures, masks, comp_imgs)
                    compare_img_path = os.path.join(self.val_test_from_train_img_save_path, f"{i:0>3d}" + '.jpg')
                    # 保存图片
                    save_image(val_grid, compare_img_path)

                    val_pbar.update(1)


    def load_model(self):
        print('load_model')


    def get_edge_model(self):
        """
        获取边缘检测
        :return: 模型
        """
        # 获取边缘检测
        edge_detect = res_skip()

        edge_detect.load_state_dict(torch.load(self.config.edge_model_path))

        myUtils.set_requires_grad(edge_detect, False)

        edge_detect.cuda()
        edge_detect.eval()

        return edge_detect

    def get_edge(self, img):
        """
        获取边缘
        :param img: 图片
        :return: 边缘
        """
        # 获取边缘
        with torch.no_grad():
            # 将-1到1的图片放缩到0-255
            img = (img + 1) * 127.5

            edge = self.edge_model(img)

            # 截取255-0
            edge = torch.clamp(edge, 0, 255)

            # 放缩到-1至1
            edge = (edge - 127.5) / 127.5

        return edge

    def get_grid(self, imgs, structures, masks, comp_imgs, comp_structures = None):
        if comp_structures is None:
            comp_imgs_structures = self.get_edge(comp_imgs)
        else:
            comp_imgs_structures = self.get_edge(comp_imgs)
            # 放缩
            comp_imgs_structures = (comp_imgs_structures - 0.5) / 0.5

        # 全部从0-1放缩到-1-1
        imgs = (imgs - 0.5) / 0.5
        structures = (structures - 0.5) / 0.5
        comp_imgs = (comp_imgs - 0.5) / 0.5



        # 都转成rgb格式
        imgs_rgb = myUtils.gray2rgb(imgs)
        structures_rgb = myUtils.gray2rgb(structures)
        masks_rgb = myUtils.gray2rgb(masks)
        # img_masked_rgb = myUtils.gray2rgb(img_masked)
        comp_imgs_rgb = myUtils.gray2rgb(comp_imgs)
        comp_imgs_structures_rgb = myUtils.gray2rgb(comp_imgs_structures, mode='RED')
        mask_red = myUtils.gray2rgb(masks, mode='RED')
        # 从【0,1】放缩到【-1,1】
        mask_red = (mask_red - 0.5) / 0.5

        # 在img的mask区域填充为红色
        img_masked_red = torch.where(masks.byte() == True, mask_red, imgs)  # 将 mask 区域的像素值设为红色 (1, 0, 0)

        # 拼接structures和comp_imgs_structures的mask区域
        # comp_imgs_structures_rgb_x = comp_imgs_structures_rgb * (1 - masks_rgb) + structures_rgb * masks_rgb
        comp_imgs_structures_rgb_x = comp_imgs_structures_rgb * masks_rgb + structures_rgb * (1 - masks_rgb)
        grid_list = [imgs_rgb, structures_rgb, masks_rgb, img_masked_red, comp_imgs_rgb, comp_imgs_structures_rgb_x]

        return myUtils.make_val_grid_list(grid_list)





if __name__ == '__main__':
    config = Config()
    trainer = Trainer_Our(config)
    # trainer.load_model_last()
    trainer.val()
    #
    # edge_detect = res_skip()
    #
    # edge_detect.load_state_dict(torch.load('../edge_detector/erika.pth'))


