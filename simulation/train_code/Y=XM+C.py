from __future__ import print_function

import os

import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from architecture import MST

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
# dtype = torch.cuda.DoubleTensor
# dtype = torch.double
device = torch.device('cuda')
torch.cuda.manual_seed(seed=666)


########################################### 辅助函数 ###########################################
def thres_21(L, tau, M, N, B):
    S = torch.sqrt(torch.sum(torch.mul(L, L), 2))
    S[S == 0] = 1
    T = 1 - tau / S
    T[T < 0] = 0
    R = T.reshape(M, N, 1).repeat((1, 1, B))
    res = torch.mul(R, L)
    return res


################################ 读取数据 ################################
Data = scio.loadmat("./SimuData.mat")
Clean, Mask, Cloud, Noisy = Data["Clean"], Data["Mask"], Data["Cloud"], Data["Noisy"]
# M, N, = 384, 384
M, N, = 256, 256
B, T = 4, 4
# 相邻四张图像为同一个时间，因而初始的维度为M, N, T, B
Clean = torch.from_numpy(Clean[:M, :N, :].reshape(M, N, T, B)).to(device)
Mask = torch.from_numpy(Mask[:M, :N, :].reshape(M, N, T, B)).to(device)  # 云=0
Cloud = torch.from_numpy(Cloud[:M, :N, :].reshape(M, N, T, B)).to(device)  # 云=1
Noisy = torch.from_numpy(Noisy[:M, :N, :].reshape(M, N, T, B)).to(device)

# 进行维度调整：M, N, B, T
Clean = torch.permute(Clean, [0, 1, 3, 2])
Mask = torch.permute(Mask, [0, 1, 3, 2])
Cloud = torch.permute(Cloud, [0, 1, 3, 2])
Noisy = torch.permute(Noisy, [0, 1, 3, 2])

# 将第二张观测图像设置为干净图像
Cloud[:, :, :, 1] = 0
Mask[:, :, :, 1] = 1
Noisy[:, :, :, 1] = Clean[:, :, :, 1]
G = Clean[:, :, :, 1].permute([2, 0, 1]).reshape(1, B, M, N).to(device)

################################ 网络输入 ################################
model = MST(dim=16).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
################################ 全局设置 ################################
eigen_num = 4


def get_args(args_list):
    count = 0
    for i in range(len(args_list[0])):
        for j in range(len(args_list[1])):
            for k in range(len(args_list[2])):
                for l in range(len(args_list[3])):
                    # if not (i == j and j == k):
                    yield args_list[0][i], args_list[1][j], args_list[2][k], args_list[3][l]
                count += 1


candidates2 = [
    # [1.000, 1.000, 0.001],  # best
    [1.000, 1.000, 0.001],  # best    candidate1
    # [1.000, 1.000, 0.010],    # condidate2
    # [1.000, 1.000, 0.010],    # condidate2
    # [0.001, 1.000, 0.001],
    # [0.010, 1.000, 0.010],
    # [0.100, 1.000, 0.001],
    # [0.010, 1.000, 0.100],      # condidate3
    # [0.010, 1.000, 0.001],
    # [0.001, 1.000, 0.010],
    # [1.000, 1.000, 0.100],
    # [0.010, 0.100, 0.001],
    # [0.001, 1.000, 0.100],
    # [0.100, 1.000, 0.100],
]

LR = 0.003  # try 0.01 0.001 0.0001
iter_num, epoch_num = 250, 200
order = 8
decrease = False
up = True
for lambda1, lambda2, rho in candidates2:
    rho_init = rho
    iters = 0
    order += 1
    # log_dir = "./logs6-[Y=XM+C lr outer(4 0.9) inner(75 0.6)]-candidate2/iter[%3d]-epoch[%3d]-lambda1[%.3f]-lambda2[%.3f]-rho[%.3f]-order[%03d]" \
    #             % (iter_num, epoch_num, lambda1, lambda2, rho, order)
    log_dir = "./logs7-[Y=XM+C]-candidate2/iter[%3d]-epoch[%3d]-lambda1[%.3f]-lambda2[%.3f]-rho[%.3f]-order[%03d]" \
              % (iter_num, epoch_num, lambda1, lambda2, rho, order)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # else:
    # print("log文件夹冲突！！！")
    # exit()
    # continue

    ################################ 初始化 ################################
    Y = Noisy.clone()
    X = Noisy.clone()
    W = Noisy.clone()
    # C = Cloud.clone()
    C = torch.zeros(Y.shape).type(dtype).to(device)

    psnr = 0
    ################################ 子问题更新 ################################
    while iters < iter_num:
        # if psnr > 34:
        #     break
        iters += 1
        ################################ 更新子问题X ################################

        # if decrease and (iters % 4 == 0) and LR > 0.00001:
        #     LR *= 0.9
        LR = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
        # scheduler = lr_scheduler.StepLR(optimizer, 75, 0.6,last_epoch=-1)
        # lr_change = lambda epoch: 50 / (50 + epoch)
        # scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lr_change)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [125], gamma=0.6, last_epoch=-1)
        loss_history, loss1_history, loss2_history = [], [], []
        epoch = 0
        while epoch < epoch_num:
            epoch += 1
            optimizer.zero_grad()
            # X: M, N, B, T
            Y_temp = torch.permute(Y, [2, 3, 0, 1]).reshape(1, B * T, M, N).type(dtype).to(device)
            Mask_temp = torch.permute(Mask, [2, 3, 1, 0]).reshape(1, B*T, M, N).type(dtype).to(device)
            out = model(Y_temp, Mask_temp)
            X = torch.permute(out.reshape(B, T, M, N), [2, 3, 0, 1])
            loss1 = 1 / 2 * torch.norm(Y - torch.mul(X, Mask) - C, 'fro') ** 2
            loss2 = rho / 2 * torch.norm(X - W, 'fro') ** 2
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            # scheduler.step()
            loss_history.append(loss.item())
            loss1_history.append(loss1.item())
            loss2_history.append(loss2.item())
            if epoch % 40 == 0:
                print("\tThe %03dth iters, the %03dth epoch, loss: %.5f" % (iters, epoch, loss.item()))
            Y, C, W, X = Y.detach(), C.detach(), W.detach(), X.detach()

        Y, C, W, X = Y.detach(), C.detach(), W.detach(), X.detach()
        # if up and (rho < rho_init * 1000):
        #     rho *= 1.05
        #     lambda1 *= 1.05
        # balance *= 1.03
        ################################ 更新子问题W ################################
        U, s, VH = torch.linalg.svd(X.reshape(M * N, B * T), full_matrices=False)  # B*T, M*N
        print("s: ", s)
        s = s - lambda1 / rho
        s[s < 0] = 0
        S = torch.diag(s)
        W = torch.mm(torch.mm(U, S), VH).reshape(M, N, B, T)
        ################################ 更新子问题C ################################
        L = Y - torch.mul(X, Mask)
        for t in range(T):
            C[:, :, :, t] = thres_21(L[:, :, :, t], lambda2, M, N, B)

        mse = torch.sum((X - Clean) ** 2)
        Recover = torch.mul(Y, Mask) + torch.mul(X, 1 - Mask)
        psnr_rec = compare_psnr(Recover.detach().cpu().numpy(), Clean.detach().cpu().numpy(), data_range=2)
        psnr = compare_psnr(X.detach().cpu().numpy(), Clean.detach().cpu().numpy(), data_range=2)
        ssim = 0
        for k in range(B):
            for l in range(T):
                ssim += compare_ssim(X[:, :, k, l].detach().cpu().numpy(), Clean[:, :, k, l].detach().cpu().numpy())
        ssim = ssim / (B * T)
        print("The %03dth iters, the %03dth epoch, loss: %.5f, mse: %.5f, psnr: %.5f, ssim: %.5f" % (
            iters, epoch, loss.item(), mse, psnr, ssim))

        image_Clean = Clean.cpu().numpy()
        image_Y = Y.cpu().numpy()
        image_X = X.cpu().detach().numpy()
        image_C = C.cpu().numpy()
        plt.figure(figsize=(20, 20))
        for i in range(4):
            plt.subplot(5, 4, 1 + i)
            plt.title("Clean")
            plt.imshow(image_Clean[:, :, [3, 2, 0], i])
            plt.axis('off')

            plt.subplot(5, 4, 5 + i)
            plt.title("X")
            plt.imshow(image_X[:, :, [3, 2, 0], i])
            plt.axis('off')

            plt.subplot(5, 4, 9 + i)
            plt.title("Y")
            plt.imshow(image_Y[:, :, [3, 2, 0], i])
            plt.axis('off')

            plt.subplot(5, 4, 13 + i)
            plt.title("Y-X")
            plt.imshow(image_Y[:, :, [3, 2, 0], i] - image_X[:, :, [3, 2, 0], i])
            plt.axis('off')

            plt.subplot(5, 4, 17 + i)
            plt.title("C")
            plt.imshow(image_C[:, :, 0, i], cmap='gray')
            plt.axis('off')
        result_path = "%s/iter[%03d, %03d]-pnsr[%.3f, %.3f]-ssim[%.3f]-mse[%.5E]-loss[%.5E].png" % (
            log_dir, iters, epoch, psnr, psnr_rec, ssim, mse, loss.item())
        plt.savefig(result_path)
        plt.clf()
        plt.figure(figsize=(30, 60))
        plt.subplot(6, 1, 1)
        plt.title("total loss")
        plt.plot([i for i in range(len(loss_history[50:]))], loss_history[50:])

        plt.subplot(6, 1, 2)
        plt.title("loss1")
        plt.plot([i for i in range(len(loss1_history[50:]))], loss1_history[50:])

        plt.subplot(6, 1, 3)
        plt.title("loss2")
        plt.plot([i for i in range(len(loss2_history[50:]))], loss2_history[50:])

        plt.subplot(6, 1, 4)
        plt.title("total loss")
        plt.plot([i for i in range(len(loss_history))], loss_history)

        plt.subplot(6, 1, 5)
        plt.title("loss1")
        plt.plot([i for i in range(len(loss1_history))], loss1_history)

        plt.subplot(6, 1, 6)
        plt.title("loss2")
        plt.plot([i for i in range(len(loss2_history))], loss2_history)

        loss_path = "%s/iter[%03d, %03d]-LR[%f]-rho[%.3f]-loss1[%.5E]-loss2[%.5E].png" % (
            log_dir, iters, epoch, LR, rho, loss1.item(), loss2.item())
        plt.savefig(loss_path)
        plt.clf()

        plt.close()
    # decrease = True
    # up = False
