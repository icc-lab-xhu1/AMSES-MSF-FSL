import numpy as np
import pandas as pd
import torch

from model.WPS import Generator, Discriminator, LSTM_AD
from model.LSTM_VAE import LSTMVAE1
from model.MHAR import MHSA_AR
from model.MAD_GAN import MAD_GAN
from model.USAD import USAD
from model.DAGMM import DAGMM
from model.MTAD import MTAD_GAT
from model.OmniAnomaly import OmniAnomaly
from model.CAE_M import CAE_M
from model.MSCRED import MSCRED
from data_factory.data_loader import *
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from sknetwork.ranking import PageRank
import networkx as nx
from torch.utils.data import DataLoader, Dataset

from utils.optimizer import *
from tqdm import tqdm
import  torch.nn as nn

# from solver import EarlyStopping
from src.eval_methods import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prCal(scoreList, prk, rightOne):
    """计算scoreList的prk值

    Params:
        scoreList: list of tuple (node, score)
        prk: the top n nodes to consider
        rightOne: ground truth nodes
    """
    prkSum = 0
    for k in range(min(prk, len(scoreList))):
        if scoreList[k][0] in rightOne:
            prkSum = prkSum + 1
    denominator = min(len(rightOne), prk)
    return prkSum / denominator
def pr_stat(scoreList, rightOne, k=5):
    topk_list = range(1, k + 1)
    prkS = [0] * len(topk_list)
    for j, k in enumerate(topk_list):
        prkS[j] += prCal(scoreList, k, rightOne)
    return prkS

def my_acc(scoreList, rightOne, n=None):
    """Accuracy for Root Cause Analysis with multiple causes.
    Refined from the Acc metric in TBAC paper.
    """
    node_rank = [_[0] for _ in scoreList]
    if n is None:
        n = len(scoreList)
    s = 0.0
    for i in range(len(rightOne)):
        if rightOne[i] in node_rank:
            rank = node_rank.index(rightOne[i]) + 1
            s += (n - max(0, rank - len(rightOne))) / n
        else:
            s += 0
    s /= len(rightOne)
    return s

class EarlyStopping:
    def __init__(self, model_save_path, patience=3, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.model_save_path = model_save_path

    def __call__(self, val_loss, model_G, model_D, Predictor, model, model3, model4, model5, model6, model7, model8, model9, model10, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(self, model_G, model_D, Predictor, model, model3, model4, model5, model6, model7, model8, model9, model10, epoch)
        # elif score < self.best_score + self.delta:
        #     self.counter += 1
        #     #
        #
        #     (f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        else:
            self.best_score = score
            self.save_model(self, model_G, model_D, Predictor, model, model3, model4, model5, model6, model7, model8, model9, model10, epoch)
            self.counter = 0
    def save_model(self, val_loss, model_G, model_D, Predictor, model, model3, model4, model5, model6, model7, model8, model9, model10, epoch):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        # print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_G_state_dict': model_G.state_dict(),
            'model_D_state_dict': model_D.state_dict(),
            'model_P_state_dict': Predictor.state_dict(),
            'model': model.state_dict(),
            'model3': model3.state_dict(),
            'model4': model4.state_dict(),
            'model5': model5.state_dict(),
            'model6': model6.state_dict(),
            'model7': model7.state_dict(),
            'model8': model8.state_dict(),
            'model9': model9.state_dict(),
            'model10': model10.state_dict(),
            # 'optimizerG_state_dict': optimizer_G.state_dict(),
            # 'optimizerD_state_dict': optimizer_D.state_dict(),
            # 'optimizerP_state_dict': optimizer_P.state_dict(),
            # 'accuracy_list': accuracy_list
            }, file_path)
        self.val_loss_min = val_loss
class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, step = 1, horizon=2):
        self.data = data
        self.window = window
        self.horizon = horizon
        self.step = step

    def __getitem__(self, index):
        x = self.data[index: index + self.window]
        y = self.data[index + self.window: index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return (self.data.shape[0] - self.window) // self.step + 1 - self.horizon


class Pretrain(object):
    DEFAULTS = {}
    def __init__(self, config):

        self.__dict__.update(Pretrain.DEFAULTS, **config)


    def train(self, train_data, test_data, label, shape, epochs, values, s):
        print("train shape", train_data.shape)
        print("test shape",test_data.shape)
        print("lable shape", label.shape)

        criterion = nn.MSELoss(reduction='none')
        generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim, input_c=shape)
        discriminator = Discriminator(win_size=self.win_size, input_c=shape)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        predictor = LSTM_AD(feats=shape)
        # predictor = Predictor(win_size=self.win_size, input_c=self.input_c, dropout=0.2)
        optimizer_P = torch.optim.Adam(predictor.parameters(), lr=self.lr)

        model = LSTMVAE1(
            shape, shape, latent_size=self.latent_dim, device=device
        )
        optimizer_lstm = torch.optim.Adam(model.parameters(), lr=self.lr)
        model3 = MHSA_AR(
            shape, self.win_size, shape)
        optimizer_mhar = torch.optim.Adam(model3.parameters(), lr=self.lr)


        model4 = MAD_GAN(
            shape, self.win_size
        )

        disoptimizer = torch.optim.Adam(model4.discriminator.parameters(), lr=self.lr)
        geroptimizer = torch.optim.Adam(model4.generator.parameters(), lr=self.lr)

        model5 = USAD(
            shape, self.win_size, self.latent_dim
        )

        optimizer_usad = torch.optim.Adam(model5.parameters(), lr=self.lr)

        model6 = DAGMM(
            self.win_size, shape, latent=self.latent_dim
        )
        optimizer_dag = torch.optim.Adam(model6.parameters(), lr=self.lr)

        model7 = MTAD_GAT(shape, self.win_size, shape)

        optimizer_mtad = torch.optim.Adam(model7.parameters(), lr=self.lr)

        model8 = OmniAnomaly(shape, latent=self.latent_dim)
        optimizer_Omni = torch.optim.Adam(model8.parameters(), lr=self.lr)

        model9 = CAE_M(shape, self.win_size)
        optimizer_cae = torch.optim.Adam(model9.parameters(), lr=self.lr)

        model10 = MSCRED(shape, self.win_size)
        optimizer_msc = torch.optim.Adam(model10.parameters(), lr = self.lr)


        if torch.cuda.is_available():
            generator.to(device)
            discriminator.to(device)
            predictor.to(device)


            generator.train()
            discriminator.train()
            predictor.train()
            model.to(device)
            model.train()
            model3.to(device)
            model3.train()

            model4.to(device)
            model4.train()

            model5.to(device)
            model5.train()

            model6.to(device)
            model6.train()

            model7.to(device)
            model7.train()

            model8.to(device)
            model8.train()

            model9.to(device)
            model9.train()

            model10.to(device)
            model10.train()


        print("======================"+s+":TRAIN MODE======================")

        Sli_data = SlidingWindowDataset(train_data, self.win_size)
        loader = DataLoader(Sli_data, batch_size=self.batch_size, shuffle=False, num_workers=8)
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(path, patience=15, verbose=False, dataset_name=self.dataset)
        rec_losses = []
        p_losses = []
        last_mse = 0
        for epoch in tqdm(range(epochs)):
            for i, (input_data, y_input_data) in enumerate(loader):

                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                optimizer_P.zero_grad()

                input_data, y_input_data = input_data.double().to(device), y_input_data.double().to(
                    device)  # (b,1,n)

                z = torch.DoubleTensor(
                    np.random.normal(0, 1,
                                     (y_input_data.shape[0], y_input_data.shape[1], y_input_data.shape[2]))).to(
                    device)
                z = z + y_input_data

                fake_input = generator(z)

                p = predictor(input_data)
                p_loss = torch.mean(criterion(p, y_input_data))
                p_loss.backward()
                optimizer_P.step()

                real_input = y_input_data

                # real_input = torch.cat((input_data[:,1:,:], y_input_data), dim=1)
                # Real input
                real_validity = discriminator(real_input)
                # print(real_input.shape)
                # print(real_validity.shape)
                # Fake input
                fake_validity = discriminator(fake_input)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_input, fake_input)

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
                # print('2',d_loss.shape)
                d_loss.backward()
                optimizer_D.step()
                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % 1 == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of input
                    # fake_input, _ , _, _ = generator(z)
                    fake_input = generator(z)

                    fake_validity = discriminator(fake_input)
                    # pre_loss = 0
                    g_loss = -torch.mean(
                        fake_validity)  # + pre_loss # + torch.mean(self.criterion(fake_input, real_input))

                    g_loss.backward()
                    optimizer_G.step()

                    rec_loss = torch.mean(criterion(fake_input, real_input))
                    rec_losses.append(rec_loss.detach().cpu().numpy())

                p_losses.append(p_loss.detach().cpu().numpy())

            pred_data = []
            loss_t1 = []
            # for i, (input_data1, y_input_data1) in enumerate(loader):
            #     # print('f',input_data.shape)
            #     input_data1 = input_data1.to(device)
            #     optimizer_lstm.zero_grad()
            #     loss1, x_hat1, (recon_loss, kld_loss) = model(input_data1)
            #     pred_data.append(x_hat1[:, -1, :].detach().cpu().numpy())
            #
            #     loss1.backward()
            #     optimizer_lstm.step()
            #     loss_t1.append(loss1.item())
            #
            #     # Train the generator every n_critic steps
            # lstm_loss = np.mean(loss_t1)

            criterion_MH = nn.MSELoss(reduction='mean')
            t_loss = []
            for i, (input_data1, y_input_data1) in enumerate(loader):
                # print('f',input_data1.shape)
                y_input_data1 = y_input_data1.to(device)
                input_data1 = input_data1.to(device)
                optimizer_mhar.zero_grad()
                preds = model3(input_data1)
                y_input_data1 = y_input_data1.view(y_input_data1.shape[0], -1)
                forecast_loss = torch.sqrt(criterion_MH(y_input_data1, preds))
                loss = forecast_loss
                # print(loss)
                loss.backward()
                optimizer_mhar.step()
                t_loss.append(loss.item())

            forecast_b_losses = np.array(t_loss)

            MHSA_AR_loss = np.mean(forecast_b_losses)


            bcel = nn.BCELoss(reduction='mean')
            msel = nn.MSELoss(reduction='mean')
            mses = []
            gls = []
            for i, (input_data, y_input_data) in enumerate(loader):
                real_label, fake_label = torch.as_tensor(
                    torch.tensor(np.repeat(0.9, input_data.shape[0]), dtype=torch.float64)), torch.tensor(
                    np.repeat(0.1, input_data.shape[0]), dtype=torch.float64) # label smoothing

                real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                real_label = real_label.to(device)
                fake_label = fake_label.to(device)
                disoptimizer.zero_grad()
                _, real_score, fake_score = model4(input_data)
                # print('real_score', real_score.shape)
                # print('real_label', real_label.shape)
                dl = bcel(real_score, real_label) + bcel(fake_score, fake_label)
                dl.backward()

                disoptimizer.step()
                geroptimizer.zero_grad()
                preds, _, fake_score = model4(input_data)
                mse = msel(preds, input_data)
                gl = bcel(fake_score, real_label)
                loss = mse + gl
                loss.backward()
                geroptimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
            mses_losses = np.array(mses)
            gls_losses = np.array(gls)
            f_loss = np.mean(mses_losses) + np.mean(gls_losses)

            losses = []
            l = nn.MSELoss(reduction='none')
            n = epoch + 1
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data = input_data.to(device)
                ae1s, ae2s, ae2ae1s = model5(input_data)
                input_data = input_data.view(input_data.shape[0], -1)
                l1 = (1 / n) * l(ae1s, input_data) + (1 - 1 / n) * l(ae2ae1s, input_data)
                l2 = (1 / n) * l(ae2s, input_data) - (1 - 1 / n) * l(ae2ae1s, input_data)
                loss = torch.mean(l1 + l2)
                optimizer_usad.zero_grad()
                loss.backward()
                optimizer_usad.step()
                # print(loss)
                losses.append(loss.item())
            losses = np.mean(losses)

            rec_losses_dag = []
            for i, (input_data, y_input_data) in enumerate(loader):
                # print('f',input_data.shape)
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                input_data = input_data.view(input_data.shape[0], -1)
                optimizer_dag.zero_grad()
                _, x_hat, z, gamma = model6(input_data)
                # rec_data.append(pred.detach().cpu().numpy())
                l1, l2 = l(x_hat, input_data), l(gamma, input_data)
                dag_loss = torch.mean(l1) + torch.mean(l2)

                dag_loss.backward()

                optimizer_dag.step()

                rec_losses_dag.append(dag_loss.item())

            mtad_losses = []
            for i, (input_data, y_input_data) in enumerate(loader):
                # print('f',input_data.shape)
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                forecast_criterion = nn.MSELoss()
                optimizer_mtad.zero_grad()
                preds, recons = model7(input_data)
                y_input_data = y_input_data.view(y_input_data.shape[0], -1)
                forecast_loss = torch.sqrt(forecast_criterion(y_input_data, preds))
                recon_loss = torch.sqrt(forecast_criterion(input_data, recons))
                loss_mtad = forecast_loss + recon_loss
                loss_mtad.backward()

                optimizer_mtad.step()

                mtad_losses.append(loss_mtad.item())
            rec_omni = []
            for i, (input_data, y_input_data) in enumerate(loader):

                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                optimizer_Omni.zero_grad()
                y_pred, mu, logvar, hidden = model8(input_data)
                MSE = l(y_pred, input_data)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss_omni = MSE + model8.beta * KLD
                # print(loss)
                loss_omni = torch.mean(loss_omni)

                loss_omni.backward()

                optimizer_Omni.step()

                rec_omni.append(loss_omni.item())

            losses_cae = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                x = model9(input_data)
                loss_cae = torch.mean(l(x, input_data))
                optimizer_cae.zero_grad()
                loss_cae.backward()
                optimizer_cae.step()
                losses_cae.append(loss_cae.item())

            losses_msc = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                x = model10(input_data)
                loss_msc = torch.mean(l(x, input_data))
                optimizer_cae.zero_grad()
                loss_msc.backward()
                optimizer_msc.step()
                losses_msc.append(loss_msc.item())


            if epoch % 1 == 0:
                mse = np.average(rec_losses) +  MHSA_AR_loss + f_loss + losses + np.average(rec_losses_dag) + np.average(mtad_losses) + np.average(rec_omni) + np.mean(losses_cae) + np.mean(losses_msc)
                # tqdm.write(
                #     "Epoch: {0}, Steps: {1} | g_loss Loss: {2:.7f} d_loss Loss: {3:.7f} lstm_vae Loss: {4:.7f} MHSA_AR Loss: {5:.7f} MAD_GAN Loss:{6:.7f} USAD Loss:{7:.7f} DAGMM Loss:{8:.7f} MTAD Loss:{9:.7f} Omni Loss:{10:.7} CAE_M Loss:{11:.7} MSCRED Loss:{12:.7}".format(
                #         epoch + 1, i, g_loss.item(), d_loss.item(), last_mse - mse, lstm_loss, MHSA_AR_loss, f_loss, losses, dag_loss.item(), loss_mtad.item(), loss_omni.item(), loss_cae.item(), loss_msc.item()))
                tqdm.write(
                    "Epoch: {0}, Steps: {1} | g_loss Loss: {2:.7f} d_loss Loss: {3:.7f}  MHSA_AR Loss: {4:.7f} MAD_GAN Loss:{5:.7f} USAD Loss:{6:.7f} DAGMM Loss:{7:.7f} MTAD Loss:{8:.7f} Omni Loss:{9:.7} CAE_M Loss:{10:.7} MSCRED Loss:{11:.7}".format(
                        epoch + 1, i, g_loss.item(), d_loss.item(), last_mse - mse,  MHSA_AR_loss, f_loss,
                        losses, dag_loss.item(), loss_mtad.item(), loss_omni.item(), loss_cae.item(), loss_msc.item()))
                last_mse = mse
            early_stopping(mse, generator, discriminator, predictor, model, model3, model4, model5, model6, model7, model8, model9, model10, epoch)

        fname = f'{self.model_save_path}_{self.dataset}/model.ckpt'
        checkpoint = torch.load(fname, map_location='cpu')

        generator.load_state_dict(checkpoint['model_G_state_dict'])
        discriminator.load_state_dict(checkpoint['model_D_state_dict'])
        predictor.load_state_dict(checkpoint['model_P_state_dict'])

        # model.load_state_dict(checkpoint['model'])
        # model.to(device)
        # model.eval()

        model3.load_state_dict(checkpoint['model3'])
        model3.to(device)
        model3.eval()

        model4.load_state_dict(checkpoint['model4'])
        model4.to(device)
        model4.eval()

        model5.load_state_dict(checkpoint['model5'])
        model5.to(device)
        model5.eval()

        model6.load_state_dict(checkpoint['model6'])
        model6.to(device)
        model6.eval()

        model7.load_state_dict(checkpoint['model7'])
        model7.to(device)
        model7.eval()

        model8.load_state_dict(checkpoint['model8'])
        model8.to(device)
        model8.eval()

        model9.load_state_dict(checkpoint['model9'])
        model9.to(device)
        model9.eval()

        model10.load_state_dict(checkpoint['model10'])
        model10.to(device)
        model10.eval()

        generator.to(device)
        discriminator.to(device)
        predictor.to(device)
        generator.eval()
        discriminator.eval()
        predictor.eval()

        print("======================"+s+":TEST MODE======================")
        label = label[self.win_size: -1]
        two_label = []
        for i in label:
            if i == values:
                a = 1
                two_label.append(a)
            else:
                a = 0
                two_label.append(a)
        two_label = np.array(two_label)
        Sli_data = SlidingWindowDataset(test_data, self.win_size)
        loader = DataLoader(Sli_data, batch_size=self.batch_size, shuffle=False, num_workers=8)

        criterion = nn.MSELoss(reduction='none')

        test_energy = []

        for i, (input_data, y_input_data) in enumerate(loader):
            input_data, y_input_data = input_data.double().to(device), y_input_data.double().to(
                device)  # (b,1,n)

            # z = torch.FloatTensor(np.random.normal(0, 1, (input_data.shape[0], self.latent_dim))).to(device) # (input_data.shape[0], self.latent_dim)
            z = torch.DoubleTensor(
                np.random.normal(0, 1,
                                 (y_input_data.shape[0], y_input_data.shape[1], y_input_data.shape[2]))).to(
                device)
            z = z + y_input_data
            # fake_input, _, _, _ = generator(z)
            fake_input = generator(z)
            g_loss = criterion(y_input_data, fake_input)  # (b,w,n)

            p = predictor(input_data)
            p_loss = criterion(p, y_input_data)
            d_loss = ((discriminator(y_input_data)) * (-1) + 1)

            loss = (self.alpha) * g_loss + (
                    1 - self.alpha) * d_loss + self.beta * p_loss  # torch.Size([32, 6, 38])
            # print('loss',loss.shape)
            loss = torch.mean(loss, dim=1)  # (b,n)

            win_loss = torch.mean(loss, dim=-1)  # (b)
            # print('win_loss',win_loss.shape)
            test_energy.append(win_loss.detach().cpu().numpy())

        test_energy = np.concatenate(test_energy, axis=0).reshape(-1)

        test_energy = np.array(test_energy)

        score = test_energy


        bf_eval_1 = bf_search(score, two_label, start=0.001, end=1, step_num=150, verbose=False)
        # print('WPS', bf_eval_1)
        actual = []
        recons = []
        # for i, (input_data, y_input_data) in enumerate(loader):
        #     input_data = input_data.to(device)
        #     y_input_data = y_input_data.to(device)
        #     recon_x = torch.cat((input_data[:, 1:, :], y_input_data), dim=1)
        #     actual.append(recon_x[:, -1, :].detach().cpu().numpy())
        #     loss1, x_hat1, (recon_loss, kld_loss) = model(recon_x)
        #     recons.append(x_hat1[:, -1, :].detach().cpu().numpy())
        # recons = np.concatenate(recons, axis=0)
        # actual = np.concatenate(actual, axis=0)
        # # print(actual.shape)
        # # np.save(r'C:\pycharmwork\Recon_Cas\dataset\actual.npy',actual)
        # a_score = np.sqrt((recons - actual) ** 2)
        # anomaly_scores = np.mean(a_score, 1)
        # # test_label = data['label'].values
        #
        # # print(np.sum(two_label))
        # bf_eval_2 = bf_search(anomaly_scores, two_label, start=0.001, end=1, step_num=150, verbose=False)
        # print('LSTM-VAE', bf_eval_2)

        preds = []
        actual = []
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data = input_data.to(device)
            y_input_data = y_input_data.to(device)
            y_input_data = y_input_data.view(y_input_data.shape[0], -1)
            actual.append(y_input_data.detach().cpu().numpy())
            y_hat = model3(input_data)
            preds.append(y_hat.detach().cpu().numpy())
        actual = np.concatenate(actual, axis = 0)
        preds = np.concatenate(preds, axis=0)
        a_score = np.sqrt((preds - actual) ** 2)
        anomaly_scores = np.mean(a_score, 1)

        # print(len(two_label1))
        bf_eval_3 = bf_search(anomaly_scores, two_label, start=0.001, end=1, step_num=150, verbose=False)

        recons = []
        actual = []
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data = input_data.to(device)
            y_input_data = y_input_data.to(device)
            recon_x = torch.cat((input_data[:, 2:, :], y_input_data), dim=1)
            actual.append(recon_x[:, -2, :].detach().cpu().numpy())
            window_recon, _, __ = model4(recon_x)
            recons.append(window_recon[:, -2, :].detach().cpu().numpy())

        recons = np.concatenate(recons, axis=0)
        actual = np.concatenate(actual, axis=0)
        a_score = np.sqrt((recons - actual) ** 2)
        anomaly_scores = np.mean(a_score, 1)

        bf_eval_4 = bf_search(anomaly_scores, two_label, start=0.001, end=1, step_num=150, verbose=False)
        losses = []
        l = nn.MSELoss(reduction='none')
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data = input_data.to(device)
            y_input_data = y_input_data.to(device)
            recon_x = torch.cat((input_data[:, 2:, :], y_input_data), dim=1)
            # actual.append(recon_x[:, -1, :].detach().cpu().numpy())
            ae1, ae2, ae2ae1 = model5(recon_x)
            # print(ae1.shape)
            # print(ae2.shape)
            recon_x = input_data.view(recon_x.shape[0], -1)
            # print(input_data.shape)
            loss = 0.1 * l(ae1, recon_x) + 0.9 * l(ae2ae1, recon_x)
            loss = torch.mean(loss, dim=1)
            # print(loss.shape)
            losses.append(loss.detach().cpu().numpy())

        losses = np.concatenate(losses, axis=0)
        bf_eval_5 = bf_search(losses, two_label, start=0.001, end=1, step_num=150, verbose=False)
        recons = []
        actual = []
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data1 = input_data.to(device)
            y_input_data = y_input_data.to(device)
            recon_x = torch.cat((input_data1[:, 2:, :], y_input_data), dim=1)
            actual.append(recon_x[:, -2, :].detach().cpu().numpy())
            _, x_hat, _, _ = model6(recon_x)
            x_hat = x_hat.view(input_data.shape[0], input_data.shape[1], -1)
            # print(y_pred.shape)
            recons.append(x_hat[:, -2, :].detach().cpu().numpy())
        recons = np.concatenate(recons, axis=0)
        # print(recons.shape)
        actual = np.concatenate(actual, axis=0)
        a_score = np.sqrt((recons - actual) ** 2)
        anomaly_scores = np.mean(a_score, 1)
        bf_eval_6 = bf_search(anomaly_scores, two_label, start=0.001, end=1, step_num=150, verbose=False)

        preds = []
        recons = []
        actual = []
        pre_actual = []
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data1 = input_data.to(device)
            y_input_data = y_input_data.to(device)
            y_hat, _ = model7(input_data1)
            recon_x = torch.cat((input_data1[:, 2:, :], y_input_data), dim=1)
            actual.append(recon_x[:, -2, :].detach().cpu().numpy())
            pre_actual.append(y_input_data.detach().cpu().numpy())
            _, window_recon = model7(recon_x)
            preds.append(y_hat.detach().cpu().numpy())
            recons.append(window_recon[:, -2, :].detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = np.concatenate(actual, axis=0)
        pre_actual = np.concatenate(pre_actual, axis=0)
        pre_actual = pre_actual.reshape(pre_actual.shape[0], -1)
        # print(actual.shape)
        a_score = np.mean(np.sqrt((preds - pre_actual) ** 2), 1) + np.mean(np.sqrt((recons - actual) ** 2), 1)
        bf_eval_7 = bf_search(a_score, two_label, start=0.001, end=1, step_num=150, verbose=False)

        recons = []
        actual = []
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data1 = input_data.to(device)
            y_input_data = y_input_data.to(device)
            recon_x = torch.cat((input_data1[:, 2:, :], y_input_data), dim=1)
            actual.append(recon_x[:, -2, :].detach().cpu().numpy())
            y_pred, _, _, hidden = model8(recon_x)
            # print(y_pred.shape)
            recons.append(y_pred[:, -2, :].detach().cpu().numpy())
        recons = np.concatenate(recons, axis=0)
        actual = np.concatenate(actual, axis=0)
        a_score = np.sqrt((recons - actual) ** 2)
        anomaly_scores = np.mean(a_score, 1)
        bf_eval_8 = bf_search(anomaly_scores, two_label, start=0.001, end=1, step_num=150, verbose=False)

        recons = []
        actual = []
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data1 = input_data.to(device)
            y_input_data = y_input_data.to(device)
            recon_x = torch.cat((input_data1[:, 2:, :], y_input_data), dim=1)
            actual.append(recon_x[:, -2, :].detach().cpu().numpy())
            x_hat = model9(recon_x)
            # print(y_pred.shape)
            recons.append(x_hat[:, -2, :].detach().cpu().numpy())
        recons = np.concatenate(recons, axis=0)
        actual = np.concatenate(actual, axis=0)
        a_score = np.sqrt((recons - actual) ** 2)
        anomaly_scores = np.mean(a_score, 1)
        bf_eval_9 = bf_search(anomaly_scores, two_label, start=0.001, end=1, step_num=150, verbose=False)

        recons = []
        actual = []
        for i, (input_data, y_input_data) in enumerate(loader):
            input_data1 = input_data.to(device)
            y_input_data = y_input_data.to(device)
            recon_x = torch.cat((input_data1[:, 2:, :], y_input_data), dim=1)
            actual.append(recon_x[:, -2, :].detach().cpu().numpy())
            x_hat = model10(recon_x)
            # print(y_pred.shape)
            recons.append(x_hat[:, -2, :].detach().cpu().numpy())
        recons = np.concatenate(recons, axis=0)
        actual = np.concatenate(actual, axis=0)
        a_score = np.sqrt((recons - actual) ** 2)
        anomaly_scores = np.mean(a_score, 1)
        bf_eval_10 = bf_search(anomaly_scores, two_label, start=0.001, end=1, step_num=150, verbose=False)



        # dict = {'MHAR': bf_eval_3['f1'], 'LSTM-VAE': bf_eval_2['f1'], 'WPS': bf_eval_1['f1'], 'MAD_GAN': bf_eval_4['f1'], 'USAD': bf_eval_5['f1'], 'DAGMM': bf_eval_6['f1'], 'MTAD':bf_eval_7['f1'], "Omni": bf_eval_8['f1'], "CAE_M":bf_eval_9['f1'], "MSCRED":bf_eval_10['f1']}
        dict = {'MHAR': bf_eval_3['f1'], 'WPS': bf_eval_1['f1'],
                'MAD_GAN': bf_eval_4['f1'], 'USAD': bf_eval_5['f1'], 'DAGMM': bf_eval_6['f1'], 'MTAD': bf_eval_7['f1'],
                "Omni": bf_eval_8['f1'], "CAE_M": bf_eval_9['f1'], "MSCRED": bf_eval_10['f1']}
        return dict
    def rca(self, data, true_root_cause):

        cg = pc(data.to_numpy(), 0.05, fisherz, False, 0, -1)
        adj = cg.G.graph
        # print(i + 'PC result')
        # print(adj)
        # Change the adj to graph
        G = nx.DiGraph()
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i, j] == -1:
                    G.add_edge(i, j)
                if adj[i, j] == 1:
                    G.add_edge(j, i)
        nodes = sorted(G.nodes())
        adj_pc = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))
        pagerank = PageRank()
        try:
            scores = pagerank.fit_transform(adj_pc.T)
            dict_s = {}
            for inx, i in enumerate(nodes):
                dict_s.update({i: scores[inx]})
            rank_pc = sorted(dict_s.items(), key=lambda item: item[1], reverse=True)
            # print('rank',rank)
            prks_pc = pr_stat(rank_pc, true_root_cause, k=3)
            # acc_pc = my_acc(rank_pc, true_root_cause)
        except ValueError:
            prks_pc = 0
        except Exception:
            prks_pc = 0

        X = data
        # X = data.iloc[:,1:]
        X = X - np.tile(np.mean(X, axis=0), (X.shape[0], 1))
        X = np.dot(X, np.diag(1 / np.std(X, axis=0)))
        maxP = 5  # maximum number of parents when searching the graph
        parameters = {'kfold': 10, 'lambda': 0.01}
        # X = X[:50, :]
        Record = ges(X, 'local_score_CV_general', maxP=maxP, parameters=parameters)
        adj = Record['G'].graph

        # Change the adj to graph
        G = nx.DiGraph()
        for i in range(len(adj)):
            for j in range(len(adj)):
                if adj[i, j] == -1:
                    G.add_edge(i, j)
                if adj[i, j] == 1:
                    G.add_edge(j, i)
        nodes = sorted(G.nodes())
        # print(nodes)
        adj_ges = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))
        # print(adj)
        pos = nx.circular_layout(G)
        nx.draw(G, pos=pos, with_labels=True)

        pagerank = PageRank()
        try:
            scores = pagerank.fit_transform(adj_ges.T)
        # print(scores)
        #cmap = plt.cm.coolwarm

            dict_s = {}
            for inx, i in enumerate(nodes):
                dict_s.update({i:scores[inx]})
            rank_ges = sorted(dict_s.items(), key=lambda item:item[1], reverse=True)
            prks_ges = pr_stat(rank_ges, true_root_cause, k=3)
            # acc_ges = my_acc(rank_ges, true_root_cause)
        except ValueError:
            prks_ges = 0
        except Exception:
            prks_ges = 0


        model = lingam.ICALiNGAM()
        model.fit(X)

        # print('LiNGAM result')
        # print(model.causal_order_)  # the later virable is unable to cause the former virable
        # print(model.adjacency_matrix_)

        # Acutal matrix: row points to col
        adj_lin = model.adjacency_matrix_.T
        # For PageRank, should adj.T
        # print(adj)
        org_G = nx.from_numpy_matrix(adj_lin, parallel_edges=True, create_using=nx.DiGraph)
        pos=nx.circular_layout(org_G)
        nx.draw(org_G, pos=pos, with_labels=True)

        # PageRank

        pagerank = PageRank()
        try:
            scores = pagerank.fit_transform(np.abs(adj_lin.T))
            # print(scores)
            # cmap = plt.cm.coolwarm

            score_dict = {}
            for i, s in enumerate(scores):
                score_dict[i] = s
            rank_lin = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
            prks_lin = pr_stat(rank_lin, true_root_cause, k=3)
        # acc_lin = my_acc(rank_ges, true_root_cause)
        except ValueError:
            prks_lin = 0
        except Exception:
            prks_lin = 0


        rca_dict = {'PC': np.mean(prks_pc), 'GES': np.mean(prks_ges), 'Lingam': np.mean(prks_lin)}
        # rca_dict = {'PC': acc_pc, 'GES': acc_ges, 'Lingam': acc_lin}
        return rca_dict

