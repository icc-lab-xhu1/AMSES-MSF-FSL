import pandas as pd
from sklearn import metrics
from utils.utils import *
from model.WPS import Generator, Discriminator, LSTM_AD
from model.LSTM_VAE import LSTMVAE1
from model.MHAR import MHSA_AR
from model.MAD_GAN import MAD_GAN
from model.USAD import USAD
from model.DAGMM import DAGMM
from model.MTAD import MTAD_GAT
from model.CAE_M import CAE_M
from model.OmniAnomaly import OmniAnomaly
from model.MSCRED import MSCRED
# from data_factory.data_loader import *
from data_factory.data_loader_train import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pickle as pk
from src.eval_methods import *

from utils.optimizer import *



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


# class dataSegLoader(object):
#     def __len__(self):
#         if self.mode == "train":
#
#         elif (self.mode == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1 - self.horizon
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.mode == "train":
#             return np.double(self.train[index:index + self.win_size]), np.double(self.train[index + self.win_size:index + self.win_size + self.horizon])
#         elif (self.mode == 'test'):
#             return np.double(self.test[index:index + self.win_size]), np.double(self.test[index + self.win_size:index + self.win_size + self.horizon])
#


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


    def save_model_WPS(self, val_loss, model_G, model_D, Predictor,  epoch, data_name, num):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        # print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'model_G_state_dict': model_G.state_dict(),
            'model_D_state_dict': model_D.state_dict(),
            'model_P_state_dict': Predictor.state_dict(),
        }, file_path)
        self.val_loss_min = val_loss

    def save_model_LSTM(self, val_loss,  model, epoch, data_name, num):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        # print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'LSTM-VAE': model.state_dict(),
            # 'optimizerG_state_dict': optimizer_G.state_dict(),
            # 'optimizerD_state_dict': optimizer_D.state_dict(),
            # 'optimizerP_state_dict': optimizer_P.state_dict(),
            # 'accuracy_list': accuracy_list
        }, file_path)
        self.val_loss_min = val_loss
    def save_model_MHAR(self, val_loss,  model3, epoch, data_name ,num):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        # print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'MHAR': model3.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss

    def save_model_MAD(self, val_loss, model4, epoch, data_name, num):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        # print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'MAD_GAN': model4.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss
    def save_model_USAD(self, val_loss, model5, epoch, data_name, num):
        # if self.verbose:
        #     print('Validation loss decreased (%f --> %f).  Saving model ...'%(float(self.val_loss_min),float(val_loss)))
        # print('Saving model ...')
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'USAD': model5.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss

    def save_model_DAGMM(self, val_loss, model6, epoch, data_name, num):

        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'DAGMM': model6.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss

    def save_model_MTAD(self, val_loss, model7, epoch, data_name, num):
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'MTAD': model7.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss

    def save_model_Omni(self, val_loss, model8, epoch, data_name, num):
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'Omni': model8.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss

    def save_model_CAE(self, val_loss, model9, epoch, data_name, num):
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'CAE_M': model9.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss
    def save_model_MSC(self, val_loss, model10, epoch, data_name, num):
        folder = f'{self.model_save_path}_{self.dataset}_{data_name}_{num}/'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'MSCRED': model10.state_dict(),

        }, file_path)
        self.val_loss_min = val_loss


# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss(reduction='none')

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device(device)
        self.criterion = nn.MSELoss(reduction='none')

    def train(self, excute, data_name):
        # print(excute, num)
        print("======================TRAIN MODE" + excute + data_name + "======================")
        confidence_margin = 0.5
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(path, patience=15, verbose=False, dataset_name=self.dataset)
        rec_losses = []
        p_losses = []
        last_mse = 0
        self.cur_dataset, self.train_loader = get_loader_segment(self.data_path, anomaly = self.anomaly, batch_size=self.batch_size,
                                                                 win_size=self.win_size,
                                                                 mode='train',
                                                                 dataset=data_name)
        print(self.cur_dataset.train.shape)
        if excute == 'WPS':

            generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim,
                                  input_c=self.cur_dataset.train.shape[1] - 1)
            discriminator = Discriminator(win_size=self.win_size, input_c=self.cur_dataset.train.shape[1] - 1)
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

            predictor = LSTM_AD(feats=self.cur_dataset.train.shape[1] - 1)
            optimizer_P = torch.optim.Adam(predictor.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                generator.to(device)

                discriminator.to(device)
                predictor.to(device)

                generator.train()
                discriminator.train()
                predictor.train()
            for epoch in tqdm(range(self.num_epochs)):
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    # print('input:', input_data.shape)
                    # print('y_input:', y_input_data.shape)
                    optimizer_G.zero_grad()
                    optimizer_D.zero_grad()
                    optimizer_P.zero_grad()

                    input_data, y_input_data = input_data.double().to(self.device), y_input_data.double().to(
                        self.device)  # (b,1,n)
                    y_true = y_input_data[:, :, -1]


                    input_data = input_data[:, :, :-1]
                    y_input_data = y_input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # print('y_true', y_true.shape)
                    # y_true = y_true.reshape(y_true.shape[0], -1)

                    # z = torch.FloatTensor(np.random.normal(0, 1, (input_data.shape[0], self.latent_dim))).to(device) # (input_data.shape[0], self.latent_dim)
                    z = torch.DoubleTensor(
                        np.random.normal(0, 1,
                                         (y_input_data.shape[0], y_input_data.shape[1], y_input_data.shape[2]))).to(
                        device)
                    z = z + y_input_data
                    # print(z.shape)
                    # Generate a batch of input
                    # fake_input, _ , _, _= generator(z)
                    fake_input = generator(z)

                    p = predictor(input_data)
                    # print('p',p.shape)
                    # print('y',y_input_data.shape)
                    p_loss = self.criterion(p, y_input_data)

                    dev1 = p_loss
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, p_loss.shape[-1]), dtype=torch.float64)).to(device)

                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    # print('inlier_loss1', inlier_loss1.shape)
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)
                    loss1 = torch.mean(loss1)
                    loss1.backward()
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

                        rec_loss = torch.mean(self.criterion(fake_input, real_input))
                        rec_losses.append(rec_loss.detach().cpu().numpy())

                    p_losses.append(loss1.detach().cpu().numpy())
                if epoch % 1 == 0:
                    mse = np.average(rec_losses)
                    tqdm.write(
                        "Epoch: {0}, Steps: {1} | g_loss Loss: {2:.7f} d_loss Loss: {3:.7f} MSE: {4:.7f} SPD: {5:.7f} PMSE: {6:.7f}".format(
                            epoch + 1, i, loss1.item(), d_loss.item(), mse, last_mse - mse, np.average(p_losses)))
                #     last_mse = mse
                early_stopping.save_model_WPS(mse, generator, discriminator, predictor, epoch, data_name, excute)
        if excute == "LSTM-VAE":
            model = LSTMVAE1(
                self.cur_dataset.train.shape[1] - 1, self.cur_dataset.train.shape[1] - 1, latent_size=self.latent_dim,
                device=device
            )
            optimizer_P1 = torch.optim.Adam(model.parameters(), lr=self.lr)

            if torch.cuda.is_available():
                model.to(device)
                model.train()
            for epoch in tqdm(range(self.num_epochs)):
                pred_data = []
                loss_t1 = []
                for i, (input_data1, y_input_data1) in enumerate(self.train_loader):
                    # print('f',input_data.shape)
                    input_data1 = input_data1.to(self.device)
                    y_input_data1 = y_input_data1.to(self.device)
                    y_true = input_data1[:, :, -1]
                    input_data = input_data1[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # y_true = y_true.reshape(y_true.shape[0], -1)

                    optimizer_P1.zero_grad()
                    loss1, x_hat1, (recon_loss, kld_loss) = model(input_data)

                    dev1 = loss1
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, loss1.shape[-1]), dtype=torch.float64)).to(device)

                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)
                    loss1 = torch.mean(loss1)
                    # print('loss',loss1.shape)

                    loss1.backward()
                    optimizer_P1.step()
                    loss_t1.append(loss1.item())

                lstm_loss = np.mean(loss_t1)
                if epoch % 1 == 0:
                    mse = lstm_loss
                    tqdm.write(
                        "Epoch: {0}, Steps: {1} |  LSTM_VAE Loss: {2:.7f}".format(
                            epoch + 1, i, lstm_loss))
                #     last_mse = mse
                early_stopping.save_model_LSTM(mse, model, epoch, data_name, excute)

        if excute == 'MHAR':
            model3 = MHSA_AR(
                self.cur_dataset.train.shape[1] - 1, self.win_size, self.cur_dataset.train.shape[1] - 1)
            optimizer_P3 = torch.optim.Adam(model3.parameters(), lr=self.lr)

            if torch.cuda.is_available():
                model3.to(device)
                model3.train()
            for epoch in tqdm(range(self.num_epochs)):

                # criterion_MH = nn.MSELoss(reduction='mean')
                t_loss = []
                for i, (input_data1, y_input_data1) in enumerate(self.train_loader):
                    # print('f',input_data1.shape)
                    y_input_data1 = y_input_data1.to(device)
                    input_data1 = input_data1.to(device)
                    # print('y_input', y_input_data1.shape)
                    # print(y_input_data1)
                    y_true = y_input_data1[:, :, -1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data1.shape[2] - 1)
                    y_true = y_true.reshape(y_true.shape[0], -1)
                    # print('y_true', y_true.shape)
                    input_data1 = input_data1[:, :, :-1]
                    y_input_data1 = y_input_data1[:, :, :-1]

                    # print('x', input_data1)
                    optimizer_P3.zero_grad()
                    preds = model3(input_data1)
                    # print('preds',preds.shape)
                    y_input_data1 = y_input_data1.reshape(y_input_data1.shape[0], -1)
                    # a_score = np.sqrt((recons - actual) ** 2)
                    forecast_loss = torch.sqrt((y_input_data1 - preds) ** 2)
                    # print('fore', forecast_loss.shape)

                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, forecast_loss.shape[1]), dtype=torch.float64)).to(device)
                    dev = forecast_loss
                    inlier_loss = torch.abs(dev)
                    outlier_loss = torch.abs(torch.maximum(confidence_margin - dev, other))
                    # print('inlier_loss', inlier_loss.shape)
                    loss1 = (1 - y_true) * (inlier_loss) + y_true * (outlier_loss)
                    # print(forecast_loss)

                    loss1 = torch.mean(loss1)

                    loss = loss1
                    # print(loss)
                    loss.backward()
                    optimizer_P3.step()
                    t_loss.append(loss.item())

                forecast_b_losses = np.array(t_loss)
                MHSA_AR_loss = np.mean(forecast_b_losses)
                if epoch % 1 == 0:
                    mse = MHSA_AR_loss
                    tqdm.write(
                        "Epoch: {0}, Steps: {1} |  MHSA_AR Loss: {2:.7f}".format(
                            epoch + 1, i, MHSA_AR_loss))
                    # last_mse = mse
                early_stopping.save_model_MHAR(mse, model3, epoch, data_name, excute)

        if excute == "MAD_GAN":
            model4 = MAD_GAN(
                self.cur_dataset.train.shape[1] - 1, self.win_size
            )

            disoptimizer = torch.optim.Adam(model4.discriminator.parameters(), lr=self.lr)
            geroptimizer = torch.optim.Adam(model4.generator.parameters(), lr=self.lr)

            if torch.cuda.is_available():
                model4.to(device)
                model4.train()
            mses = []
            gls = []
            # bcel = nn.BCELoss(reduction='none')
            bcel = nn.BCELoss(reduction='mean')
            msel = nn.MSELoss(reduction='none')
            for epoch in tqdm(range(self.num_epochs)):
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    real_label, fake_label = torch.as_tensor(
                        torch.tensor(np.repeat(0.9, input_data.shape[0]), dtype=torch.float64)), torch.tensor(
                        np.repeat(0.1, input_data.shape[0]), dtype=torch.float64)  # label smoothing
                    real_label = real_label.clone().detach()
                    fake_label = fake_label.clone().detach()

                    real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
                    input_data = input_data.to(device)
                    y_input_data = y_input_data.to(device)

                    y_true = input_data[:, :, -1]
                    input_data = input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # y_true = y_true.reshape(y_true.shape[0], -1)

                    real_label = real_label.to(self.device)
                    fake_label = fake_label.to(self.device)
                    disoptimizer.zero_grad()
                    _, real_score, fake_score = model4(input_data)

                    dl = bcel(real_score, real_label) + bcel(fake_score, fake_label)
                    dl.backward()

                    disoptimizer.step()
                    geroptimizer.zero_grad()
                    preds, _, fake_score = model4(input_data)
                    mse = msel(preds, input_data)
                    gl = bcel(fake_score, real_label)

                    dev1 = mse
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, mse.shape[-1]), dtype=torch.float64)).to(device)

                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)
                    loss1 = torch.mean(loss1)
                    mses.append(loss1.item())
                    loss = loss1 + gl
                    loss.backward()
                    geroptimizer.step()

                    gls.append(gl.item())
                mses_losses = np.array(mses)
                # print(f_loss)
                gls_losses = np.array(gls)
                f_loss = np.mean(mses_losses) + np.mean(gls_losses)

                if epoch % 1 == 0:
                    mse = np.average(f_loss)
                    tqdm.write(
                        "Epoch:{0}, Step:{1}, loss:{2:.7f}".format(epoch + 1, i, f_loss)
                    )
                    # last_mse = mse
                early_stopping.save_model_MAD(mse, model4, epoch, data_name, excute)
        if excute == "USAD":
            l = nn.MSELoss(reduction='none')
            model5 = USAD(
                self.cur_dataset.train.shape[1] - 1, self.win_size, self.latent_dim
            )

            optimizer = torch.optim.Adam(model5.parameters(), lr=self.lr)

            if torch.cuda.is_available():
                model5.to(device)
                model5.train()
            for epoch in tqdm(range(self.num_epochs)):
                n = epoch + 1
                losses = []
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    input_data = input_data.to(device)
                    y_true = input_data[:, :, -1]
                    input_data = input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    y_true = y_true.reshape(y_true.shape[0], -1)

                    ae1s, ae2s, ae2ae1s = model5(input_data)
                    input_data = input_data.reshape(input_data.shape[0], -1)
                    l1 = (1 / n) * l(ae1s, input_data) + (1 - 1 / n) * l(ae2ae1s, input_data)
                    l2 = (1 / n) * l(ae2s, input_data) - (1 - 1 / n) * l(ae2ae1s, input_data)

                    dev1 = l1
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, l1.shape[1]), dtype=torch.float64)).to(device)

                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)

                    dev2 = l2
                    inlier_loss2 = torch.abs(dev2)
                    outlier_loss2 = torch.abs(torch.maximum(confidence_margin - dev2, other))
                    loss2 = (1 - y_true) * (inlier_loss2) + y_true * (outlier_loss2)

                    #
                    # print('l1', loss1.shape)
                    # print('l2', loss2.shape)
                    loss = torch.mean(loss1 + loss2)
                    # print('loss',loss)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                losses = np.mean(losses)
                # print(losses)
                # losses = np.mean(losses)
                if epoch % 1 == 0:
                    mse = losses
                    tqdm.write(
                        "Epoch:{0}, Step:{1}, loss:{2:.7f}".format(epoch + 1, i, losses)
                    )
                    # last_mse = mse
                early_stopping.save_model_USAD(mse, model5, epoch, data_name, excute)

        if excute == 'DAGMM':
            model6 = DAGMM(self.win_size, self.cur_dataset.train.shape[1] - 1, latent=self.latent_dim)
            optimizer_P1 = torch.optim.Adam(model6.parameters(), lr=self.lr)

            if torch.cuda.is_available():
                model6.to(device)
            # print(model6.device)
            rec_losses = []
            rec_data = []
            l = nn.MSELoss(reduction='none')
            for epoch in tqdm(range(self.num_epochs)):
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    # print('f',input_data.shape)
                    input_data = input_data.to(device)
                    # print(input_data.shape)
                    y_true = input_data[:, :, -1]
                    # print('input_data',input_data.shape)
                    y_input_data = y_input_data.to(device)
                    input_data = input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # print('y_true', y_true)
                    y_true = y_true.reshape(y_true.shape[0], -1)
                    # print(input_data)

                    y_input_data = y_input_data[:, :, :-1]

                    input_data = input_data.reshape(input_data.shape[0], -1)
                    optimizer_P1.zero_grad()
                    _, x_hat, z, gamma = model6(input_data)
                    l1, l2 = l(x_hat, input_data), l(gamma, input_data)

                    dev1 = l1
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, l1.shape[-1]), dtype=torch.float64)).to(device)
                    # print('dev1', c.shape)
                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    # print('inlier_loss1', inlier_loss1.shape)
                    # print('outlier_loss1', outlier_loss1.shape)
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)

                    dev2 = l2
                    inlier_loss2 = torch.abs(dev2)
                    outlier_loss2 = torch.abs(torch.maximum(confidence_margin - dev2, other))
                    loss2 = (1 - y_true) * (inlier_loss2) + y_true * (outlier_loss2)
                    # print('l1', l1.shape)
                    # print('l2', l2.shape)
                    loss = torch.mean(loss1) + torch.mean(loss2)

                    loss.backward()

                    optimizer_P1.step()

                    rec_losses.append(loss.item())
                # rec_losses = np.mean(rec_losses)
                if epoch % 1 == 0:
                    mse = np.average(rec_losses)
                    tqdm.write(
                        "Epoch:{0}, Step:{1}, loss:{2:.7f}".format(epoch + 1, i, loss.item())
                    )
                early_stopping.save_model_DAGMM(mse, model6, epoch, data_name, excute)
        if excute == 'MTAD':
            model7 = MTAD_GAT(self.cur_dataset.train.shape[1] - 1, self.win_size, self.cur_dataset.train.shape[1] - 1)

            optimizer = torch.optim.Adam(model7.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                model7.to(device)
            mtad_losses = []
            for epoch in tqdm(range(self.num_epochs)):
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    # print('f',input_data.shape)
                    input_data = input_data.to(device)
                    y_input_data = y_input_data.to(device)

                    ypred_true = y_input_data[:, :, -1]
                    ypred_true = ypred_true.unsqueeze(2)
                    ypred_true = ypred_true.repeat(1, 1, input_data.shape[2] - 1)

                    y_true = input_data[:, :, -1]
                    input_data = input_data[:, :, :-1]
                    y_input_data = y_input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # y_true = y_true.reshape(y_true.shape[0], -1)
                    optimizer.zero_grad()
                    # print(input_data.shape)
                    preds, recons = model7(input_data)
                    y_input_data = y_input_data.reshape(y_input_data.shape[0], -1)
                    forecast_loss = torch.sqrt((y_input_data - preds) ** 2)
                    forecast_loss = forecast_loss.reshape(y_input_data.shape[0], 2, -1)
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, forecast_loss.shape[-1]), dtype=torch.float64)).to(device)
                    dev = forecast_loss
                    inlier_loss = torch.abs(dev)
                    outlier_loss = torch.abs(torch.maximum(confidence_margin - dev, other))
                    # print('ypred_true', ypred_true.shape)
                    # print('inlier_loss1', inlier_loss.shape)
                    loss1 = (1 - ypred_true) * (inlier_loss) + ypred_true * (outlier_loss)
                    loss1 = torch.mean(loss1)
                    # print('loss', loss1.shape)

                    recon_loss = torch.sqrt((input_data - recons) ** 2)
                    dev2 = recon_loss
                    other_recons = torch.as_tensor(
                        torch.tensor(np.repeat(0, recon_loss.shape[-1]), dtype=torch.float64)).to(device)
                    inlier_loss2 = torch.abs(dev2)
                    outlier_loss2 = torch.abs(torch.maximum(confidence_margin - dev2, other_recons))
                    loss2 = (1 - y_true) * (inlier_loss2) + y_true * (outlier_loss2)
                    loss2 = torch.mean(loss2)
                    # print('loss', loss2.shape)
                    loss_mtad = loss1 + loss2
                    # print('loss',loss_mtad.shape)

                    loss_mtad.backward()

                    optimizer.step()

                    mtad_losses.append(loss_mtad.item())
                if epoch % 1 == 0:
                    mse = np.average(mtad_losses)
                    tqdm.write(
                        "Epoch:{0}, Step:{1}, loss:{2:.7f}".format(epoch + 1, i, loss_mtad.item())
                    )
                early_stopping.save_model_MTAD(mse, model7, epoch, data_name, excute)
        if excute == 'Omni':
            model8 = OmniAnomaly(self.cur_dataset.train.shape[1] - 1, latent=self.latent_dim)
            optimizer8 = torch.optim.Adam(model8.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                model8.to(device)

            rec_losses = []
            l = nn.MSELoss(reduction='none')
            for epoch in tqdm(range(self.num_epochs)):
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    input_data = input_data.to(device)
                    y_input_data = y_input_data.to(device)
                    y_true = input_data[:, :, -1]
                    input_data = input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # y_true = y_true.reshape(y_true.shape[0], -1)
                    y_input_data = y_input_data[:, :, :-1]
                    optimizer8.zero_grad()
                    y_pred, mu, logvar, hidden = model8(input_data)
                    MSE = l(y_pred, input_data)
                    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                    loss = MSE + model8.beta * KLD

                    dev1 = loss
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, loss.shape[-1]), dtype=torch.float64)).to(device)
                    # print('dev', dev1.shape)
                    # print('other', other.shape)
                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)

                    # print(loss)
                    loss = torch.mean(loss1)

                    loss.backward()

                    optimizer8.step()

                    rec_losses.append(loss.item())

                if epoch % 1 == 0:
                    mse = np.average(rec_losses)
                    tqdm.write(
                        "Epoch:{0}, Step:{1}, loss:{2:.7f}".format(epoch + 1, i, loss.item())
                    )
                early_stopping.save_model_Omni(mse, model8, epoch, data_name, excute)
        if excute == 'CAE_M':
            model9 = CAE_M(self.cur_dataset.train.shape[1] - 1, self.win_size)
            optimizer9 = torch.optim.Adam(model9.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                model9.to(device)

            rec_losses = []
            l = nn.MSELoss(reduction='none')
            for epoch in tqdm(range(self.num_epochs)):
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    input_data = input_data.to(device)
                    y_input_data = y_input_data.to(device)
                    y_true = input_data[:, :, -1]
                    input_data = input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # y_true = y_true.reshape(y_true.shape[0], -1)

                    optimizer9.zero_grad()
                    y_pred = model9(input_data)
                    loss = l(y_pred, input_data)

                    dev1 = loss
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, loss.shape[-1]), dtype=torch.float64)).to(device)

                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)

                    # print(loss)
                    loss = torch.mean(loss1)

                    loss.backward()

                    optimizer9.step()

                    rec_losses.append(loss.item())

                if epoch % 1 == 0:
                    mse = np.average(rec_losses)
                    tqdm.write(
                        "Epoch:{0}, Step:{1}, loss:{2:.7f}".format(epoch + 1, i, loss.item())
                    )
                early_stopping.save_model_CAE(mse, model9, epoch, data_name, excute)
        if excute == 'MSCRED':
            model10 = MSCRED(self.cur_dataset.train.shape[1] - 1, self.win_size)
            optimizer10 = torch.optim.Adam(model10.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                model10.to(device)

            rec_losses = []
            l = nn.MSELoss(reduction='none')
            for epoch in tqdm(range(self.num_epochs)):
                for i, (input_data, y_input_data) in enumerate(self.train_loader):
                    input_data = input_data.to(device)
                    y_input_data = y_input_data.to(device)
                    y_true = input_data[:, :, -1]
                    input_data = input_data[:, :, :-1]
                    y_true = y_true.unsqueeze(2)
                    y_true = y_true.repeat(1, 1, input_data.shape[2])
                    # y_true = y_true.reshape(y_true.shape[0], -1)

                    optimizer10.zero_grad()
                    y_pred = model10(input_data)
                    loss = l(y_pred, input_data)

                    dev1 = loss
                    other = torch.as_tensor(
                        torch.tensor(np.repeat(0, loss.shape[-1]), dtype=torch.float64)).to(device)

                    inlier_loss1 = torch.abs(dev1)
                    outlier_loss1 = torch.abs(torch.maximum(confidence_margin - dev1, other))
                    loss1 = (1 - y_true) * (inlier_loss1) + y_true * (outlier_loss1)
                    # print(loss)
                    loss = torch.mean(loss1)

                    loss.backward()

                    optimizer10.step()

                    rec_losses.append(loss.item())

                if epoch % 1 == 0:
                    mse = np.average(rec_losses)
                    tqdm.write(
                        "Epoch:{0}, Step:{1}, loss:{2:.7f}".format(epoch + 1, i, loss.item())
                    )
                early_stopping.save_model_MSC(mse, model10, epoch, data_name, excute)

    def test(self, excute, data, label, values_1, data_name):

        print("======================TEST MODE" + data_name + "======================")
        score_1, score_2 = 0, 0
        thre_1, thre_2 = 0, 0
        shape = data.shape[1]
        label = label[self.win_size:-1]
        # if len(data) % 2 == 0:
        #     label = label[self.win_size:]# step == 2
        # else:
        #     label = label[self.win_size:-1]

        # print('label', len(label))
        # model_G = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.input_c, e_layers=1,dropout=0.2)
        generator = Generator(win_size=self.win_size, latent_dim=self.latent_dim, input_c=shape)
        discriminator = Discriminator(win_size=self.win_size, input_c=shape)
        predictor = LSTM_AD(feats=shape)

        model = LSTMVAE1(
            shape, shape, latent_size=self.latent_dim, device=device
        )
        model3 = MHSA_AR(shape, self.win_size, shape)

        model4 = MAD_GAN(shape, self.win_size)

        model5 = USAD(
            shape, self.win_size, self.latent_dim
        )
        model6 = DAGMM(
            self.win_size, shape, latent=self.latent_dim
        )
        model7 = MTAD_GAT(shape, self.win_size, shape)
        model8 = OmniAnomaly(shape, latent=self.latent_dim)


        Sli_data = SlidingWindowDataset(data, self.win_size)
        loader = DataLoader(Sli_data, batch_size=self.batch_size, shuffle=False, num_workers=8)
        criterion = nn.MSELoss(reduction='none')
        if 'WPS' in excute:
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"WPS"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            generator.load_state_dict(checkpoint['model_G_state_dict'])
            discriminator.load_state_dict(checkpoint['model_D_state_dict'])
            predictor.load_state_dict(checkpoint['model_P_state_dict'])
            generator.to(device)
            discriminator.to(device)
            predictor.to(device)
            generator.eval()
            discriminator.eval()
            predictor.eval()
            test_energy = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data, y_input_data = input_data.double().to(self.device), y_input_data.double().to(
                    self.device)  # (b,1,n)

                # z = torch.FloatTensor(np.random.normal(0, 1, (input_data.shape[0], self.latent_dim))).to(device) # (input_data.shape[0], self.latent_dim)
                z = torch.DoubleTensor(
                    np.random.normal(0, 1, (y_input_data.shape[0], y_input_data.shape[1], y_input_data.shape[2]))).to(
                    device)
                z = z + y_input_data
                # fake_input, _, _, _ = generator(z)
                fake_input = generator(z)
                g_loss = criterion(y_input_data, fake_input)  # (b,w,n)

                p = predictor(input_data)
                p_loss = criterion(p, y_input_data)
                d_loss = ((discriminator(y_input_data)) * (-1) + 1)

                loss = (self.alpha) * g_loss + (1 - self.alpha) * d_loss + self.beta * p_loss  # torch.Size([32, 6, 38])
                # print('loss',loss.shape)
                loss = torch.mean(loss, dim=1)  # (b,n)

                win_loss = torch.mean(loss, dim=-1)  # (b)
                # print('win_loss',win_loss.shape)
                test_energy.append(win_loss.detach().cpu().numpy())

            test_energy = np.concatenate(test_energy, axis=0).reshape(-1)

            test_energy = np.array(test_energy)


            score = test_energy
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(score, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'WPS':
                    score_1 = test_energy
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'WPS':
                    score_2 = test_energy
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = test_energy
                score_2 = 0
            # return score
        # data.to_csv(r'/home/user1/SYJ/Multi__Class/mem.csv')
        # data.to_csv(r'./mem.csv')

        if 'LSTM-VAE' in excute:
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"LSTM-VAE"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model.load_state_dict(checkpoint['LSTM-VAE'])
            model.to(device)
            model.eval()
            actual = []
            recons = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                recon_x = torch.cat((input_data[:, 1:, :], y_input_data), dim=1)
                actual.append(recon_x[:, -1, :].detach().cpu().numpy())
                loss1, x_hat1, (recon_loss, kld_loss) = model(recon_x)
                recons.append(x_hat1[:, -1, :].detach().cpu().numpy())
            recons = np.concatenate(recons, axis=0)
            # print(recons.shape)
            actual = np.concatenate(actual, axis=0)
            # print(actual.shape)
            # np.save(r'C:\pycharmwork\Recon_Cas\dataset\actual.npy',actual)
            a_score = np.sqrt((recons - actual) ** 2)
            anomaly_scores = np.mean(a_score, 1)

            # test_label = data['label'].values
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(anomaly_scores, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'LSTM-VAE':
                    score_1 = anomaly_scores
                if excute[1] == 'LSTM-VAE':
                    score_2 = anomaly_scores
            else:
                score_1 = anomaly_scores
                score_2 = 0
            # return anomaly_scores
        # data.to_csv(r'/home/user1/SYJ/Multi__Class/latency.csv')
        if 'MHAR' in excute:
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"MHAR"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model3.load_state_dict(checkpoint['MHAR'])
            model3.to(device)
            model3.eval()
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
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(anomaly_scores, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'MHAR':
                    score_1 = anomaly_scores
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'MHAR':
                    score_2 = anomaly_scores
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = anomaly_scores
                score_2 = 0
            # return anomaly_scores

        if 'MAD_GAN' in excute:
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"MAD_GAN"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model4.load_state_dict(checkpoint['MAD_GAN'])
            model4.to(device)
            model4.eval()
            recons = []
            actual = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                recon_x = torch.cat((input_data[:, 2:, :], y_input_data), dim=1)
                actual.append(recon_x[:, -2, :].detach().cpu().numpy())
                # print('actual', len(actual))
                window_recon, _, __ = model4(recon_x)
                # window_recon = window_recon[:, -2:, :].reshape(y_input_data.shape)
                # print('window shape:', window_recon.shape)

                # np.save(r'./result/window' + str(i) + '.npy', window_recon.detach().cpu().numpy())
                recons.append(window_recon[:, -2, :].detach().cpu().numpy())

            recons = np.concatenate(recons, axis=0)
            actual = np.concatenate(actual, axis=0)
            a_score = np.sqrt((recons - actual) ** 2)
            anomaly_scores = np.mean(a_score, 1)
            print('anomaly_score', anomaly_scores)
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(anomaly_scores, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'MAD_GAN':
                    score_1 = anomaly_scores
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'MAD_GAN':
                    score_2 = anomaly_scores
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = anomaly_scores
                score_2 = 0
            # return anomaly_scores

        if 'USAD' in excute:

            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"USAD"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model5.load_state_dict(checkpoint['USAD'])
            model5.to(device)
            model5.eval()
            losses = []
            l = nn.MSELoss(reduction='none')
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data = input_data.to(device)
                y_input_data = y_input_data.to(device)
                recon_x = torch.cat((input_data[:, 2:, :], y_input_data), dim=1)
                ae1, ae2, ae2ae1 = model5(recon_x)
                input_data = input_data.view(input_data.shape[0], -1)
                loss = 0.1 * l(ae1, input_data) + 0.9 * l(ae2ae1, input_data)
                loss = torch.mean(loss, dim=1)
                losses.append(loss.detach().cpu().numpy())
            losses = np.concatenate(losses, axis=0)
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(losses, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'USAD':
                    score_1 = losses
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'USAD':
                    score_2 = losses
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = losses
                score_2 = 0
            # return losses
        if 'DAGMM' in excute:
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"DAGMM"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model6.load_state_dict(checkpoint['DAGMM'])
            model6.to(device)
            model6.eval()
            # print(model6.device)
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
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(anomaly_scores, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'DAGMM':
                    score_1 = anomaly_scores
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'DAGMM':
                    score_2 = anomaly_scores
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = anomaly_scores
                score_2 = 0
            # return anomaly_scores

        if 'MTAD' in excute:
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"MTAD"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model7.load_state_dict(checkpoint['MTAD'])
            model7.to(device)
            model7.eval()
            preds = []
            recons = []
            actual = []
            pre_actual = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data1 = input_data.to(device)
                y_input_data = y_input_data.to(device)
                # print(input_data1.shape)
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
            # print(pre_actual.shape)
            # print(preds.shape)
            # print(actual.shape)
            # print(recons.shape)
            a_score = np.mean(np.sqrt((preds - pre_actual) ** 2), 1) + np.mean(np.sqrt((recons - actual) ** 2), 1)
            # a_score = np.mean(np.sqrt((preds - actual) ** 2), 1)
            # print(a_score.shape)
            # anomaly_scores = np.mean(a_score, 1)
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(a_score , two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'MTAD':
                    score_1 = a_score
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'MTAD':
                    score_2 = a_score
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = a_score
                score_2 = 0
            # return anomaly_scores
        if 'Omni' in excute:
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"Omni"}/model.ckpt'
            # print(fname)
            checkpoint = torch.load(fname, map_location='cpu')
            model8.load_state_dict(checkpoint['Omni'])
            model8.to(device)
            model8.eval()
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
            print(actual.shape)
            a_score = np.sqrt((recons - actual) ** 2)
            anomaly_scores = np.mean(a_score, 1)
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(anomaly_scores, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'Omni':
                    score_1 = anomaly_scores
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'Omni':
                    score_2 = anomaly_scores
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = anomaly_scores
                score_2 = 0
            # return anomaly_scores
        if 'CAE_M' in excute:
            model9 = CAE_M(shape, self.win_size)
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"CAE_M"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model9.load_state_dict(checkpoint['CAE_M'])
            model9.to(device)
            model9.eval()
            recons = []
            actual = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data1 = input_data.to(device)
                y_input_data = y_input_data.to(device)
                recon_x = torch.cat((input_data1[:, 2:, :], y_input_data), dim=1)
                actual.append(recon_x[:, -2, :].detach().cpu().numpy())
                y_pred = model9(recon_x)
                # print(y_pred.shape)
                recons.append(y_pred[:, -2, :].detach().cpu().numpy())
            recons = np.concatenate(recons, axis=0)
            actual = np.concatenate(actual, axis=0)
            a_score = np.sqrt((recons - actual) ** 2)
            anomaly_scores = np.mean(a_score, 1)
            print('anomaly_score',len(anomaly_scores))
            print('label', len(label))
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(anomaly_scores, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'CAE_M':
                    score_1 = anomaly_scores
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'CAE_M':
                    score_2 = anomaly_scores
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = anomaly_scores
                score_2 = 0

        if 'MSCRED' in excute:
            model10 = MSCRED(shape, self.win_size)
            fname = f'{self.model_save_path}_{self.dataset}_{data_name}_{"MSCRED"}/model.ckpt'
            checkpoint = torch.load(fname, map_location='cpu')
            model10.load_state_dict(checkpoint['MSCRED'])
            model10.to(device)
            model10.eval()
            recons = []
            actual = []
            for i, (input_data, y_input_data) in enumerate(loader):
                input_data1 = input_data.to(device)
                y_input_data = y_input_data.to(device)
                recon_x = torch.cat((input_data1[:, 2:, :], y_input_data), dim=1)
                actual.append(recon_x[:, -2, :].detach().cpu().numpy())
                y_pred = model10(recon_x)
                # print(y_pred.shape)
                recons.append(y_pred[:, -2, :].detach().cpu().numpy())
            recons = np.concatenate(recons, axis=0)
            actual = np.concatenate(actual, axis=0)
            a_score = np.sqrt((recons - actual) ** 2)
            anomaly_scores = np.mean(a_score, 1)
            two_label1 = []
            for i in label:
                if i == values_1:
                    a = 1
                    two_label1.append(a)
                else:
                    a = 0
                    two_label1.append(a)
            two_label1 = np.array(two_label1)
            bf_eval_1 = bf_search(anomaly_scores, two_label1, start=0.001, end=1, step_num=150, verbose=False)
            print(data_name, excute, bf_eval_1)
            if len(excute) == 2:
                if excute[0] == 'MSCRED':
                    score_1 = anomaly_scores
                    thre_1 = bf_eval_1['threshold']
                if excute[1] == 'MSCRED':
                    score_2 = anomaly_scores
                    thre_2 = bf_eval_1['threshold']
            else:
                score_1 = anomaly_scores
                score_2 = 0

        return score_1, score_2, thre_1, thre_2
