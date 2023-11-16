import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from bigdl.chronos.model.tcmf.data_loader import TCMFDataLoader
from bigdl.chronos.model.tcmf.time import TimeCovariates
import logging
logger = logging.getLogger(__name__)

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        if False:
            return 10
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if False:
            return 10
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1, init=True):
        if False:
            print('Hello World!')
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if False:
            return 10
        if self.init:
            nn.init.normal_(self.conv1.weight, std=0.001)
            nn.init.normal_(self.conv2.weight, std=0.001)
            self.conv1.weight[:, 0, :] += 1.0 / self.kernel_size
            self.conv2.weight += 1.0 / self.kernel_size
            nn.init.normal_(self.conv1.bias, std=1e-06)
            nn.init.normal_(self.conv2.bias, std=1e-06)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalBlockLast(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, init=True):
        if False:
            i = 10
            return i + 15
        super(TemporalBlockLast, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1, self.conv2, self.chomp2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if False:
            for i in range(10):
                print('nop')
        if self.init:
            nn.init.normal_(self.conv1.weight, std=0.001)
            nn.init.normal_(self.conv2.weight, std=0.001)
            self.conv1.weight[:, 0, :] += 1.0 / self.kernel_size
            self.conv2.weight += 1.0 / self.kernel_size
            nn.init.normal_(self.conv1.bias, std=1e-06)
            nn.init.normal_(self.conv2.bias, std=1e-06)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):
        if False:
            while True:
                i = 10
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1, init=True):
        if False:
            return 10
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_channels = num_channels
        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.dropout = dropout
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                layers += [TemporalBlockLast(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout, init=init)]
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout, init=init)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.network(x)

class LocalModel(object):

    def __init__(self, Ymat, num_inputs=1, num_channels=[32, 32, 32, 32, 32, 1], kernel_size=7, dropout=0.2, vbsize=300, hbsize=128, lr=0.0005, val_len=10, test=True, end_index=120, normalize=False, start_date='2016-1-1', freq='H', covariates=None, use_time=False, dti=None, Ycov=None):
        if False:
            i = 10
            return i + 15
        '\n        Arguments:\n        Ymat: input time-series n*T\n        num_inputs: always set to 1\n        num_channels: list containing channel progression of temporal comvolution network\n        kernel_size: kernel size of temporal convolution filters\n        dropout: dropout rate for each layer\n        vbsize: vertical batch size\n        hbsize: horizontal batch size\n        lr: learning rate\n        val_len: validation length\n        test: always set to True\n        end_index: no data is touched fro training or validation beyond end_index\n        normalize: normalize dataset before training or not\n        start_data: start data in YYYY-MM-DD format (give a random date if unknown)\n        freq: "H" hourly, "D": daily and for rest see here:\n            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html\n            # timeseries-offset-aliases\n        covariates: global covariates common for all time series r*T,\n        where r is the number of covariates\n        Ycov: per time-series covariates n*l*T, l such covariates per time-series\n        use_time: if false, default trime-covriates are not used\n        dti: date time object can be explicitly supplied here, leave None if default options are\n            to be used\n        '
        self.start_date = start_date
        if use_time:
            self.time = TimeCovariates(start_date=start_date, freq=freq, normalized=True, num_ts=Ymat.shape[1])
            if dti is not None:
                self.time.dti = dti
            time_covariates = self.time.get_covariates()
            if covariates is None:
                self.covariates = time_covariates
            else:
                self.covariates = np.vstack([time_covariates, covariates])
        else:
            self.covariates = covariates
        self.Ycov = Ycov
        self.freq = freq
        self.vbsize = vbsize
        self.hbsize = hbsize
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.lr = lr
        self.val_len = val_len
        self.Ymat = Ymat
        self.end_index = end_index
        self.normalize = normalize
        self.kernel_size = kernel_size
        self.dropout = dropout
        if normalize:
            Y = Ymat
            m = np.mean(Y[:, 0:self.end_index], axis=1)
            s = np.std(Y[:, 0:self.end_index], axis=1)
            s += 1.0
            Y = (Y - m[:, None]) / s[:, None]
            mini = np.abs(np.min(Y))
            self.Ymat = Y + mini
            self.m = m
            self.s = s
            self.mini = mini
        if self.Ycov is not None:
            self.num_inputs += self.Ycov.shape[1]
        if self.covariates is not None:
            self.num_inputs += self.covariates.shape[0]
        self.seq = TemporalConvNet(num_inputs=self.num_inputs, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, init=True)
        self.seq = self.seq.float()
        self.D = TCMFDataLoader(Ymat=self.Ymat, vbsize=vbsize, hbsize=hbsize, end_index=end_index, val_len=val_len, covariates=self.covariates, Ycov=self.Ycov)
        self.val_len = val_len

    def train_model(self, num_epochs=300, num_workers=1, early_stop=False, tenacity=10):
        if False:
            return 10
        if num_workers == 1:
            return self.train_model_local(num_epochs=num_epochs, early_stop=early_stop, tenacity=tenacity)
        else:
            from bigdl.chronos.model.tcmf.local_model_distributed_trainer import train_yseq_hvd
            import ray
            from bigdl.orca.ray import OrcaRayContext
            ray_ctx = OrcaRayContext.get()
            Ymat_id = ray.put(self.Ymat)
            covariates_id = ray.put(self.covariates)
            Ycov_id = ray.put(self.Ycov)
            trainer_config_keys = ['vbsize', 'hbsize', 'end_index', 'val_len', 'lr', 'num_inputs', 'num_channels', 'kernel_size', 'dropout']
            trainer_config = {k: self.__dict__[k] for k in trainer_config_keys}
            (model, val_loss) = train_yseq_hvd(epochs=num_epochs, workers_per_node=num_workers // ray_ctx.num_ray_nodes, Ymat_id=Ymat_id, covariates_id=covariates_id, Ycov_id=Ycov_id, **trainer_config)
            self.seq = model
            return val_loss

    @staticmethod
    def loss(out, target):
        if False:
            while True:
                i = 10
        criterion = nn.L1Loss()
        return criterion(out, target) / torch.abs(target.data).mean()

    def train_model_local(self, num_epochs=300, early_stop=False, tenacity=10):
        if False:
            for i in range(10):
                print('nop')
        '\n        early_stop: set true for using early stop\n        tenacity: patience for early_stop\n        '
        print('Training Local Model(Tconv)')
        optimizer = optim.Adam(params=self.seq.parameters(), lr=self.lr)
        iter_count = 0
        loss_all = []
        min_val_loss = float('inf')
        scount = 0
        val_loss = 0
        (inp_test, out_target_test, _, _) = self.D.supply_test()
        while self.D.epoch < num_epochs:
            last_epoch = self.D.epoch
            (inp, out_target, _, _) = self.D.next_batch()
            current_epoch = self.D.epoch
            inp = Variable(inp)
            out_target = Variable(out_target)
            optimizer.zero_grad()
            out = self.seq(inp)
            loss = LocalModel.loss(out, out_target)
            iter_count = iter_count + 1
            for p in self.seq.parameters():
                p.requires_grad = True
            loss.backward()
            for p in self.seq.parameters():
                p.grad.data.clamp_(max=100000.0, min=-100000.0)
            optimizer.step()
            loss_all = loss_all + [loss.item()]
            if current_epoch > last_epoch:
                inp_test = Variable(inp_test)
                out_target_test = Variable(out_target_test)
                out_test = self.seq(inp_test)
                val_loss = LocalModel.loss(out_test, out_target_test).item()
                print('Entering Epoch:{}'.format(current_epoch))
                print('Train Loss:{}'.format(np.mean(loss_all)))
                print('Validation Loss:{}'.format(val_loss))
                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    scount = 0
                    self.saved_seq = pickle.loads(pickle.dumps(self.seq))
                else:
                    scount += 1
                    if scount > tenacity and early_stop:
                        self.seq = self.saved_seq
                        break
        return val_loss

    @staticmethod
    def convert_to_input(data):
        if False:
            while True:
                i = 10
        (n, m) = data.shape
        inp = torch.from_numpy(data).view(1, n, m)
        inp = inp.transpose(0, 1).float()
        return inp

    @staticmethod
    def convert_covariates(data, covs):
        if False:
            print('Hello World!')
        (nd, td) = data.shape
        rcovs = np.repeat(covs.reshape(1, covs.shape[0], covs.shape[1]), repeats=nd, axis=0)
        rcovs = torch.from_numpy(rcovs).float()
        return rcovs

    @staticmethod
    def convert_ycovs(data, ycovs):
        if False:
            print('Hello World!')
        ycovs = torch.from_numpy(ycovs).float()
        return ycovs

    @staticmethod
    def convert_from_output(T):
        if False:
            for i in range(10):
                print('nop')
        out = T.view(T.size(0), T.size(2))
        return np.array(out.detach())

    @staticmethod
    def predict_future_batch(data, covariates=None, ycovs=None, future=10, model=None):
        if False:
            print('Hello World!')
        valid_cov = covariates is not None
        inp = LocalModel.convert_to_input(data)
        if valid_cov:
            cov = LocalModel.convert_covariates(data, covariates)
            inp = torch.cat((inp, cov[:, :, 0:inp.size(2)]), 1)
        if ycovs is not None:
            ycovs = LocalModel.convert_ycovs(data, ycovs)
            inp = torch.cat((inp, ycovs[:, :, 0:inp.size(2)]), 1)
        ci = inp.size(2)
        for i in range(future):
            out = model(inp)
            output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
            if valid_cov:
                output = torch.cat((output, cov[:, :, ci].view(cov.size(0), cov.size(1), 1)), 1)
            if ycovs is not None:
                output = torch.cat((output, ycovs[:, :, ci].view(ycovs.size(0), ycovs.size(1), 1)), 1)
            out = torch.cat((inp, output), dim=2)
            inp = out
            ci += 1
        out = out[:, 0, :].view(out.size(0), 1, out.size(2))
        y = LocalModel.convert_from_output(out)
        return y

    @staticmethod
    def _predict_future(data, ycovs, covariates, model, future, I):
        if False:
            return 10
        out = None
        for i in range(len(I) - 1):
            bdata = data[range(I[i], I[i + 1]), :]
            batch_ycovs = ycovs[range(I[i], I[i + 1]), :, :] if ycovs is not None else None
            cur_out = LocalModel.predict_future_batch(bdata, covariates, batch_ycovs, future, model)
            out = np.vstack([out, cur_out]) if out is not None else cur_out
        return out

    def predict_future(self, data_in, covariates=None, ycovs=None, future=10, bsize=40, normalize=False, num_workers=1):
        if False:
            return 10
        '\n        data_in: input past data in same format of Ymat\n        covariates: input past covariates\n        ycovs: input past individual covariates\n        future: number of time-points to predict\n        bsize: batch size for processing (determine according to gopu memory limits)\n        normalize: should be set according to the normalization used in the class initialization\n        num_workers: number of workers to run prediction. if num_workers > 1, then prediction will\n        run in distributed mode and there has to be an activate OrcaRayContext.\n        '
        with torch.no_grad():
            if normalize:
                data = (data_in - self.m[:, None]) / self.s[:, None]
                data += self.mini
            else:
                data = data_in
            (n, T) = data.shape
            I = list(np.arange(0, n, bsize))
            I.append(n)
            model = self.seq
            if num_workers > 1:
                import ray
                import math
                batch_num_per_worker = math.ceil(len(I) / num_workers)
                indexes = [I[i:i + batch_num_per_worker + 1] for i in range(0, len(I) - 1, batch_num_per_worker)]
                logger.info(f'actual number of workers used in prediction is {len(indexes)}')
                data_id = ray.put(data)
                covariates_id = ray.put(covariates)
                ycovs_id = ray.put(ycovs)
                model_id = ray.put(model)

                @ray.remote
                def predict_future_worker(I):
                    if False:
                        i = 10
                        return i + 15
                    data = ray.get(data_id)
                    covariates = ray.get(covariates_id)
                    ycovs = ray.get(ycovs_id)
                    model = ray.get(model_id)
                    out = LocalModel._predict_future(data, ycovs, covariates, model, future, I)
                    return out
                remote_out = ray.get([predict_future_worker.remote(index) for index in indexes])
                out = np.concatenate(remote_out, axis=0)
            else:
                out = LocalModel._predict_future(data, ycovs, covariates, model, future, I)
            if normalize:
                temp = (out - self.mini) * self.s[:, None] + self.m[:, None]
                out = temp
        return out

    def rolling_validation(self, Ymat, tau=24, n=7, bsize=90, alpha=0.3):
        if False:
            print('Hello World!')
        last_step = Ymat.shape[1] - tau * n
        rg = 1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels) - 1)
        self.seq = self.seq.eval()
        if self.covariates is not None:
            covs = self.covariates[:, last_step - rg:last_step + tau]
        else:
            covs = None
        if self.Ycov is not None:
            ycovs = self.Ycov[:, :, last_step - rg:last_step + tau]
        else:
            ycovs = None
        data_in = Ymat[:, last_step - rg:last_step]
        out = self.predict_future(data_in, covariates=covs, ycovs=ycovs, future=tau, bsize=bsize, normalize=self.normalize)
        predicted_values = []
        actual_values = []
        S = out[:, -tau:]
        predicted_values += [S]
        R = Ymat[:, last_step:last_step + tau]
        actual_values += [R]
        print('Current window wape:{}'.format(wape(S, R)))
        for i in range(n - 1):
            last_step += tau
            rg = 1 + 2 * (self.kernel_size - 1) * 2 ** (len(self.num_channels) - 1)
            if self.covariates is not None:
                covs = self.covariates[:, last_step - rg:last_step + tau]
            else:
                covs = None
            if self.Ycov is not None:
                ycovs = self.Ycov[:, :, last_step - rg:last_step + tau]
            else:
                ycovs = None
            data_in = Ymat[:, last_step - rg:last_step]
            out = self.predict_future(data_in, covariates=covs, ycovs=ycovs, future=tau, bsize=bsize, normalize=self.normalize)
            S = out[:, -tau:]
            predicted_values += [S]
            R = Ymat[:, last_step:last_step + tau]
            actual_values += [R]
            print('Current window wape:{}'.format(wape(S, R)))
        predicted = np.hstack(predicted_values)
        actual = np.hstack(actual_values)
        dic = {}
        dic['wape'] = wape(predicted, actual)
        dic['mape'] = mape(predicted, actual)
        dic['smape'] = smape(predicted, actual)
        dic['mae'] = np.abs(predicted - actual).mean()
        dic['rmse'] = np.sqrt(((predicted - actual) ** 2).mean())
        dic['nrmse'] = dic['rmse'] / np.sqrt((actual ** 2).mean())
        baseline = Ymat[:, Ymat.shape[1] - n * tau - tau:Ymat.shape[1] - tau]
        dic['baseline_wape'] = wape(baseline, actual)
        dic['baseline_mape'] = mape(baseline, actual)
        dic['baseline_smape'] = smape(baseline, actual)
        return dic