import argparse
import torch
import torch.nn as nn
from utils.custom_mlp import MLP, Exp
from utils.mnist_cached import MNISTCached, mkdir_p, setup_data_loaders
from utils.vae_plots import mnist_test_tsne_ssvae, plot_conditional_samples_ssvae
from visdom import Visdom
import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam

class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset

    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """

    def __init__(self, output_size=10, input_size=784, z_dim=50, hidden_layers=(500,), config_enum=None, use_cuda=False, aux_loss_multiplier=None):
        if False:
            return 10
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.setup_networks()

    def setup_networks(self):
        if False:
            i = 10
            return i + 15
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers
        self.encoder_y = MLP([self.input_size] + hidden_sizes + [self.output_size], activation=nn.Softplus, output_activation=nn.Softmax, allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        self.encoder_z = MLP([self.input_size + self.output_size] + hidden_sizes + [[z_dim, z_dim]], activation=nn.Softplus, output_activation=[None, Exp], allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        self.decoder = MLP([z_dim + self.output_size] + hidden_sizes + [self.input_size], activation=nn.Softplus, output_activation=nn.Sigmoid, allow_broadcast=self.allow_broadcast, use_cuda=self.use_cuda)
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
        if False:
            print('Hello World!')
        '\n        The model corresponds to the following generative process:\n        p(z) = normal(0,I)              # handwriting style (latent)\n        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)\n        p(x|y,z) = bernoulli(loc(y,z))   # an image\n        loc is given by a neural network  `decoder`\n\n        :param xs: a batch of scaled vectors of pixels from an image\n        :param ys: (optional) a batch of the class labels i.e.\n                   the digit corresponding to the image(s)\n        :return: None\n        '
        pyro.module('ss_vae', self)
        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate('data'):
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))
            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (1.0 * self.output_size)
            ys = pyro.sample('y', dist.OneHotCategorical(alpha_prior), obs=ys)
            loc = self.decoder([zs, ys])
            pyro.sample('x', dist.Bernoulli(loc, validate_args=False).to_event(1), obs=xs)
            return loc

    def guide(self, xs, ys=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        The guide corresponds to the following:\n        q(y|x) = categorical(alpha(x))              # infer digit from an image\n        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer handwriting style from an image and the digit\n        loc, scale are given by a neural network `encoder_z`\n        alpha is given by a neural network `encoder_y`\n\n        :param xs: a batch of scaled vectors of pixels from an image\n        :param ys: (optional) a batch of the class labels i.e.\n                   the digit corresponding to the image(s)\n        :return: None\n        '
        with pyro.plate('data'):
            if ys is None:
                alpha = self.encoder_y(xs)
                ys = pyro.sample('y', dist.OneHotCategorical(alpha))
            (loc, scale) = self.encoder_z([xs, ys])
            pyro.sample('z', dist.Normal(loc, scale).to_event(1))

    def classifier(self, xs):
        if False:
            while True:
                i = 10
        '\n        classify an image (or a batch of images)\n\n        :param xs: a batch of scaled vectors of pixels from an image\n        :return: a batch of the corresponding class labels (as one-hots)\n        '
        alpha = self.encoder_y(xs)
        (res, ind) = torch.topk(alpha, 1)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        if False:
            print('Hello World!')
        '\n        this model is used to add an auxiliary (supervised) loss as described in the\n        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".\n        '
        pyro.module('ss_vae', self)
        with pyro.plate('data'):
            if ys is not None:
                alpha = self.encoder_y(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample('y_aux', dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        if False:
            print('Hello World!')
        '\n        dummy guide function to accompany model_classify in inference\n        '
        pass

def run_inference_for_epoch(data_loaders, losses, periodic_interval_batches):
    if False:
        print('Hello World!')
    '\n    runs the inference algorithm for an epoch\n    returns the values of all losses separately on supervised and unsupervised parts\n    '
    num_losses = len(losses)
    sup_batches = len(data_loaders['sup'])
    unsup_batches = len(data_loaders['unsup'])
    batches_per_epoch = sup_batches + unsup_batches
    epoch_losses_sup = [0.0] * num_losses
    epoch_losses_unsup = [0.0] * num_losses
    sup_iter = iter(data_loaders['sup'])
    unsup_iter = iter(data_loaders['unsup'])
    ctr_sup = 0
    for i in range(batches_per_epoch):
        is_supervised = i % periodic_interval_batches == 1 and ctr_sup < sup_batches
        if is_supervised:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1
        else:
            (xs, ys) = next(unsup_iter)
        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].step(xs, ys)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(xs)
                epoch_losses_unsup[loss_id] += new_loss
    return (epoch_losses_sup, epoch_losses_unsup)

def get_accuracy(data_loader, classifier_fn, batch_size):
    if False:
        return 10
    '\n    compute the accuracy over the supervised training set or the testing set\n    '
    (predictions, actuals) = ([], [])
    for (xs, ys) in data_loader:
        predictions.append(classifier_fn(xs))
        actuals.append(ys)
    accurate_preds = 0
    for (pred, act) in zip(predictions, actuals):
        for i in range(pred.size(0)):
            v = torch.sum(pred[i] == act[i])
            accurate_preds += v.item() == 10
    accuracy = accurate_preds * 1.0 / (len(predictions) * batch_size)
    return accuracy

def visualize(ss_vae, viz, test_loader):
    if False:
        return 10
    if viz:
        plot_conditional_samples_ssvae(ss_vae, viz)
        mnist_test_tsne_ssvae(ssvae=ss_vae, test_loader=test_loader)

def main(args):
    if False:
        return 10
    '\n    run inference for SS-VAE\n    :param args: arguments for SS-VAE\n    :return: None\n    '
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)
    viz = None
    if args.visualize:
        viz = Visdom()
        mkdir_p('./vae_results')
    ss_vae = SSVAE(z_dim=args.z_dim, hidden_layers=args.hidden_layers, use_cuda=args.cuda, config_enum=args.enum_discrete, aux_loss_multiplier=args.aux_loss_multiplier)
    adam_params = {'lr': args.learning_rate, 'betas': (args.beta_1, 0.999)}
    optimizer = Adam(adam_params)
    guide = config_enumerate(ss_vae.guide, args.enum_discrete, expand=True)
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=1, strict_enumeration_warning=False)
    loss_basic = SVI(ss_vae.model, guide, optimizer, loss=elbo)
    losses = [loss_basic]
    if args.aux_loss:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        loss_aux = SVI(ss_vae.model_classify, ss_vae.guide_classify, optimizer, loss=elbo)
        losses.append(loss_aux)
    try:
        logger = open(args.logfile, 'w') if args.logfile else None
        data_loaders = setup_data_loaders(MNISTCached, args.cuda, args.batch_size, sup_num=args.sup_num)
        periodic_interval_batches = int(MNISTCached.train_data_size / (1.0 * args.sup_num))
        unsup_num = MNISTCached.train_data_size - args.sup_num
        (best_valid_acc, corresponding_test_acc) = (0.0, 0.0)
        for i in range(0, args.num_epochs):
            (epoch_losses_sup, epoch_losses_unsup) = run_inference_for_epoch(data_loaders, losses, periodic_interval_batches)
            avg_epoch_losses_sup = map(lambda v: v / args.sup_num, epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)
            str_loss_sup = ' '.join(map(str, avg_epoch_losses_sup))
            str_loss_unsup = ' '.join(map(str, avg_epoch_losses_unsup))
            str_print = '{} epoch: avg losses {}'.format(i, '{} {}'.format(str_loss_sup, str_loss_unsup))
            validation_accuracy = get_accuracy(data_loaders['valid'], ss_vae.classifier, args.batch_size)
            str_print += ' validation accuracy {}'.format(validation_accuracy)
            test_accuracy = get_accuracy(data_loaders['test'], ss_vae.classifier, args.batch_size)
            str_print += ' test accuracy {}'.format(test_accuracy)
            if best_valid_acc < validation_accuracy:
                best_valid_acc = validation_accuracy
                corresponding_test_acc = test_accuracy
            print_and_log(logger, str_print)
        final_test_accuracy = get_accuracy(data_loaders['test'], ss_vae.classifier, args.batch_size)
        print_and_log(logger, 'best validation accuracy {} corresponding testing accuracy {} last testing accuracy {}'.format(best_valid_acc, corresponding_test_acc, final_test_accuracy))
        visualize(ss_vae, viz, data_loaders['test'])
    finally:
        if args.logfile:
            logger.close()
EXAMPLE_RUN = 'example run: python ss_vae_M2.py --seed 0 --cuda -n 2 --aux-loss -alm 46 -enum parallel -sup 3000 -zd 50 -hl 500 -lr 0.00042 -b1 0.95 -bs 200 -log ./tmp.log'
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='SS-VAE\n{}'.format(EXAMPLE_RUN))
    parser.add_argument('--cuda', action='store_true', help='use GPU(s) to speed up training')
    parser.add_argument('--jit', action='store_true', help='use PyTorch jit to speed up training')
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of epochs to run')
    parser.add_argument('--aux-loss', action='store_true', help='whether to use the auxiliary loss from NIPS 14 paper (Kingma et al.). It is not used by default ')
    parser.add_argument('-alm', '--aux-loss-multiplier', default=46, type=float, help='the multiplier to use with the auxiliary loss')
    parser.add_argument('-enum', '--enum-discrete', default='parallel', help='parallel, sequential or none. uses parallel enumeration by default')
    parser.add_argument('-sup', '--sup-num', default=3000, type=float, help='supervised amount of the data i.e. how many of the images have supervised labels')
    parser.add_argument('-zd', '--z-dim', default=50, type=int, help='size of the tensor representing the latent variable z variable (handwriting style for our MNIST dataset)')
    parser.add_argument('-hl', '--hidden-layers', nargs='+', default=[500], type=int, help='a tuple (or list) of MLP layers to be used in the neural networks representing the parameters of the distributions in our model')
    parser.add_argument('-lr', '--learning-rate', default=0.00042, type=float, help='learning rate for Adam optimizer')
    parser.add_argument('-b1', '--beta-1', default=0.9, type=float, help='beta-1 parameter for Adam optimizer')
    parser.add_argument('-bs', '--batch-size', default=200, type=int, help='number of images (and labels) to be considered in a batch')
    parser.add_argument('-log', '--logfile', default='./tmp.log', type=str, help='filename for logging the outputs')
    parser.add_argument('--seed', default=None, type=int, help='seed for controlling randomness in this example')
    parser.add_argument('--visualize', action='store_true', help='use a visdom server to visualize the embeddings')
    args = parser.parse_args()
    assert args.sup_num % args.batch_size == 0, 'assuming simplicity of batching math'
    assert MNISTCached.validation_size % args.batch_size == 0, 'batch size should divide the number of validation examples'
    assert MNISTCached.train_data_size % args.batch_size == 0, "batch size doesn't divide total number of training data examples"
    assert MNISTCached.test_size % args.batch_size == 0, 'batch size should divide the number of test examples'
    main(args)