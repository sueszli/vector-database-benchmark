"""Perceptual Path Length (PPL)."""
import numpy as np
import dnnlib.tflib
import pickle
from net import *
from checkpointer import Checkpointer
from model import Model
from launcher import run
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from dlutils import download
import tqdm
from matplotlib import pyplot as plt
dnnlib.tflib.init_tf()
download.from_google_drive('1CIDc9i070KQhHlkr4yIwoJC8xqrwjE0_', directory='metrics')

def normalize(v):
    if False:
        i = 10
        return i + 15
    return v / torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True))

def slerp(a, b, t):
    if False:
        for i in range(10):
            print('nop')
    a = normalize(a)
    b = normalize(b)
    d = torch.sum(a * b, dim=-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)
    return normalize(d)

class PPL:

    def __init__(self, cfg, num_samples, epsilon, space, sampling, minibatch_size, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert space in ['z', 'w']
        assert sampling in ['full', 'end']
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.minibatch_size = minibatch_size
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, lod, celeba_style=False):
        if False:
            print('Hello World!')
        distance_measure = pickle.load(open('metrics/vgg16_zhang_perceptual.pkl', 'rb'))
        gpu_count = torch.cuda.device_count()
        all_distances = []
        for _ in tqdm.tqdm(range(0, self.num_samples, self.minibatch_size)):
            torch.cuda.set_device(0)
            lat_t01 = torch.randn([self.minibatch_size * 2, self.cfg.MODEL.LATENT_SPACE_SIZE])
            lerp_t = torch.rand(self.minibatch_size) * (1.0 if self.sampling == 'full' else 0.0)
            if self.space == 'w':
                dlat_t01 = mapping(lat_t01)
                (dlat_t0, dlat_t1) = (dlat_t01[0::2], dlat_t01[1::2])
                dlat_e0 = torch.lerp(dlat_t0, dlat_t1, lerp_t[:, np.newaxis, np.newaxis])
                dlat_e1 = torch.lerp(dlat_t0, dlat_t1, lerp_t[:, np.newaxis, np.newaxis] + self.epsilon)
                dlat_e01 = torch.reshape(torch.stack([dlat_e0, dlat_e1], dim=1), dlat_t01.shape)
            else:
                (lat_t0, lat_t1) = (lat_t01[0::2], lat_t01[1::2])
                lat_e0 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis])
                lat_e1 = slerp(lat_t0, lat_t1, lerp_t[:, np.newaxis] + self.epsilon)
                lat_e01 = torch.reshape(torch.stack([lat_e0, lat_e1], dim=1), lat_t01.shape)
                dlat_e01 = mapping(lat_e01)
            images = decoder(dlat_e01, lod, 1.0, noise='batch_constant')
            if not celeba_style:
                c = int(images.shape[2] // 8)
                images = images[:, :, c * 3:c * 7, c * 2:c * 6]
            else:
                c = int(images.shape[2])
                h = (7.0 - 3.0) / 8.0 * (2.0 / 1.641)
                w = (6.0 - 2.0) / 8.0 * (2.0 / 1.641)
                vc = (7.0 + 3.0) / 2.0 / 8.0
                hc = (6.0 + 2.0) / 2.0 / 8.0
                h = int(h * c)
                w = int(w * c)
                hc = int(hc * c)
                vc = int(vc * c)
                images = images[:, :, vc - h // 2:vc + h // 2, hc - w // 2:hc + w // 2]
            if images.shape[2] > 256:
                factor = images.shape[2] // 256
                images = torch.reshape(images, [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
                images = torch.mean(images, dim=(3, 5))
            images = (images + 1) * (255 / 2)
            (img_e0, img_e1) = (images[0::2], images[1::2])
            res = distance_measure.run(img_e0.cpu().numpy(), img_e1.cpu().numpy(), num_gpus=gpu_count, assume_frozen=True) * (1 / self.epsilon ** 2)
            all_distances.append(res)
        all_distances = np.concatenate(all_distances, axis=0)
        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)
        logger.info('Result %s = %f' % (self.sampling, np.mean(filtered_distances)))

def sample(cfg, logger):
    if False:
        return 10
    torch.cuda.set_device(0)
    model = Model(startf=cfg.MODEL.START_CHANNEL_COUNT, layer_count=cfg.MODEL.LAYER_COUNT, maxf=cfg.MODEL.MAX_CHANNEL_COUNT, latent_size=cfg.MODEL.LATENT_SPACE_SIZE, truncation_psi=cfg.MODEL.TRUNCATIOM_PSI, truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF, mapping_layers=cfg.MODEL.MAPPING_LAYERS, channels=cfg.MODEL.CHANNELS, generator=cfg.MODEL.GENERATOR, encoder=cfg.MODEL.ENCODER)
    model.cuda(0)
    model.eval()
    model.requires_grad_(False)
    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_d
    mapping_fl = model.mapping_f
    dlatent_avg = model.dlatent_avg
    logger.info('Trainable parameters generator:')
    count_parameters(decoder)
    logger.info('Trainable parameters discriminator:')
    count_parameters(encoder)
    arguments = dict()
    arguments['iteration'] = 0
    model_dict = {'discriminator_s': encoder, 'generator_s': decoder, 'mapping_tl_s': mapping_tl, 'mapping_fl_s': mapping_fl, 'dlatent_avg': dlatent_avg}
    checkpointer = Checkpointer(cfg, model_dict, {}, logger=logger, save=False)
    checkpointer.load()
    model.eval()
    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        if False:
            while True:
                i = 10
        (Z, _) = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_f.num_layers, 1)
        return Z

    def decode(x):
        if False:
            while True:
                i = 10
        layer_idx = torch.arange(2 * cfg.MODEL.LAYER_COUNT)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, 1.2 * ones, ones)
        x = torch.lerp(model.dlatent_avg.buff.data, x, coefs)
        return model.decoder(x, layer_count - 1, 1, noise=True)
    logger.info('Evaluating PPL metric')
    decoder = nn.DataParallel(decoder)
    with torch.no_grad():
        ppl = PPL(cfg, num_samples=50000, epsilon=0.0001, space='w', sampling='full', minibatch_size=16 * torch.cuda.device_count())
        ppl.evaluate(logger, mapping_fl, decoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2, celeba_style=cfg.PPL_CELEBA_ADJUSTMENT)
    with torch.no_grad():
        ppl = PPL(cfg, num_samples=50000, epsilon=0.0001, space='w', sampling='end', minibatch_size=16 * torch.cuda.device_count())
        ppl.evaluate(logger, mapping_fl, decoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2, celeba_style=cfg.PPL_CELEBA_ADJUSTMENT)
if __name__ == '__main__':
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-ppl', default_config='configs/ffhq.yaml', world_size=gpu_count, write_log=False)