from dataloader import *
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from model import Model
from launcher import run
from net import *
import numpy as np
import tensorflow as tf
import principal_directions.classifier

def parse_tfrecord_np(record):
    if False:
        return 10
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    dlat = ex.features.feature['dlat'].bytes_list.value[0]
    lat = ex.features.feature['lat'].bytes_list.value[0]
    return (np.fromstring(data, np.uint8).reshape(shape), np.fromstring(dlat, np.float32), np.fromstring(lat, np.float32))

class Predictions:

    def __init__(self, cfg, minibatch_gpu):
        if False:
            i = 10
            return i + 15
        self.minibatch_size = minibatch_gpu
        self.cfg = cfg

    def evaluate(self, logger, mapping, decoder, lod, attrib_idx):
        if False:
            for i in range(10):
                print('nop')
        result_expr = []
        rnd = np.random.RandomState(5)
        with tf.Graph().as_default(), tf.Session() as sess:
            ds = tf.data.TFRecordDataset('principal_directions/generated_data.000')
            ds = ds.batch(self.minibatch_size)
            batch = ds.make_one_shot_iterator().get_next()
            classifier = principal_directions.classifier.make_classifier(attrib_idx)
            i = 0
            while True:
                try:
                    records = sess.run(batch)
                    images = []
                    dlats = []
                    lats = []
                    for r in records:
                        (im, dlat, lat) = parse_tfrecord_np(r)
                        images.append(im)
                        dlats.append(dlat)
                        lats.append(lat)
                    images = np.stack(images)
                    dlats = np.stack(dlats)
                    lats = np.stack(lats)
                    logits = classifier.run(images, None, num_gpus=1, assume_frozen=True)
                    logits = torch.tensor(logits)
                    predictions = torch.softmax(torch.cat([logits, -logits], dim=1), dim=1)
                    result_dict = dict(latents=lats, dlatents=dlats)
                    result_dict[attrib_idx] = predictions.cpu().numpy()
                    result_expr.append(result_dict)
                    i += 1
                except tf.errors.OutOfRangeError:
                    break
        results = {key: np.concatenate([value[key] for value in result_expr], axis=0) for key in result_expr[0].keys()}
        np.save('principal_directions/wspace_att_%d' % attrib_idx, results)

def main(cfg, logger):
    if False:
        print('Hello World!')
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
    logger.info('Extracting attributes')
    decoder = nn.DataParallel(decoder)
    indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]
    with torch.no_grad():
        p = Predictions(cfg, minibatch_gpu=4)
        for i in indices:
            p.evaluate(logger, mapping_fl, decoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2, i)
if __name__ == '__main__':
    gpu_count = 1
    run(main, get_cfg_defaults(), description='StyleGAN', default_config='configs/ffhq.yaml', world_size=gpu_count, write_log=False)