import os, json, torch
from models.hifigan.env import AttrDict
from models.hifigan.models import Generator
MAX_WAV_VALUE = 32768.0

def vocoder(hifi_gan_path, hifi_gan_name):
    if False:
        print('Hello World!')
    device = torch.device('cpu')
    config_file = os.path.join(os.path.split(hifi_gan_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(hifi_gan_path + hifi_gan_name, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def vocoder2(config, hifi_gan_ckpt_path):
    if False:
        for i in range(10):
            print('nop')
    device = torch.device('cpu')
    global h
    generator = Generator(config.model).to(device)
    state_dict_g = torch.load(hifi_gan_ckpt_path, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def vocoder_inference(vocoder, melspec, max_db, min_db):
    if False:
        for i in range(10):
            print('nop')
    with torch.no_grad():
        x = melspec * (max_db - min_db) + min_db
        device = torch.device('cpu')
        x = torch.FloatTensor(x).to(device)
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze().numpy()
    return audio