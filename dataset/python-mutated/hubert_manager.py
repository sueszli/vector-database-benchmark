import os.path
import shutil
import urllib.request
import huggingface_hub

class HubertManager:

    @staticmethod
    def make_sure_hubert_installed(download_url: str='https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt', model_path: str=''):
        if False:
            while True:
                i = 10
        if not os.path.isfile(model_path):
            print('Downloading HuBERT base model')
            urllib.request.urlretrieve(download_url, model_path)
            print('Downloaded HuBERT')
            return model_path
        return None

    @staticmethod
    def make_sure_tokenizer_installed(model: str='quantifier_hubert_base_ls960_14.pth', repo: str='GitMylo/bark-voice-cloning', model_path: str=''):
        if False:
            for i in range(10):
                print('nop')
        model_dir = os.path.dirname(model_path)
        if not os.path.isfile(model_path):
            print('Downloading HuBERT custom tokenizer')
            huggingface_hub.hf_hub_download(repo, model, local_dir=model_dir, local_dir_use_symlinks=False)
            shutil.move(os.path.join(model_dir, model), model_path)
            print('Downloaded tokenizer')
            return model_path
        return None