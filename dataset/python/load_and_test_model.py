import utils.py

folder = 'C:MLP_Models/4L_M3'
for filename in os.listdir(folder):
    mlp_model = load(os.path.join(folder, filename))