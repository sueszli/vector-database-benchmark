import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from text_loader import TextDataset
import seq2seq_models as sm
from seq2seq_models import str2tensor, EOS_token, SOS_token
HIDDEN_SIZE = 100
N_LAYERS = 1
BATCH_SIZE = 1
N_EPOCH = 100
N_CHARS = 128

def test():
    if False:
        for i in range(10):
            print('nop')
    encoder_hidden = encoder.init_hidden()
    word_input = str2tensor('hello')
    (encoder_outputs, encoder_hidden) = encoder(word_input, encoder_hidden)
    print(encoder_outputs)
    decoder_hidden = encoder_hidden
    word_target = str2tensor('pytorch')
    for c in range(len(word_target)):
        (decoder_output, decoder_hidden) = decoder(word_target[c], decoder_hidden)
        print(decoder_output.size(), decoder_hidden.size())

def train(src, target):
    if False:
        print('Hello World!')
    src_var = str2tensor(src)
    target_var = str2tensor(target, eos=True)
    encoder_hidden = encoder.init_hidden()
    (encoder_outputs, encoder_hidden) = encoder(src_var, encoder_hidden)
    hidden = encoder_hidden
    loss = 0
    for c in range(len(target_var)):
        token = target_var[c - 1] if c else str2tensor(SOS_token)
        (output, hidden) = decoder(token, hidden)
        loss += criterion(output, target_var[c])
    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.data[0] / len(target_var)

def translate(enc_input='thisissungkim.iloveyou.', predict_len=100, temperature=0.9):
    if False:
        for i in range(10):
            print('nop')
    input_var = str2tensor(enc_input)
    encoder_hidden = encoder.init_hidden()
    (encoder_outputs, encoder_hidden) = encoder(input_var, encoder_hidden)
    hidden = encoder_hidden
    predicted = ''
    dec_input = str2tensor(SOS_token)
    for c in range(predict_len):
        (output, hidden) = decoder(dec_input, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        if top_i is EOS_token:
            break
        predicted_char = chr(top_i)
        predicted += predicted_char
        dec_input = str2tensor(predicted_char)
    return (enc_input, predicted)
encoder = sm.EncoderRNN(N_CHARS, HIDDEN_SIZE, N_LAYERS)
decoder = sm.DecoderRNN(HIDDEN_SIZE, N_CHARS, N_LAYERS)
if torch.cuda.is_available():
    decoder.cuda()
    encoder.cuda()
print(encoder, decoder)
test()
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(dataset=TextDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print('Training for %d epochs...' % N_EPOCH)
for epoch in range(1, N_EPOCH + 1):
    for (i, (srcs, targets)) in enumerate(train_loader):
        train_loss = train(srcs[0], targets[0])
        if i % 100 is 0:
            print('[(%d %d%%) %.4f]' % (epoch, epoch / N_EPOCH * 100, train_loss))
            print(translate(srcs[0]), '\n')
            print(translate(), '\n')