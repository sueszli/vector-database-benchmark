import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from text_loader import TextDataset
import seq2seq_models as sm
from seq2seq_models import cuda_variable, str2tensor, EOS_token, SOS_token
N_LAYERS = 1
BATCH_SIZE = 1
N_EPOCH = 100
N_CHARS = 128
HIDDEN_SIZE = N_CHARS

def test():
    if False:
        return 10
    encoder_test = sm.EncoderRNN(10, 10, 2)
    decoder_test = sm.AttnDecoderRNN(10, 10, 2)
    if torch.cuda.is_available():
        encoder_test.cuda()
        decoder_test.cuda()
    encoder_hidden = encoder_test.init_hidden()
    word_input = cuda_variable(torch.LongTensor([1, 2, 3]))
    (encoder_outputs, encoder_hidden) = encoder_test(word_input, encoder_hidden)
    print(encoder_outputs.size())
    word_target = cuda_variable(torch.LongTensor([1, 2, 3]))
    decoder_attns = torch.zeros(1, 3, 3)
    decoder_hidden = encoder_hidden
    for c in range(len(word_target)):
        (decoder_output, decoder_hidden, decoder_attn) = decoder_test(word_target[c], decoder_hidden, encoder_outputs)
        print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
        decoder_attns[0, c] = decoder_attn.squeeze(0).cpu().data

def train(src, target):
    if False:
        for i in range(10):
            print('nop')
    loss = 0
    src_var = str2tensor(src)
    target_var = str2tensor(target, eos=True)
    encoder_hidden = encoder.init_hidden()
    (encoder_outputs, encoder_hidden) = encoder(src_var, encoder_hidden)
    hidden = encoder_hidden
    for c in range(len(target_var)):
        token = target_var[c - 1] if c else str2tensor(SOS_token)
        (output, hidden, attention) = decoder(token, hidden, encoder_outputs)
        loss += criterion(output, target_var[c])
    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.data[0] / len(target_var)

def translate(enc_input='thisissungkim.iloveyou.', predict_len=100, temperature=0.9):
    if False:
        return 10
    input_var = str2tensor(enc_input)
    encoder_hidden = encoder.init_hidden()
    (encoder_outputs, encoder_hidden) = encoder(input_var, encoder_hidden)
    hidden = encoder_hidden
    predicted = ''
    dec_input = str2tensor(SOS_token)
    attentions = []
    for c in range(predict_len):
        (output, hidden, attention) = decoder(dec_input, hidden, encoder_outputs)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        attentions.append(attention.view(-1).data.cpu().numpy().tolist())
        if top_i is EOS_token:
            break
        predicted_char = chr(top_i)
        predicted += predicted_char
        dec_input = str2tensor(predicted_char)
    return (predicted, attentions)
if __name__ == '__main__':
    encoder = sm.EncoderRNN(N_CHARS, HIDDEN_SIZE, N_LAYERS)
    decoder = sm.AttnDecoderRNN(HIDDEN_SIZE, N_CHARS, N_LAYERS)
    if torch.cuda.is_available():
        decoder.cuda()
        encoder.cuda()
    print(encoder, decoder)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=TextDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print('Training for %d epochs...' % N_EPOCH)
    for epoch in range(1, N_EPOCH + 1):
        for (i, (srcs, targets)) in enumerate(train_loader):
            train_loss = train(srcs[0], targets[0])
            if i % 1000 is 0:
                print('[(%d/%d %d%%) %.4f]' % (epoch, N_EPOCH, i * len(srcs) * 100 / len(train_loader), train_loss))
                (output, _) = translate(srcs[0])
                print(srcs[0], output, '\n')
                (output, attentions) = translate()
                print('thisissungkim.iloveyou.', output, '\n')