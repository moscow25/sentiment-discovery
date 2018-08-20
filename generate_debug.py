###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import os
import math

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from apex.reparameterization import apply_weight_norm, remove_weight_norm

import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# To fit solutions
from scipy.optimize import minimize
#import seaborn as sns
#sns.set_style({'font.family': 'monospace'})

import beam_search

parser = argparse.ArgumentParser(description='PyTorch Sentiment Discovery Generation/Visualization')

# Model parameters.
parser.add_argument('--model', type=str, default='mLSTM',
                    help='type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM, GRU')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=4096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--all_layers', action='store_true',
                    help='if more than one layer is used, extract features from all layers, not just the last layer')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--load_model', type=str, default='model.pt',
                    help='model checkpoint to use')
parser.add_argument('--save', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--gen_length', type=int, default='1000',
                    help='number of tokens to generate')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--fp16', action='store_true',
                    help='run in fp16 mode')
parser.add_argument('--neuron', type=int, default=-1,
                    help='''specifies which neuron to analyze for visualization or overwriting.
                         Defaults to maximally weighted neuron during classification steps''')
parser.add_argument('--visualize', action='store_true',
                    help='generates heatmap of main neuron activation [not working yet]')
parser.add_argument('--overwrite', type=float, default=None,
                    help='Overwrite value of neuron s.t. generated text reads as a +1/-1 classification')
parser.add_argument('--text', default='',
                    help='warm up generation with specified text first')
parser.add_argument('--emotion_text', default='',
                    help='transfer emotion from text, if supplied (try copy/merge top X neurons')
parser.add_argument('--num_emotion_neurons', type=int, default='25',
                    help='Cap number of neurons for emotion transfer?')
parser.add_argument('--emotion_factor', type=float, default='0.2',
                    help='Interpolate from emotion embedding -- 0.0 to 1.0')
parser.add_argument('--emotion_vector', default='',
                    help='Path to numpy file with vector to apply to hidden state (model specific)')
parser.add_argument('--emotion_gram_matrix', default='',
                    help='Gram matrix for average emotion in a category (which you try to match?)')
parser.add_argument('--style_weight', type=int, default=0,
                    help='How much to weight for style transfer -- content_weight == 1.0')
parser.add_argument('--attention', action='store_true',
                    help='')
parser.add_argument('--beam', type=int, default=1,
                    help='beam search decoder if value > 1. Default: 1 (no beam search)')
parser.add_argument('--topk', type=int, default=1,
                    help='display top k decodings from beam (must be <= `--beam` and >=1)')
args = parser.parse_args()

args.data_size = 256

# Set the random seed manually for reproducibility.
if args.seed >= 0:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

#if args.temperature < 1e-3:
#    parser.error("--temperature has to be greater or equal 1e-3")

#model = model.RNNModel(args.model, args.data_size, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).cuda()
model = model.RNNAutoEncoderModel(args.model, args.data_size, args.emsize, args.nhid, args.nlayers, args.dropout, tie_weights=False, teacher_force=False, attention=args.attention).cuda()

if args.fp16:
    model.half()
with open(args.load_model, 'rb') as f:
    sd = torch.load(f)

if 'rng' in sd:
    del sd['rng']
if 'cuda_rng' in sd:
    del sd['cuda_rng']

try:
#    print('try load w/o weightnorm')
    model.load_state_dict(sd)
#    print('load w/o weightnorm success')
except:
#    print('try with weight norm')
    apply_weight_norm(model.encoder.rnn, hook_child=True)
#    if not args.tied:
    apply_weight_norm(model.decoder.rnn, hook_child=True)
    #print([n for n,p in model.named_parameters()])
    model.load_state_dict(sd)
    remove_weight_norm(model)

#try:
#    model.load_state_dict(sd)
#except:
#    apply_weight_norm(model.encoder.rnn, hook_child=False)
#    apply_weight_norm(model.decoder.rnn, hook_child=False)
#    model.load_state_dict(sd)
#    remove_weight_norm(model)

beam = None
if args.beam > 1:
    beam = beam_search.BeamDecoder(args.beam, cuda=torch.cuda.is_available(), n_best=args.topk, vocab_pad=ord(' '))

def get_unforced_output(unforced_output):
    if beam is None:
        unforced_output = list(unforced_output[-1].squeeze().cpu().numpy())
        unforced_output = [chr(int(x)) for x in unforced_output]
        unforced_output = ''.join(unforced_output).replace('\n', '\\n')
    else:
        unforced_output = list(unforced_output[-1])
    #    print (unforced_output[0][0])
    #    exit()
        unforced_output = [''.join([chr(int(idx.cpu().data.item())) for idx in kth_best[0]]).replace('\n', '\\n') for kth_best in unforced_output]
    #    unforced_output = [chr(int(x)) for x in unforced_output]
    return unforced_output

def get_neuron_and_polarity(sd, neuron):
    """return a +/- 1 indicating the polarity of the specified neuron in the module"""
    if neuron == -1:
        neuron = None
    if 'classifier' in sd:
        sd = sd['classifier']
        if 'weight' in sd:
            weight = sd['weight']
        else:
            return neuron, 1
    else:
        return neuron, 1
    if neuron is None:
        val, neuron = torch.max(torch.abs(weight[0].float()), 0)
        neuron = neuron[0]
    val = weight[0][neuron]
    if val >= 0:
        polarity = 1
    else:
        polarity = -1
    return neuron, polarity

def process_hidden(cell, hidden, neuron, mask=False, mask_value=1, polarity=1):
    feat = cell.data[:, neuron]
    rtn_feat = feat.clone()
    if mask:
#        feat.fill_(mask_value*polarity)
        hidden.data[:, neuron].fill_(mask_value*polarity)
    return rtn_feat[0]

def model_step(model, input, neuron=None, mask=False, mask_value=1, polarity=1):
    out, _ = model(input)
    if neuron is not None:
        hidden = model.rnn.rnns[-1].hidden
        if len(hidden) > 1:
            hidden, cell = hidden
        else:
            hidden = cell = hidden
        feat = process_hidden(cell, hidden, neuron, mask, mask_value, polarity)
        return out, feat
    return out

# Sample character with a temperature...
def sample(out, temperature):
    if temperature == 0:
        char_idx = torch.max(out.squeeze().data, 0)[1][0]
    else:
        word_weights = out.float().squeeze().data.div(args.temperature).exp().cpu()
        char_idx = torch.multinomial(word_weights, 1)[0]
    return char_idx

def process_text(text, model, input, temperature, neuron=None, mask=False, overwrite=1, polarity=1):
    chrs = []
    vals = []
    for c in text:
        input.data.fill_(int(ord(c)))
        if neuron:
            ch, val = model_step(model, input, neuron, mask, overwrite, polarity)
            vals.append(val)
        else:
            ch = model_step(model, input, neuron, mask, overwrite, polarity)
#        ch = sample(ch, temperature)
    input.data.fill_(sample(ch, temperature))
    chrs = list(text)
#    chrs.append(chr(ch))
    return chrs, vals

# Generates with temperature...
def generate(gen_length, model, input, temperature, neuron=None, mask=False, overwrite=1, polarity=1):
    chrs = []
    vals = []
    for i in range(gen_length):
        chrs.append(chr(input.data[0]))
        if neuron:
            ch, val = model_step(model, input, neuron, mask, overwrite, polarity)
            vals.append(val)
        else:
            ch = model_step(model, input, neuron, mask, overwrite, polarity)
        ch = sample(ch, temperature)
        input.data.fill_(ch)
#        chrs.append(chr(ch))
#    chrs.pop()
    return chrs, vals

def make_heatmap(text, values, save=None, polarity=1):
    cell_height=.325
    cell_width=.15
    n_limit = 74
    text = list(map(lambda x: x.replace('\n', '\\n'), text))
    num_chars = len(text)
    total_chars = math.ceil(num_chars/float(n_limit))*n_limit
    mask = np.array([0]*num_chars + [1]*(total_chars-num_chars))
    text = np.array(text+[' ']*(total_chars-num_chars))
    values = np.array(values+[0]*(total_chars-num_chars))
    values *= polarity

    values = values.reshape(-1, n_limit)
    text = text.reshape(-1, n_limit)
    mask = mask.reshape(-1, n_limit)
    num_rows = len(values)
    plt.figure(figsize=(cell_width*n_limit, cell_height*num_rows))
    hmap=sns.heatmap(values, annot=text, mask=mask, fmt='', vmin=-1, vmax=1, cmap='RdYlGn',
                     xticklabels=False, yticklabels=False, cbar=False)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    # clear plot for next graph since we returned `hmap`
    plt.clf()
    return hmap


neuron, polarity = get_neuron_and_polarity(sd, args.neuron)
neuron = neuron if args.visualize or args.overwrite is not None else None
mask = args.overwrite is not None

model.eval()


# Hack -- upon request, dump
sample_texts = [" raised by his miserable aunt and uncle who are terrified Harry Potter will learn that he's really a wizard, just as his parents were ",
" Love love love me some bananas! I try to eat 1-2 every day for good health. These bananas were very ripe and sweet. Perfect!",
" I am sorry I wasted my hard earned money on it. The mattress was OK for about the first year. I wake up with a lot of back pain",
" My husband still wakes up with back pain. I have noticed no real difference in my sleep patterns on this bed. I was disappointed.",
"Very comfortable. My wife and I have been getting great sleep since day 1. Don't listen to the cry baby reviews. This mattress is great.",
"When I have to build a hotel, we're bombing the hell out of them. Lots of money. To those suffering, I say vote for Donald. Media hurting and left behind, I say: it looked like a million people. It's imploding as we sit with my steak.",
"Dimas 1/2 step late! Hella effort though, hella determination to even Get back on the track! Go USA! Got past the True Hardness! That BEAM's a BEEYOTCH! Les' get ta TUMBLIN'! Go USA!",
"I discovered Philip Roth, strangely enough, through a young woman who insisted that all of her friends join her in despising him. I never got to the bottom of that situation, and now I think I never will",
" They don't like bananas. They don't like plums. They don't like apricots. They love apples. Their favorite fruit is an apple. It is best for them. ",
" She doesn't like bananas. She doesn't like plums. She doesn't like apricots. She loves apples. Her favorite fruits are apples. That is best for her. ",
" He doesn't like bananas. He doesn't like plums. He doesn't like apricots. He loves apples. His favorite fruits are apples. That is best for him. ",
"Having a lil' weather outchere. Time to try that old school rain on the roof sleep inducer, got a lil' boat rock to go with that. I'm byacht to get it... Prolly pretty dope to be a kid in sailing camp. Been seeing a few lil' sailors in the mornings while I attend Chill Camp.",
"I told Ohio my promise to the American voter: If I am elected President, I will grow your money. $500 billion a year to be a Republican. The education is a disaster. Jobs are essentially nonexistent. What do you have to lose?",
"Uh Oh, lookin' like they turned the stove on & Home Cookin' is all they servin'! Bron almost had that First Down! Why would you fall for a Deron Williams shot fake?! Total Bullshit! Did that Muthafukka say No Way?! First Down James!"]

if False:
    model_name = 'len64'
    embeddings = np.empty((len(sample_texts), 4096), dtype='float32')
    txt_filename = model_name + '_texts'
    hid_filename = model_name + '_hiddens'
    model.decoder.teacher_force = False
    #print(embeddings)
    print('Storing embeddings for sentences of size %s' % str(embeddings.shape))
    for i, txt in enumerate(sample_texts):
        input_text = ' ' + txt.strip() + ' '
        print('\n-----\n%s' % input_text)
        hidden = model.encoder.rnn.init_hidden(1)
        input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_text]))).cuda().long()
        input = input.view(-1, 1).contiguous()
        unforced_output = model(input, temperature=args.temperature, beam=beam)
        #print(unforced_output)
        final_hidden = unforced_output[2][0].squeeze().data.cpu().numpy()
        print(final_hidden)
        unforced_output = get_unforced_output(unforced_output)
        embeddings[i] = final_hidden
        print((''.join(unforced_output)).replace('\n', ' '))

    # Save results to special file.
    print('Saving texts and hiddens to %s, %s' % (txt_filename, hid_filename))
    np.save(hid_filename, embeddings)
    np.save(txt_filename, np.array(sample_texts))


# HACK: Decode embeddings (artificially generated via GLOW, etc) into sentences.
generated_embeddings_filename = '/home/nyakovenko/sentiment-discovery/text_to_speech/project_average_outputs_len128.npy'
generated_embeddings_filename = '/home/nyakovenko/sentiment-discovery/text_to_speech/len128_hiddens.npy'
if True:
    model.decoder.teacher_force = False
    hidden = model.encoder.rnn.init_hidden(1)
    generated_embeddings = np.load(generated_embeddings_filename)
    model.emotion_neurons = [n for n in range(args.nhid)]
    for (i, emb) in enumerate(generated_embeddings):
        fit_hidden = Variable(torch.from_numpy(emb))
        #print(fit_hidden)
        # Try the learned hidden state?
        model.emotion_hidden_state = fit_hidden
        # How much do we boost?
        model.hidden_boost_factor = 1.0
        model.average_cell_value = True

        input_text = ' ' + args.text + ' '
        print('\n-----\n%s' % i)
        input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_text]))).cuda().long()
        input = input.view(-1, 1).contiguous()
        model.encoder.rnn.reset_hidden(1)
        unforced_output = model(input, temperature=args.temperature, beam=beam)
        unforced_output = get_unforced_output(unforced_output)
        print('unforced generated from hidden:')
        print((''.join(unforced_output)))

    # Turn off hidden state changing
    model.emotion_neurons = []

# Run through text -- no emotion transfer
print('Text autoencoder -- no emotion xfer')

input_text = ' ' + args.text + ' '
hidden = model.encoder.rnn.init_hidden(1)
input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_text]))).cuda().long()
input = input.view(-1, 1).contiguous()

unforced_output = model(input, temperature=args.temperature, beam=beam)
unforced_output = get_unforced_output(unforced_output)

# What happens if we run it twice? Naive hidden state warmup...
#unforced_output = model(input, temperature=args.temperature, beam=beam)
#unforced_output = get_unforced_output(unforced_output)

model.decoder.teacher_force = True
model.encoder.rnn.reset_hidden(1)
forced_output = model(input, temperature=args.temperature)
# Capture content encoder hidden state.
content_hidden_state = forced_output[2][0].squeeze().data#.cpu().numpy()
forced_output = list(forced_output[-1].squeeze().cpu().numpy())
forced_output = [chr(int(x)) for x in forced_output]

# Remove Forced Output -- it's not very interesting. 
#print('-----forced output-----')
#print((''.join(forced_output)).replace('\n', ' '))
print('\n-----unforced output-----')
print((''.join(unforced_output)).replace('\n', ' '))

if not args.emotion_text:
    print('--> No emotion text given.')
    exit()

# Now try emotion text -- by itself. Save encoder state
input_emotion_text = ' ' + args.emotion_text + ' '
input_emotion = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_emotion_text]))).cuda().long()
input_emotion = input_emotion.view(-1, 1).contiguous()
model.decoder.teacher_force = False
model.encoder.rnn.reset_hidden(1)
unforced_emotion_output = model(input_emotion, temperature=args.temperature, beam=beam)
unforced_emotion_output = get_unforced_output(unforced_emotion_output)

# What happens if we run it twice? Naive hidden state warmup...
#unforced_emotion_output = model(input_emotion, temperature=args.temperature, beam=beam)
#unforced_emotion_output = get_unforced_output(unforced_emotion_output)

model.decoder.teacher_force = True
model.encoder.rnn.reset_hidden(1)

# If we want to get the Gram over several steps of the emo text?
emo_steps = min(len(input_emotion_text)-10, 50) 
all_grams = []
for s in range(emo_steps):
    d = emo_steps - s - 1
    part_emo_text = input_emotion_text[:-d]
    # Skip all states in the middle of words
    if not(d == 0 or input_emotion_text[-d] == ' ') or len(part_emo_text) == 0:
        continue
    # Un-comment if we want full text
    #print(part_emo_text)
    part_emo_input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in part_emo_text]))).cuda().long()
    part_emo_input = part_emo_input.view(-1, 1).contiguous()
    part_output = model(part_emo_input, temperature=args.temperature)
    part_hid_state = part_output[2][0].squeeze().data
    part_gram = torch.ger(part_hid_state, part_hid_state).cpu().numpy()
    all_grams.append(part_gram)
all_grams = np.array(all_grams)
average_gram = np.mean(all_grams, axis=0)

forced_emotion_output = model(input_emotion, temperature=args.temperature)
# Key is that we capture the emotion encoder hidden state.
emotion_hidden_state = forced_emotion_output[2][0].squeeze().data#.cpu().numpy()

# Compute "outer product" on hidden state
# TODO: Average outer product over steps...
#outer_product_hidden_state = torch.ger(emotion_hidden_state, emotion_hidden_state)
outer_product_hidden_state=average_gram

emotion_cell_state = forced_emotion_output[2][1].squeeze().data#.cpu().numpy()
forced_emotion_output = list(forced_emotion_output[-1].squeeze().cpu().numpy())
forced_emotion_output = [chr(int(x)) for x in forced_emotion_output]

#print('\n-----forced emotion output-----')
#print((''.join(forced_emotion_output)).replace('\n', ' '))
print('\n-----unforced emotion output-----')
print((''.join(unforced_emotion_output)).replace('\n', ' '))

#print('Emotion hidden state %d' % len(emotion_hidden_state))
#print(emotion_hidden_state)
#print('Emotion cell state %d' % len(emotion_cell_state))
#print(emotion_cell_state)

# top 25
# emotion_neurons = [1162, 3494, 898, 732, 2022, 986, 2743, 3572, 977, 1526, 3737, 781, 490, 2264, 287, 2506, 550, 680, 3635, 3650, 4054, 3078, 2846, 3616, 159]
# top 156 [all neurons from transfer.py]
#emotion_neurons = [1162, 3494, 898, 732, 2022, 986, 2743, 3572, 977, 1526, 3737, 781, 490, 2264, 287, 2506, 550, 680, 3635, 3650, 4054, 3078, 2846, 3616, 159, 2701, 2987, 2858, 797, 1506, 1766, 1657, 922, 3168, 1682, 3428, 1140, 2872, 3982, 336, 237, 1862, 1083, 3984, 903, 19, 3760, 572, 3130, 3272, 4039, 3778, 2768, 110, 276, 1203, 1498, 1941, 1910, 3908, 2055, 1364, 2698, 2587, 3565, 115, 1666, 1937, 1825, 319, 1330, 1081, 103, 2482, 1287, 84, 3670, 2144, 963, 1522, 3241, 2955, 3160, 2536, 3997, 1190, 2978, 2888, 501, 1243, 3407, 1309, 470, 3296, 1587, 162, 1260, 3292, 2802, 3911, 2389, 1084, 3525, 2007, 1936, 377, 2643, 3092, 2183, 3820, 1413, 2780, 3173, 3430, 2039, 3274, 3252, 3865, 2485, 1775, 1702, 88, 2880, 3295, 3232, 2693, 654, 662, 577, 51, 3798, 1302, 790, 1494, 1045, 1120, 2552, 1562, 2878, 3302, 355, 1804, 3395, 1036, 2750, 1105, 516, 927, 2060, 1711, 1500, 3813, 2636, 3405, 2993, 447]
# For no-TF model -- shockingly, a similar list. Same initialization...
#emotion_neurons = [1162, 179, 3982, 1134, 2698, 3494, 3523, 732, 4048, 669, 490, 2298, 3399, 4033, 1100, 3022, 1920, 1344, 1728, 1506, 1517, 1827, 2872, 3106, 340, 572, 2318, 473, 3760, 134, 316, 796, 3405, 229, 1230, 163, 760, 1200, 1657, 1377, 540, 2573, 3295, 3208, 709, 2227, 2531, 352, 1941, 97, 1483, 1004, 2424, 1430, 371, 1783, 3738, 1379, 1620, 203, 3274, 2930, 2825, 2779, 3632, 1643, 3937, 1575, 1909, 2649, 1836, 728, 2587, 1522, 1672, 225, 537, 1937, 4006, 3751, 2310, 803, 631, 3423, 48, 853, 1989, 1862, 3750, 2803, 2787, 4069, 149, 3438, 305, 1488, 426, 841, 3922, 3446]
# /data/nicky/experiments/mlstm-AEnc-30-70-4096-Amazon-FP16-decoder-pretrain-noTied-len128-noCellcp-noXform/e112000_transfer/sentiment/topAveVector
emotion_neurons = [179, 1162, 1827, 3494, 2366, 296, 2618, 3650, 4053, 340, 3635, 2858, 235, 1100, 3632, 813, 2110, 1964, 4033, 2060, 3399, 2230, 3621, 1008, 2822, 2691, 2538, 1728, 3345, 139, 788, 445, 1717, 2880, 3295, 3158, 873, 986, 3891, 408, 2521, 581, 3022, 2482, 1839, 792, 2773, 3676, 3111, 2563, 540, 3052, 1498, 1654, 3596, 1305, 355, 161, 1003, 1553, 537, 2441, 1506, 2431, 3364, 3488, 146, 2066, 3376, 2434, 168, 3258, 345, 1941, 868, 3372, 669, 1497, 732, 3777, 2290, 1215, 3685, 1882, 1806, 3199, 1619, 1409, 177, 817, 3774, 3166, 2976, 1862, 1140, 4076, 507, 1306, 513, 4039]
# /data/nicky/experiments/mlstm-AEnc-30-70-4096-Amazon-FP16-decoder-pretrain-noTied-len64-noCellcp/e198000.pt
emotion_neurons = [1162, 3494, 3941, 1302, 2330, 1643, 3737, 2292, 340, 2743, 980, 2522, 796, 3157, 986, 1443, 1306, 2249, 1827, 2176, 1107, 3622, 2242, 1359, 2080, 4053, 3158, 3394, 366, 1791, 1134, 649, 35, 598, 51, 3621, 3551, 2389, 2060, 2911, 3756, 1647, 2227, 591, 1334, 3897, 1292, 3200, 4033, 1445, 1569, 1812, 3892, 3052, 3092, 3033, 788, 3070, 3632, 2010, 3779, 584, 1970, 3891, 3650, 2878, 2619, 2366, 1008, 781, 305, 2615, 924, 2624, 1801, 1978, 1709, 2313, 1304, 1154, 276, 181, 1921, 1862, 3473, 2751, 2231, 355, 2888, 3100, 736, 1904, 1555, 3167, 296, 1422, 3862, 372, 2092, 1086, 882, 888, 1430, 852, 1070, 371, 3003, 2222, 3331, 863, 2934, 2880, 2697, 1800, 1831, 3405, 105, 537, 2197, 2991, 1243, 3652, 168, 3203, 3076, 3469, 3984, 614, 162, 2079, 2861, 3106, 316, 1515, 2570, 1612, 1202, 2022, 272, 3139, 1409, 3854, 912, 3647, 1004, 1714, 3543, 2145, 3596, 3275]
print('Using emotion xfer on %d neurons' % len(emotion_neurons))
#for n in emotion_neurons:
#    print('\t%d:\t%.3f/%.3f' % (n, emotion_hidden_state[n], emotion_cell_state[n]))

# Now transfer these values during the next content-based generation
#model.emotion_neurons = emotion_neurons[:args.num_emotion_neurons]
# Maybe transfer all neurons?
model.emotion_neurons = [n for n in range(args.nhid)]
print('transfering on %d neurons' % len(model.emotion_neurons))
model.emotion_hidden_state = emotion_hidden_state
model.emotion_cell_state = emotion_cell_state
# How much do we boost?
model.hidden_boost_factor = args.emotion_factor
print('transfering with %.3f emo factor' % model.hidden_boost_factor)
model.cell_boost_factor = 0.0
# Do we average values, or over-write (default is overwrite)
model.average_cell_value = True

# TODO -- if supplied, load "sentiment vector" and apply it to the hidden state.
# This is the average in Pos - Neg dimension across neurons.
# TODO -- add multiple to emotion vector (can make it negative)
# TODO -- if applied to hidden state... make sure we clip final values
if args.emotion_vector:
    print('Use emotion vector %s' % args.emotion_vector)
    emotion_vector = np.load(args.emotion_vector)
    #emotion_vector *= 5.0 # can increase, or take in a different direction?
    print(emotion_vector)
    print(np.argsort(emotion_vector)[-20:])
    print(emotion_vector[np.argsort(emotion_vector)])
    # Add to hidden or cell state -- depending on the model.
    model.use_added_hidden_state = True
    model.added_hidden_state_vector = torch.from_numpy(emotion_vector * 2.0).float().cuda()
    #model.use_added_cell_state = True
    #model.added_cell_state_vector = torch.from_numpy(emotion_vector * -30.0).float().cuda()

# Run through text -- with attempted emotion transfer
print('\nText autoencoder -- attempt emotion xfer')

input_text = ' ' + args.text + ' '
hidden = model.encoder.rnn.init_hidden(1)
input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_text]))).cuda().long()
input = input.view(-1, 1).contiguous()
model.decoder.teacher_force = False
model.encoder.rnn.reset_hidden(1)
unforced_output = model(input, temperature=args.temperature, beam=beam)
unforced_output = get_unforced_output(unforced_output)
#model.decoder.teacher_force = True
#model.encoder.rnn.reset_hidden(1)
#forced_output = model(input, temperature=args.temperature)
#forced_output = list(forced_output[-1].squeeze().cpu().numpy())
#forced_output = [chr(int(x)) for x in forced_output]
#print('-----forced output-----')
#print((''.join(forced_output)).replace('\n', ' '))
print('\n-----unforced output-----')
print((''.join(unforced_output)).replace('\n', ' '))

# Need SGD approach to style loss.
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = target_feature.detach()
    def forward(self, input):
        G = torch.ger(input,input)
        self.loss = F.mse_loss(G, self.target)
        return input


fit_hidden = content_hidden_state.clone()
print(fit_hidden)
fit_hidden = Variable(fit_hidden)
print(fit_hidden)
fit_hidden.requires_grad = True

xfer_model = nn.Sequential()
content_loss = ContentLoss(content_hidden_state)
xfer_model.add_module("content_loss_{}".format(0), content_loss)
outer_product_hidden_state = torch.from_numpy(outer_product_hidden_state).cuda()
if args.emotion_gram_matrix:
    print('Loading emotion gram matrix from %s' % args.emotion_gram_matrix)
    outer_product_hidden_state = np.load(args.emotion_gram_matrix)
    print(outer_product_hidden_state.shape)
    outer_product_hidden_state = torch.from_numpy(outer_product_hidden_state).float().cuda()
style_loss = StyleLoss(outer_product_hidden_state)
xfer_model.add_module("style_loss_{}".format(0), style_loss)
optimizer = optim.LBFGS([fit_hidden])

num_steps = 20
style_weight = 100 # 50 # 100 # 200 # 100. #200.
if args.style_weight:
    style_weight = args.style_weight
content_weight = 1.
print('Fitting with content weight %.2f, style weight %.2f' % (content_weight, style_weight))
if True:
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            optimizer.zero_grad()
            xfer_model(fit_hidden)
            style_score = 0
            content_score = 0

            style_score = style_loss.loss
            content_score += content_loss.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 1 == 0:
                #print("run {}:".format(run))
                print('run {} Style Loss : {:8f} Content Loss: {:8f}'.format(
                    run, style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    print(fit_hidden)


# Try the learned hidden state?
model.emotion_hidden_state = fit_hidden
# How much do we boost?
model.hidden_boost_factor = args.emotion_factor

input_text = '\n ' + args.text + ' \n'
hidden = model.encoder.rnn.init_hidden(1)
input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_text]))).cuda().long()
input = input.view(-1, 1).contiguous()
model.decoder.teacher_force = False
model.encoder.rnn.reset_hidden(1)
unforced_output = model(input, temperature=args.temperature, beam=beam)
unforced_output = get_unforced_output(unforced_output)
model.decoder.teacher_force = True
model.encoder.rnn.reset_hidden(1)
forced_output = model(input, temperature=args.temperature)
forced_output = list(forced_output[-1].squeeze().cpu().numpy())
forced_output = [chr(int(x)) for x in forced_output]
#print('-----forced output-----')
#print((''.join(forced_output)).replace('\n', ' '))
print('\n-----unforced output-----')
print((''.join(unforced_output)).replace('\n', ' '))


#print(res)

exit()

input = Variable(torch.LongTensor([int(ord('\n'))])).cuda()
input = input.view(1,1).contiguous()
model_step(model, input, neuron, mask, args.overwrite, polarity)
input.data.fill_(int(ord(' ')))
out = model_step(model, input, neuron, mask, args.overwrite, polarity)
if neuron is not None:
    out = out[0]
input.data.fill_(sample(out, args.temperature))

outchrs = []
outvals = []
#with open(args.save, 'w') as outf:
with torch.no_grad():
    if args.text != '':
        chrs, vals = process_text(args.text, model, input, args.temperature, neuron, mask, args.overwrite, polarity)
        outchrs += chrs
        outvals += vals
    chrs, vals = generate(args.gen_length, model, input, args.temperature, neuron, mask, args.overwrite, polarity)
    outchrs += chrs
    outvals += vals
outstr = ''.join(outchrs)
print(outstr)
with open(args.save, 'w') as f:
    f.write(outstr)

if args.visualize:
    make_heatmap(outchrs, outvals, os.path.splitext(args.save)[0]+'.png', polarity)
