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
                    help='Patho to numpy file with vector to apply to hidden state (model specific)')
parser.add_argument('--attention', action='store_true',
                    help='')
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

# Run through text -- no emotion transfer
print('Text autoencoder -- no emotion xfer')

input_text = '\n ' + args.text + ' \n'
hidden = model.encoder.rnn.init_hidden(1)
input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_text]))).cuda().long()
input = input.view(-1, 1).contiguous()
unforced_output = model(input, temperature=args.temperature)
model.decoder.teacher_force = True
model.encoder.rnn.reset_hidden(1)
forced_output = model(input, temperature=args.temperature)
forced_output = list(forced_output[-1].squeeze().cpu().numpy())
unforced_output = list(unforced_output[-1].squeeze().cpu().numpy())
forced_output = [chr(int(x)) for x in forced_output]
unforced_output = [chr(int(x)) for x in unforced_output]
# Capture content encoder hidden state.
content_hidden_state = forced_emotion_output[2][0].squeeze().data#.cpu().numpy()

print('-----forced output-----')
print((''.join(forced_output)).replace('\n', ' '))
print('-----unforced output-----')
print((''.join(unforced_output)).replace('\n', ' '))

if not args.emotion_text:
    print('--> No emotion text given.')
    exit()

# Now try emotion text -- by itself. Save encoder state
input_emotion_text = '\n ' + args.emotion_text
input_emotion = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_emotion_text]))).cuda().long()
input_emotion = input_emotion.view(-1, 1).contiguous()
model.decoder.teacher_force = False
model.encoder.rnn.reset_hidden(1)
unforced_emotion_output = model(input_emotion, temperature=args.temperature)
model.decoder.teacher_force = True
model.encoder.rnn.reset_hidden(1)
forced_emotion_output = model(input_emotion, temperature=args.temperature)
# Key is that we capture the emotion encoder hidden state.
emotion_hidden_state = forced_emotion_output[2][0].squeeze().data#.cpu().numpy()

# Compute "outer product" on hidden state
# TODO: Average outer product over steps...
print(forced_emotion_output[2])
outer_product_hidden_state = torch.ger(emotion_hidden_state, emotion_hidden_state)
print(outer_product_hidden_state)

emotion_cell_state = forced_emotion_output[2][1].squeeze().data#.cpu().numpy()
forced_emotion_output = list(forced_emotion_output[-1].squeeze().cpu().numpy())
unforced_emotion_output = list(unforced_emotion_output[-1].squeeze().cpu().numpy())
forced_emotion_output = [chr(int(x)) for x in forced_emotion_output]
unforced_emotion_output = [chr(int(x)) for x in unforced_emotion_output]

print('\n-----forced emotion output-----')
print((''.join(forced_emotion_output)).replace('\n', ' '))
print('-----unforced emotion output-----')
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

input_text = '\n ' + args.text + ' \n'
hidden = model.encoder.rnn.init_hidden(1)
input = Variable(torch.from_numpy(np.array([int(ord(c)) for c in input_text]))).cuda().long()
input = input.view(-1, 1).contiguous()
model.decoder.teacher_force = False
model.encoder.rnn.reset_hidden(1)
unforced_output = model(input, temperature=args.temperature)
model.decoder.teacher_force = True
model.encoder.rnn.reset_hidden(1)
forced_output = model(input, temperature=args.temperature)
forced_output = list(forced_output[-1].squeeze().cpu().numpy())
unforced_output = list(unforced_output[-1].squeeze().cpu().numpy())
forced_output = [chr(int(x)) for x in forced_output]
unforced_output = [chr(int(x)) for x in unforced_output]

print('-----forced output-----')
print((''.join(forced_output)).replace('\n', ' '))
print('-----unforced output-----')
print((''.join(unforced_output)).replace('\n', ' '))

# Style transfer...
# Experimental -- try solving for X so that min(||X-A||+||style(X)-style(B)||)
# where style is the gram matrix over final B, or averaged over multiple steps

def gram_transfer_loss(emb,content_emb,style_gram,style_weight):
    content_weight = 1.0 - style_weight
    content_loss = np.average((emb - content_emb)**2)
    style_loss = np.average((np.outer(emb, emb) - style_gram)**2)
    return content_weight*content_loss + style_weight*style_loss
#    return np.sum( ((coord[0] - x)**2) + ((coord[1] - y)**2) - (r**2) )

print('Attempting content/emotion style transfer')
print('content hidden state')
print(content_hidden_state)
print('emotion hidden state')
print(emotion_hidden_state)
STYLE_WEIGHT = 0.1
res = minimize(gram_transfer_loss,content_hidden_state,args = [content_hidden_state,outer_product_hidden_state,STYLE_WEIGHT])
print('Completed style transfer')

print(res)

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
