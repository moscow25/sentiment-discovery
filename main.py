import argparse
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import random
from torch.autograd import Variable

from fp16 import FP16_Module, FP16_Optimizer

import data
import model
from model import DistributedDataParallel as DDP

from apex.reparameterization import apply_weight_norm, remove_weight_norm
from configure_data import configure_data
from learning_rates import LinearLR
from LARC import LARC

parser = argparse.ArgumentParser(description='PyTorch Sentiment-Discovery Language Modeling')
parser.add_argument('--model', type=str, default='mLSTM',
                    help='type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM, GRU)')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=4096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the encoder to the decoder parameters')
parser.add_argument('--freeze', action='store_true',
                    help='freeze the encoder weights')
parser.add_argument('--freeze_decoder', action='store_true',
                    help='freeze the decoder weights (only makes sense for discriminator training')
parser.add_argument('--no_force', action='store_true',
                    help='No teacher forcing. Use temperature to sample from output distribution')
parser.add_argument('--attention', action='store_true',
                    help='use highway attention between final encoder state and decoder')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='lang_model.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default='',
                    help='path to a previously saved model checkpoint')
parser.add_argument('--save_iters', type=int, default=2000, metavar='N',
                    help='save current model progress interval')
parser.add_argument('--fp16', action='store_true',
                    help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--dynamic_loss_scale', action='store_true',
                    help='Dynamically look for loss scalar for fp16 convergance help.')
parser.add_argument('--no_weight_norm', action='store_true',
                    help='Add weight normalization to model.')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='Static loss scaling, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--world_size', type=int, default=1,
                    help='number of distributed workers')
parser.add_argument('--distributed_backend', default='gloo',
                    help='which backend to use for distributed training. One of [gloo, nccl]')
parser.add_argument('--rank', type=int, default=-1,
                    help='distributed worker rank. Typically set automatically from multiproc.py')
parser.add_argument('--optim', default='SGD',
                    help='One of SGD or Adam')
parser.add_argument('--load_pretrained', type=str, default='',
                    help='load a pretrained language model into the encoder')
parser.add_argument('--init_transform_id', action='store_true',
                    help='Initialize last hidden to init-hidden in decoder, with identity. Why? Pretrained language model.')

# Control what elements are used to pass state to the decoder. Why? May help with emotion transfer
parser.add_argument('--decoder_use_hidden', action='store_true',
                    help='Pass hidden state to decoder')
parser.add_argument('--decoder_use_cell', action='store_true',
                    help='Pass cell state to decoder')
parser.add_argument('--decoder_xform_hidden', action='store_true',
                    help='Linear transform (with a tanh()) on hidden to decoder')
parser.add_argument('--decoder_xform_cell', action='store_true',
                    help='Linear transform on cell state to decoder')
parser.add_argument('--latent_use_tanh', action='store_true',
                    help='Squash latent (hidden to hidden) transform with tanh()? Deprecating if not needed')
# Highway -- Pass encoder hidden state to every decoder step?
parser.add_argument('--decoder_highway_hidden', action='store_true',
                    help='Highway layer from encoder hidden for every decoder step? (simple +=)')

# Control for Variable Teacher Forcing @nicky
parser.add_argument('--force_ctrl', type=float, default=0.,
                    help='percentage of training with no teacher forcing. (0.2=20% no teacher forcing)')

# TODO -- pass choices for training discriminator(s) for real/fake content


# Resume training arguments @nicky
parser.add_argument('--save_optim', action='store_true',
                    help='save optimizer in addtion to model')
parser.add_argument('--load_optim', type=str, default='',
                    help='path to load optimizer to resume training')
parser.add_argument('--blowup_restore', action='store_true',
                    help='employ our blowup restoring heuristic to reload old checkpoints on blow up')

# Encoder/decoder loss ctrl
parser.add_argument('--decoder_weight', type=float, default=.6,
                    help='decoder/encoder loss weighting. encoder_weight=1-decoder_weight. Default: .7')
parser.add_argument('--encoder_disc_weight', type=float, default=0.1,
                    help='discriminator weight (from encoder hidden state directly)')
parser.add_argument('--hidden_noise_factor', type=float, default=0.2,
                    help='how much noise to add to real hidden states (for discriminator)')

# Boris LARC optimizer
parser.add_argument('--LARC', action='store_true',
                    help='use LARC optimizer. (not working with FP16)')
parser.add_argument('--trust_coeff', type=float, default=.02,
                   help='trust coefficient value for LARC optimizer')

# Add dataset args to argparser and set some defaults
data_config, data_parser = configure_data(parser)
data_config.set_defaults(data_set_type='unsupervised', transpose=True)
data_parser.set_defaults(split='100,1,1')
data_parser = parser.add_argument_group('language modeling data options')
data_parser.add_argument('--seq_length', type=int, default=256,
                         help="Maximum sequence length to process (for unsupervised rec)")
data_parser.add_argument('--eval_seq_length', type=int, default=256,
                         help="Maximum sequence length to process for evaluation")
data_parser.add_argument('--lazy', action='store_true',
                         help='whether to lazy evaluate the data set')
data_parser.add_argument('--persist_state', type=int, default=1,
                         help='0=reset state after every sample in a shard, 1=reset state after every shard, -1=never reset state')
data_parser.add_argument('--num_shards', type=int, default=102,
                         help="""number of total shards for unsupervised training dataset. If a `split` is specified,
                                 appropriately portions the number of shards amongst the splits.""")
data_parser.add_argument('--val_shards', type=int, default=0,
                         help="""number of shards for validation dataset if validation set is specified and not split from training""")
data_parser.add_argument('--test_shards', type=int, default=0,
                         help="""number of shards for test dataset if test set is specified and not split from training""")
data_parser.add_argument('--temperature', type=float, default=0.1,
                         help="""sampling temperature to use during generation -- NOTE: temp=0 broken""")


args = parser.parse_args()

torch.backends.cudnn.enabled = False
args.cuda = torch.cuda.is_available()

# initialize distributed process group and set device
if args.cuda:
    if args.rank > 0:
        torch.cuda.set_device(args.rank % torch.cuda.device_count())

if args.world_size > 1:
    distributed_init_file = os.path.splitext(args.save)[0]+'.distributed.dpt'
    torch.distributed.init_process_group(backend=args.distributed_backend, world_size=args.world_size,
                                                    init_method='file://'+distributed_init_file, rank=args.rank)

# Set the random seed manually for reproducibility.
if args.seed is not -1:
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)


if args.loss_scale != 1 and args.dynamic_loss_scale:
    raise RuntimeError("Static loss scale and dynamic loss scale cannot be used together.")

###############################################################################
# Load data
###############################################################################

# Starting from sequential data, the unsupervised dataset type loads the corpus
# into rows. With the alphabet as the our corpus and batch size 4, we get
# ┌ a b c d e f ┐
# │ g h i j k l │
# │ m n o p q r │
# └ s t u v w x ┘.
# These rows are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
#
# The unsupervised dataset further splits the corpus into shards through which
# the hidden state is persisted. The dataset also produces a hidden state
# reset mask that resets the hidden state at the start of every shard. A valid
# mask might look like
# ┌ 1 0 0 0 0 0 ... 0 0 0 1 0 0 ... ┐
# │ 1 0 0 0 0 0 ... 0 1 0 0 0 0 ... │
# │ 1 0 0 0 0 0 ... 0 0 1 0 0 0 ... │
# └ 1 0 0 0 0 0 ... 1 0 0 0 0 0 ... ┘.
# With 1 indicating to reset hidden state at that particular minibatch index

train_data, val_data, test_data = data_config.apply(args)

###############################################################################
# Build the model
###############################################################################

ntokens = args.data_size
#model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
model = model.RNNAutoEncoderModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
    dropout=args.dropout, tie_weights=args.tied, freeze=args.freeze, freeze_decoder=args.freeze_decoder,
    teacher_force=not args.no_force, attention=args.attention, init_transform_id=args.init_transform_id,
    use_latent_hidden=args.decoder_use_hidden, transform_latent_hidden=args.decoder_xform_hidden,
    latent_tanh=args.latent_use_tanh,
    use_cell_hidden=args.decoder_use_cell, transform_cell_hidden=args.decoder_xform_cell,
    decoder_highway_hidden=args.decoder_highway_hidden)
if args.cuda:
    print('Compiling model in CUDA mode [make sure]')
    model = model.cuda()
print('* number of parameters: %d' % sum([p.nelement() for p in model.parameters()]))

rnn_model = model
#print(model._modules)

if args.load != '':
    sd = torch.load(args.load, map_location=lambda storage, loc: storage)
    #@nicky: here's how to fix the reloading problem I was talking about
    #sd['decoder']=sd['encoder']
    if 'rng' in sd:
        torch.set_rng_state(sd['rng'])
        del sd['rng']
    if 'cuda_rng' in sd:
        if args.cuda:
            torch.cuda.set_rng_state(sd['cuda_rng'])
        del sd['cuda_rng']

    #sd = torch.load(args.load)
    try:
        print('try load w/o weightnorm')
        model.load_state_dict(sd)
        print('load w/o weightnorm success')
    except:
        print('try with weight norm')
        apply_weight_norm(model.encoder.rnn, hook_child=True)
        if not args.tied:
            apply_weight_norm(model.decoder.rnn, hook_child=True)
        model.load_state_dict(sd)
        remove_weight_norm(model)

if args.load_pretrained != '':
    sd = torch.load(args.load_pretrained, map_location=lambda storage, loc: storage)
    try:
        print('try load w/o weightnorm')
        model.encoder.load_state_dict(sd)
        print('load w/o weightnorm success')
    except:
        print('try with weight norm')
        apply_weight_norm(model.encoder.rnn, hook_child=True)
        if not args.tied:
            apply_weight_norm(model.decoder.rnn, hook_child=True)
        model.encoder.load_state_dict(sd)
        # If we don't tie weigths... we still want to initialize decoder to a reasonable langauge model
        if not args.tied:
            model.decoder.load_state_dict(sd)
        remove_weight_norm(model)

if not args.no_weight_norm:
    apply_weight_norm(model.encoder.rnn, hook_child=True)
    if not args.tied:
        apply_weight_norm(model.decoder.rnn, hook_child=True)

# create optimizer and fp16 models
args.save_optim = args.save_optim or args.blowup_restore
optimizer_params = None
if not args.freeze and not args.freeze_decoder:
    optimizer_params = model.parameters()
else:
    # If freezing part of the network, sadly, we have to iterate over all possible items that could be in the network
    optimizer_params = []
    if model.encoder:
        optimizer_params += list(model.encoder.parameters())
    if model.decoder:
        optimizer_params += list(model.decoder.parameters())
    if model.latent_hidden_transform:
        optimizer_params += list(model.latent_hidden_transform.parameters())
    if model.latent_cell_transform:
        optimizer_params += list(model.latent_cell_transform.parameters())
    if model.disc_enc_transform:
        optimizer_params += list(model.disc_enc_transform.parameters())
    if model.disc_enc_partial_transform:
        optimizer_params += list(model.disc_enc_partial_transform.parameters())
    if model.disc_dec_transform:
        optimizer_params += list(model.disc_dec_transform.parameters())
    if model.disc_dec_partial_transform:
        optimizer_params += list(model.disc_dec_partial_transform.parameters())
    if model.disc_combo_transform:
        optimizer_params += list(model.disc_combo_transform.parameters())
    #optimizer_params = list(model.decoder.parameters())+list(model.latent_hidden_transform.parameters())+list(model.latent_cell_transform.parameters())
if args.fp16:
    model = FP16_Module(model)
    optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)
    if args.LARC:
        optim = LARC(optim, args.trust_coeff)
    optim = FP16_Optimizer(optim,
                           static_loss_scale=args.loss_scale,
                           dynamic_loss_scale=args.dynamic_loss_scale)
else:
    optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)
    optim = LARC(optim, args.trust_coeff)


# add linear learning rate scheduler
if train_data is not None:
    num_iters = len(train_data) * args.epochs
    LR = LinearLR(optim, num_iters)

if args.load_optim != '':
    optim.load_state_dict(torch.load(args.load_optim))

# wrap model for distributed training
if args.world_size > 1:
    model = DDP(model)

# Per-character loss on reconstruction
criterion = nn.CrossEntropyLoss()
# Binary loss for discriminator (real vs fake text)
criterion_disc = nn.BCEWithLogitsLoss()

###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.seq_length.
# If source is equal to the example output of the data loading example, with
# a seq_length limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the data loader. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM. A Variable representing an appropriate
# shard reset mask of the same dimensions is also returned.

def get_batch(data):
    reset_mask_batch = data[1].long()
    data = data[0].long()
    if args.cuda:
        reset_mask_batch = reset_mask_batch.cuda()
        data = data.cuda()
    text_batch = Variable(data[:,:-1].t().contiguous(), requires_grad=False)
    target_batch = Variable(data[:,1:].t().contiguous(), requires_grad=False)
    reset_mask_batch = Variable(reset_mask_batch[:,:text_batch.size(0)].t().contiguous(), requires_grad=False)
    return text_batch, target_batch, reset_mask_batch

def init_hidden(batch_size):
    return rnn_model.encoder.rnn.init_hidden(args.batch_size)

# Return printable part of the string.
def cleanup_text(text):
    t = text.replace('\n', ' ')
    t = t.replace('\r', ' ')
    t = t.replace('\t', ' ')
    return ''.join(x for x in t if (31 < ord(x) < 127))

#def should_teacher_force():
#    p = random.random()
#    return p >= args.force_ctrl

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    init_hidden(args.batch_size)
    total_loss = 0
    ntokens = args.data_size
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            data, targets, reset_mask = get_batch(batch)
            model.reset_hidden_state = False
            output_enc, output_dec, encoder_hidden, encoder_disc, decoder_disc, combo_disc, sampled_out = model(data, reset_mask=reset_mask, temperature=args.temperature)
            loss_enc = criterion(output_enc.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
            loss_dec = criterion(output_dec.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
            w_enc, w_dec = (1-args.decoder_weight), args.decoder_weight
            loss = w_enc * loss_enc + w_dec * loss_dec
#            output_flat = output.view(-1, ntokens).contiguous().float()
            total_loss += loss.data[0]
#            total_loss += criterion(output_flat, targets.view(-1).contiguous()).data[0]
    return total_loss / max(len(data_source), 1)

def train(total_iters=0):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_enc_loss, total_dec_loss, total_enc_disc_loss, total_dec_disc_loss, total_combo_disc_loss = 0, 0, 0, 0, 0
    start_time = time.time()
    ntokens = args.data_size
    hidden = init_hidden(args.batch_size)
    curr_loss = 0.
    if args.blowup_restore:
        print('running with args.blowup_restore -- in case model explodes and needs reset')
    else:
        print('running *without* args.blowup_restore -- ur in danger if model blows up and keeps running.')

    for i, batch in enumerate(train_data):

        data, targets, reset_mask = get_batch(batch)
        #output, hidden = model(data, reset_mask=reset_mask)
        #rnn_model.decoder.teacher_force = should_teacher_force()
        model.reset_hidden_state = False
        output_enc, output_dec, encoder_hidden, encoder_disc, decoder_disc, combo_disc, sampled_out = model(data, reset_mask=reset_mask, temperature=args.temperature, variable_tf=args.force_ctrl)

        # NOTE: Make sure these values are legal -- Var or return None if not specified
        #if i % 500 == 0:
        #    print('Real (encoder) disc values:')
        #    print(F.sigmoid(encoder_disc))
        #    print('Real (decoder) disc values:')
        #    print(F.sigmoid(decoder_disc))
        #    print('Real (combined) disc values')
        #    print(F.sigmoid(combo_disc))

        if i % 1000 == 0:
            print_len = min(args.batch_size, 3)
            encoder_text, decoder_text = rnn_model.get_text_from_outputs((output_enc, output_dec), temperature=args.temperature)
            print('------\nActual text:')
            print('\n'.join([(''.join([chr(c) for c in list(targets[:,l].data.cpu().numpy())])).replace('\n',' ') for l in range(print_len)]))
            print('------\nEncoder, decoder, sampled_out text:')
            print('\n'.join([''.join(cleanup_text(text)) for text in encoder_text[:print_len]]).encode('utf-8').decode('ascii','backslashreplace'))
            print('-------')
            print('\n'.join([''.join(cleanup_text(text)) for text in decoder_text[:print_len]]).encode('utf-8').decode('ascii','backslashreplace'))
            # TODO: Decode sampled_out string via char conversion
            #print('-------')
            #print('\n'.join([''.join(cleanup_text(text)) for text in sampled_out[:print_len]]).encode('utf-8').decode('ascii','backslashreplace'))

        #loss = criterion(output.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
        loss_enc = criterion(output_enc.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
        loss_dec = criterion(output_dec.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())
        # If training discriminator -- real samples are all 1.0
        disc_ones = torch.ones_like(encoder_disc)
        loss_enc_disc = criterion_disc(encoder_disc, disc_ones)
        loss_dec_disc = criterion_disc(decoder_disc, disc_ones)
        loss_combo_disc = criterion_disc(combo_disc, disc_ones)
        loss_disc = (loss_enc_disc + loss_dec_disc + loss_combo_disc) / 3.0

        w_enc, w_dec, w_enc_disc = (1-args.decoder_weight-args.encoder_disc_weight), args.decoder_weight, args.encoder_disc_weight
        loss = w_enc * loss_enc + w_dec * loss_dec + w_enc_disc * loss_disc

        optim.zero_grad()

        if args.fp16:
            optim.backward(loss)
        else:
            loss.backward()
        total_loss += loss.data.float()
        total_enc_loss += loss_enc.data.float()
        total_dec_loss += loss_dec.data.float()
        total_enc_disc_loss += loss_enc_disc.data.float()
        total_dec_disc_loss += loss_dec_disc.data.float()
        total_combo_disc_loss += loss_combo_disc.data.float()

        # clipping gradients helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip > 0:
            if not args.fp16:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            else:
                optim.clip_fp32_grads(clip=args.clip)
        optim.step()


        # Now do a batch (or X batches) with random hidden state (or another fake hidden) for discriminator
        prev_hidden = encoder_hidden[0][0].detach()
        encoder_hidden_mean = torch.mean(prev_hidden, dim=0).unsqueeze(0)
        encoder_hidden_mean = encoder_hidden_mean.expand(args.batch_size, encoder_hidden_mean.shape[1])
        encoder_hidden_stdev = torch.std(prev_hidden, dim=0).unsqueeze(0)
        encoder_hidden_stdev = encoder_hidden_stdev.expand(args.batch_size, encoder_hidden_stdev.shape[1])
        #print(encoder_hidden_mean)
        #print(encoder_hidden_stdev)
        hidden_sampler = D.normal.Normal(encoder_hidden_mean, encoder_hidden_stdev)
        hid_sample = hidden_sampler.sample().unsqueeze(0)
        encoder_hidden_fake = hid_sample

        # Make the job harder -- average the real, and the fake
        fake_factor = args.hidden_noise_factor
        #real_factor = 1.0 - fake_factor
        #encoder_hidden_fake = prev_hidden * real_factor + encoder_hidden_fake * fake_factor
        # NOTE: Instead, vary noise level along a mean and stdev also (include bigger and smaller noise points within the batch)
        fake_factor_mean = torch.FloatTensor([fake_factor]).unsqueeze(0)
        fake_factor_mean = fake_factor_mean.expand(args.batch_size, 1)
        # TODO: Add another noise stdev factor? Overkill.
        implied_fake_stdev = min(0.2, args.hidden_noise_factor / 2.0)
        fake_factor_stdev = torch.FloatTensor([implied_fake_stdev]).unsqueeze(0)
        fake_factor_stdev = fake_factor_stdev.expand(args.batch_size, 1)
        fake_factor_sampler = D.normal.Normal(fake_factor_mean, fake_factor_stdev)
        fake_factor_vector = fake_factor_sampler.sample()
        # Set a floor on minimum noise
        fake_factor_vector = torch.clamp(fake_factor_vector, min(0.1, args.hidden_noise_factor / 10.0), 1.0)
        #print("fake factor vector for %.3f mean and %.3f stdev. " % (fake_factor, implied_fake_stdev))
        #print(fake_factor_vector)
        #print(torch.mean(fake_factor_vector))
        real_factor_vector = 1.0 - fake_factor_vector
        # TODO: Can we move to CUDA earlier in sampling? Does it matter?
        if args.cuda:
            real_factor_vector = real_factor_vector.cuda()
            encoder_hidden_fake = encoder_hidden_fake.cuda()
            fake_factor_vector = fake_factor_vector.cuda()
        encoder_hidden_fake = prev_hidden * real_factor_vector + encoder_hidden_fake * fake_factor_vector


        #print(encoder_hidden_fake)
        model.reset_hidden_state = True
        model.reset_hidden_state_value = encoder_hidden_fake
        output_enc, output_dec, encoder_hidden, encoder_disc, decoder_disc, combo_disc, sampled_out = model(data, reset_mask=reset_mask, temperature=args.temperature, variable_tf=args.force_ctrl)

        # NOTE: Make sure these values are legal -- Var or return None if not specified
        #if i % 500 == 0:
        #    print('Fake (encoder) disc values:')
        #    print(F.sigmoid(encoder_disc))
        #    print('Fake (decoder) disc values:')
        #    print(F.sigmoid(decoder_disc))
        #    print('Fake (combined) disc values')
        #    print(F.sigmoid(combo_disc))

        # If training discriminator -- fake samples are all 0.0
        disc_zeros = torch.zeros_like(encoder_disc)
        loss_enc_disc = criterion_disc(encoder_disc, disc_zeros)
        loss_dec_disc = criterion_disc(decoder_disc, disc_zeros)
        loss_combo_disc = criterion_disc(combo_disc, disc_zeros)
        loss_disc = (loss_enc_disc + loss_dec_disc + loss_combo_disc) / 3.0
        loss = w_enc_disc * loss_disc

        optim.zero_grad()

        if args.fp16:
            optim.backward(loss)
        else:
            loss.backward()
        total_enc_disc_loss += loss_enc_disc.data.float()
        total_dec_disc_loss += loss_dec_disc.data.float()
        total_combo_disc_loss += loss_combo_disc.data.float()

        # clipping gradients helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip > 0:
            if not args.fp16:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            else:
                optim.clip_fp32_grads(clip=args.clip)
        optim.step()

        # Reset for next time! Otherwise won't use real data for encoder
        model.reset_hidden_state = False

        # step learning rate and log training progress
        lr = LR.get_lr()[0]
        if not args.fp16:
            LR.step()
        else:
            # if fp16 optimizer skips gradient step due to explosion do not step lr
            if not optim.overflow:
                LR.step()
            if (args.blowup_restore and optim.loss_scale == 1 and args.dynamic_loss_scale):
                iter2load = max(0, (int((total_iters)/args.save_iters)-2)*args.save_iters)
                print('Danger! Hitting blowup. Try to do blowup restore for chkpt '+str(iter2load))
                rnn_model.load_state_dict(torch.load(os.path.join(os.path.splitext(args.save)[0], 'e%s.pt'%(str(iter2load),))))
                optim.load_state_dict(torch.load(os.path.join(os.path.splitext(args.save)[0], 'optim', 'e%s.pt'%(str(iter2load),))))
                LR.step(iter2load)

        if i % args.log_interval == 0:
            log_interval = args.log_interval if i!=0 else 1
            cur_loss = total_loss.item() / log_interval
            cur_enc_loss = total_enc_loss.item() / log_interval
            cur_dec_loss = total_dec_loss.item() / log_interval
            cur_disc_enc_loss = total_enc_disc_loss.item() / (log_interval * 2.0)
            cur_disc_dec_loss = total_dec_disc_loss.item() / (log_interval * 2.0)
            cur_disc_combo_loss = total_combo_disc_loss.item() / (log_interval * 2.0)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2E} | ms/batch {:.3E} | \
                  loss_enc {:.2E} | loss_dec {:.2E} | loss_ED {:.2E} | loss_DD {:.2E} | loss_CD {:.2E} | ppl {:8.2f}'.format(
                      epoch, i, len(train_data), lr,
                      elapsed * 1000 / log_interval, cur_enc_loss, cur_dec_loss,
                      cur_disc_enc_loss, cur_disc_dec_loss, cur_disc_combo_loss, math.exp(min(cur_loss, 20)),
                      #float(args.loss_scale) if not args.fp16 else optim.loss_scale,
                  )
            )
            total_loss = 0
            total_enc_loss, total_dec_loss, total_enc_disc_loss, total_dec_disc_loss, total_combo_disc_loss = 0, 0, 0, 0, 0
            start_time = time.time()
            sys.stdout.flush()

        # save current model progress. If distributed only save from worker 0
        if args.save_iters and total_iters % (args.save_iters) == 0 and total_iters > 0 and args.rank < 1:
            if args.rank < 1:
                fname = os.path.join(os.path.splitext(args.save)[0], 'e%s.pt'%(str(total_iters),))
                print('saving model to %s' % fname)
                with open(fname, 'wb') as f:
                    sd = rnn_model.state_dict()
                    sd['rng'] = torch.get_rng_state()
                    if args.cuda:
                        sd['cuda_rng'] = torch.cuda.get_rng_state()
                    torch.save(sd, f)
                if args.save_optim:
                    optimname = os.path.join(os.path.splitext(args.save)[0], 'optim', 'e%s.pt'%(str(total_iters),))
                    with open(optimname, 'wb') as f:
                        sd = optim.state_dict()
                        torch.save(sd, f)
                with open(os.path.join(os.path.splitext(args.save)[0], 'e%s.pt'%(str(total_iters),)), 'wb') as f:
                    torch.save(model.state_dict(), f)
            if args.cuda:
                torch.cuda.synchronize()
        total_iters += 1

    return cur_loss

# Loop over epochs.
lr = args.lr
best_val_loss = None

# If saving process intermittently create directory for saving
if args.save_iters > 0:
    if not os.path.exists(os.path.splitext(args.save)[0]) and args.rank < 1:
        os.makedirs(os.path.splitext(args.save)[0])
    if not os.path.exists(os.path.join(os.path.splitext(args.save)[0], 'optim')) and args.rank < 1:
        os.makedirs(os.path.join(os.path.splitext(args.save)[0], 'optim'))

# At any point you can hit Ctrl + C to break out of training early.
try:
    total_iters = 0
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        val_loss = train(total_iters)
        total_iters += len(train_data)
        if val_data is not None:
            print('entering eval')
            val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(min(val_loss, 20))))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss and args.rank <= 0:
            with open(args.save, 'wb') as f:
                sd = rnn_model.state_dict()
                sd['rng'] = torch.get_rng_state()
                if args.cuda:
                    sd['cuda_rng'] = torch.cuda.get_rng_state()
                torch.save(sd, f)
            best_val_loss = val_loss
        if args.cuda:
            torch.cuda.synchronize()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
if os.path.exists(args.save):
    with open(args.save, 'rb') as f:
        rnn_model.load_state_dict(torch.load(f))

if not args.no_weight_norm and args.rank <= 0:
    remove_weight_norm(rnn_model)
    with open(args.save, 'wb') as f:
        torch.save(rnn_model.state_dict(), f)
if args.cuda:
    torch.cuda.synchronize()

if test_data is not None:
    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
