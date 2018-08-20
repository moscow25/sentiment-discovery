import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random

from apex import RNN
#import QRNN

def sample(out, temperature=0.1, cpu=False, beam_step=None):
    # Temperature == 0 is broken [all results in 0-32 space, no printable characters. Use low temp > 0.]
    if beam_step is None:
        if temperature == 0:
    #        print('WARNING: Temp=0 is broken. Will not return correct results')
            char_idx = torch.max(out.squeeze().data, 0)[1]
        else:
            out = out.float().squeeze().div(temperature)
            char_weights = F.softmax(out, -1).data
    #        char_weights = out.exp()
            if cpu:
                char_weights = char_weights.cpu()
            char_idx = torch.multinomial(char_weights, 1)
        return char_idx
    else:
        out = out.float().squeeze()
        if temperature > 0:
            out = out.div(temperature)
        out = F.log_softmax(out, -1)
        return beam_step(out)

def tie_params(module_src, module_dst):

    for name, p in module_src._parameters.items():
        setattr(module_dst, name, p)
    for mname, module in module_src._modules.items():
        tie_params(module, getattr(module_dst, mname))

class RNNAutoEncoderModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                tie_weights=True, freeze=False, freeze_decoder=False, teacher_force=True,
                attention=False, init_transform_id=False,
                use_latent_hidden=True, transform_latent_hidden=True, latent_tanh=False,
                use_cell_hidden=False, transform_cell_hidden=False, decoder_highway_hidden=True,
                discriminator_encoder_hidden=False, disc_enc_nhid=1024, disc_enc_layers=2, disconnect_disc_enc_grad=True,
                discriminator_decoder_hidden=False, disc_dec_nhid=1024, disc_dec_layers=2, disconnect_disc_dec_grad=True,
                combined_disc_nhid=1024, combined_disc_layers=1, disc_collect_hiddens=True,
                disc_hidden_ave_pos_factor=1.0, disc_hidden_unroll=False, disc_hidden_reduce_dim_size=256,
                disc_hidden_cnn_layers=0, disc_hidden_cnn_nfilter=128, disc_hidden_cnn_filter_size=3, disc_hidden_cnn_max_pool=True):
        super(RNNAutoEncoderModel, self).__init__()
        self.freeze = freeze
        self.freeze_decoder = freeze_decoder
        self.encoder = RNNModel(rnn_type=rnn_type, ntoken=ntoken, ninp=ninp, nhid=nhid,
            nlayers=nlayers, dropout=dropout)
        # Parameters from first to second.
        self.tied = tie_weights
        decoder = RNNDecoder(rnn_type=rnn_type, ntoken=ntoken, ninp=ninp, nhid=nhid,
            nlayers=nlayers, dropout=dropout, teacher_force=teacher_force, attention=attention,
            transfer_model=self.encoder if self.tied else None)
        if self.tied:
            object.__setattr__(self, 'decoder', decoder)
            #object.__setattr__(self, 'decoder', self.encoder)
        else:
#            decoder = RNNModel(rnn_type=rnn_type, ntoken=ntoken, ninp=ninp, nhid=nhid,
#            nlayers=nlayers, dropout=dropout)
            self.decoder = decoder

        # Do we transform the hidden state in decoder? Hidden or cell?
        self.use_latent_hidden = use_latent_hidden
        self.transform_latent_hidden = transform_latent_hidden
        self.latent_tanh = latent_tanh
        self.use_cell_hidden = use_cell_hidden
        self.transform_cell_hidden = transform_cell_hidden
        self.highway_hidden = decoder_highway_hidden

        # HACK -- ask model to predict if input string (or passed hidden state) is real or fake text
        # Model -- simple 2-layer DNN. Predict 1 = real; 0 = fake text
        # We can pass noise as fake text. Can we distinguish that? What about more natural errors?
        self.discriminator_encoder_hidden = discriminator_encoder_hidden
        self.disc_enc_nhid = disc_enc_nhid
        self.disc_enc_layers = disc_enc_layers
        self.disconnect_disc_enc_grad = disconnect_disc_enc_grad
        self.discriminator_decoder_hidden = discriminator_decoder_hidden
        self.disc_dec_nhid = disc_dec_nhid
        self.disc_dec_layers = disc_dec_layers
        self.disconnect_disc_dec_grad = disconnect_disc_dec_grad
        self.combined_disc_nhid = combined_disc_nhid
        self.combined_disc_layers = combined_disc_layers
        # Alternatively for decoder discriminator, we can collect all outputs (not just last) and
        # A. Average the states
        # B. Weighted average to encode position
        # C. Reduce dimension and cat them all
        # D. Reduce dimension and perform CNN with a maxpool
        self.disc_collect_hiddens = disc_collect_hiddens
        self.disc_hidden_ave_pos_factor = disc_hidden_ave_pos_factor
        self.disc_hidden_unroll = disc_hidden_unroll
        self.disc_hidden_reduce_dim_size = disc_hidden_reduce_dim_size
        self.disc_hidden_cnn_layers = disc_hidden_cnn_layers
        self.disc_hidden_cnn_nfilter = disc_hidden_cnn_nfilter
        self.disc_hidden_cnn_filter_size = disc_hidden_cnn_filter_size
        self.disc_hidden_cnn_max_pool = disc_hidden_cnn_max_pool

        if self.discriminator_encoder_hidden:
            print('Building discriminator from encoder hidden state (%d x %d)' % (self.disc_enc_nhid, self.disc_enc_layers))
            assert self.disc_enc_layers >= 1 and self.disc_enc_layers <= 3, "Hardcode 1-3 layer DNN for encoder discriminator"
            fc1_disc_enc = nn.Linear(nhid, self.disc_enc_nhid)
            if self.disc_enc_layers >= 2:
                fc2_disc_enc = nn.Linear(self.disc_enc_nhid, self.disc_enc_nhid)
            if self.disc_enc_layers >= 3:
                fc3_disc_enc = nn.Linear(self.disc_enc_nhid, self.disc_enc_nhid)
            disc_enc_final = nn.Linear(self.disc_enc_nhid, 1)
            if self.disc_enc_layers == 1:
                self.disc_enc_transform = nn.Sequential(fc1_disc_enc, nn.ReLU(), disc_enc_final)
                self.disc_enc_partial_transform = nn.Sequential(fc1_disc_enc, nn.ReLU())
            elif self.disc_enc_layers == 2:
                self.disc_enc_transform = nn.Sequential(fc1_disc_enc, nn.ReLU(),
                    fc2_disc_enc, nn.ReLU(), disc_enc_final)
                self.disc_enc_partial_transform = nn.Sequential(fc1_disc_enc, nn.ReLU(),
                    fc2_disc_enc, nn.ReLU())
            elif self.disc_enc_layers == 3:
                self.disc_enc_transform = nn.Sequential(fc1_disc_enc, nn.ReLU(),
                    fc2_disc_enc, nn.ReLU(), fc3_disc_enc, nn.ReLU(), disc_enc_final)
                self.disc_enc_partial_transform = nn.Sequential(fc1_disc_enc, nn.ReLU(),
                    fc2_disc_enc, nn.ReLU(), fc3_disc_enc, nn.ReLU())
            print(self.disc_enc_transform)
        else:
            self.disc_enc_transform = None
            self.disc_enc_partial_transform = None

        # Extra transformations for "collect all hiddens" from decoder and combine them somehow
        if self.disc_collect_hiddens and self.disc_hidden_reduce_dim_size:
            self.decoder_hiddens_disc_transform = nn.Linear(nhid, self.disc_hidden_reduce_dim_size)
        else:
            self.decoder_hiddens_disc_transform = None

        # CNN network, upon request
        # Makes sense to run CNN over the sequence, and take max pool -- look or discrepancies in character outputs?
        self.conv_layers = []
        if self.disc_hidden_cnn_layers:
            for layer_id in range(self.disc_hidden_cnn_layers):
                in_channels = self.disc_hidden_reduce_dim_size if layer_id == 0 else self.disc_hidden_cnn_nfilter
                out_channels = self.disc_hidden_cnn_nfilter
                conv = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=self.disc_hidden_cnn_filter_size,
                                padding=int(self.disc_hidden_cnn_filter_size/2))
                self.conv_layers.append(conv)
            self.conv_layers = nn.ModuleList(self.conv_layers)
            # We apply the discriminator fully connected network on top of this output -- just num_channels (with a maxpool)
            # TODO: Allow channel blowup factor, if multiple CNN layers (64, 128, etc)
            # TODO: Allow more different CNN filter widths, if not a deep network
            self.cnn_out_channels = self.disc_hidden_cnn_nfilter
            print(self.conv_layers)
        else:
            self.conv_layers = None
            self.cnn_out_channels = 0

        if self.discriminator_decoder_hidden:
            print('Building discriminator from *decoder* hidden state (%d x %d)' % (self.disc_dec_nhid, self.disc_dec_layers))
            assert self.disc_dec_layers >= 1 and self.disc_dec_layers <= 3, "Hardcode 1-3 layer DNN for decoder discriminator"
            hid_input = self.cnn_out_channels if self.cnn_out_channels else nhid
            fc1_disc_dec = nn.Linear(hid_input, self.disc_dec_nhid)
            if self.disc_dec_layers >= 2:
                fc2_disc_dec = nn.Linear(self.disc_dec_nhid, self.disc_dec_nhid)
            if self.disc_dec_layers >= 3:
                fc3_disc_dec = nn.Linear(self.disc_dec_nhid, self.disc_dec_nhid)
            disc_dec_final = nn.Linear(self.disc_dec_nhid, 1)
            if self.disc_dec_layers == 1:
                self.disc_dec_transform = nn.Sequential(fc1_disc_dec, nn.ReLU(), disc_dec_final)
                self.disc_dec_partial_transform = nn.Sequential(fc1_disc_dec, nn.ReLU())
            elif self.disc_dec_layers == 2:
                self.disc_dec_transform = nn.Sequential(fc1_disc_dec, nn.ReLU(),
                    fc2_disc_dec, nn.ReLU(), disc_dec_final)
                self.disc_dec_partial_transform = nn.Sequential(fc1_disc_dec, nn.ReLU(),
                    fc2_disc_dec, nn.ReLU())
            elif self.disc_dec_layers == 3:
                self.disc_dec_transform = nn.Sequential(fc1_disc_dec, nn.ReLU(),
                    fc2_disc_dec, nn.ReLU(), fc3_disc_dec, nn.ReLU(), disc_dec_final)
                self.disc_dec_partial_transform = nn.Sequential(fc1_disc_dec, nn.ReLU(),
                    fc2_disc_dec, nn.ReLU(), fc3_disc_dec, nn.ReLU())
            print(self.disc_dec_transform)
        else:
            self.disc_dec_transform = None
            self.disc_dec_partial_transform = None

        # Perhaps too clever? But combine encoder and decoder outputs into 1-layer DNN for shared real/fake prediction
        if self.discriminator_encoder_hidden and self.discriminator_decoder_hidden and self.combined_disc_layers > 0:
            # If both encoder and decoder based discriminators exist, cat outputs and run 1x layer
            print('Building *combined* discriminator from encoder & decoder hidden state. (%d x %d)', (self.combined_disc_layers, self.combined_disc_nhid))
            assert self.combined_disc_layers >= 1 and self.combined_disc_layers <= 2, "Hardcode 1-2 layer DNN for combined discriminator"
            fc1_disc_combo = nn.Linear(self.disc_enc_nhid + self.disc_dec_nhid, self.combined_disc_nhid)
            if self.combined_disc_layers >=2:
                fc2_disc_combo = nn.Linear(self.combined_disc_nhid, self.combined_disc_nhid)
            disc_combo_final = nn.Linear(self.combined_disc_nhid, 1)
            if self.combined_disc_layers == 1:
                self.disc_combo_transform = nn.Sequential(fc1_disc_combo, nn.ReLU(), disc_combo_final)
            elif self.combined_disc_layers == 2:
                self.disc_combo_transform = nn.Sequential(fc1_disc_combo, nn.ReLU(),
                    fc2_disc_combo, nn.ReLU(), disc_combo_final)
            print(self.disc_combo_transform)
        else:
            self.disc_combo_transform = None

        if self.highway_hidden:
            print('Using highway (+=) -- from encoder hidden to each decoder step')

        # Transform final hidden state (TanH so that it's in bounds)
        # Option -- initialize hidden transfor to something like the identity matrix -- tricky with Tanh() after
        if init_transform_id:
            linear_hidden = nn.Linear(nhid, nhid)
            linear_hidden.weight.data.copy_(torch.eye(nhid) * 2.0)
            self.latent_hidden_transform = nn.Sequential(linear_hidden, nn.Tanh())
        else:
            # Else initialize with random weights
            if self.latent_tanh:
                print('Using tanh() nonlinearity on decoder latent')
                self.latent_hidden_transform = nn.Sequential(nn.Linear(nhid, nhid), nn.Tanh())
            else:
                print('*Not* using tanh() on decoder latent')
                self.latent_hidden_transform = nn.Linear(nhid, nhid)
        if not self.use_latent_hidden or not self.transform_latent_hidden:
            print('Latent hidden created but not used')
            self.latent_hidden_transform = None
        # Transform cell state [maybe not necessary]
        self.latent_cell_transform = nn.Linear(nhid, nhid)
        if init_transform_id:
            self.latent_cell_transform.weight.data.copy_(torch.eye(nhid))
        if not self.use_cell_hidden or not self.transform_cell_hidden:
            print('Latent cell created but not used')
            self.latent_cell_transform = None

        # If we overwrite state [from emotion, etc] -- values set from outside the house
        self.emotion_neurons = None
        self.emotion_hidden_state = None
        self.emotion_cell_state = None
        # Boost promotion -- or turn off with 0.0
        self.hidden_boost_factor = 1.0
        self.cell_boost_factor = 0.0
        self.average_cell_value = False # True # average, instead of replace?
        # Externally -- can also push vector simply to add to hidden state [for translation in pre-computed dimension]
        self.use_added_hidden_state = False
        self.added_hidden_state_vector = None
        self.use_added_cell_state = False
        self.added_cell_state_vector = None

        # For discriminator -- reset entire hidden state batch [before transformer and decoder]
        self.reset_hidden_state = False
        self.reset_hidden_state_value = None

    def forward(self, input, reset_mask=None, temperature=0., beam=None, variable_tf=0.):
        if self.freeze:
            with torch.no_grad():
                encoder_output, encoder_hidden = self.encode_in(input, reset_mask)
        else:
            encoder_output, encoder_hidden = self.encode_in(input, reset_mask)

        #print('encoder hidden:')
        #print(encoder_hidden)

        # If we want to manipulate neurons [from outside, N=1, etc]
        if self.emotion_neurons:
            #print('-------\nChanging cells: %s' % self.emotion_neurons)
            for n in self.emotion_neurons:
                hval = self.emotion_hidden_state[n]
                #cval = self.emotion_cell_state[n]
                if self.hidden_boost_factor != 0.0:
                    v = hval
                    if self.average_cell_value:
                        v = (v * self.hidden_boost_factor) + encoder_hidden[0][0][0][n] * (1.0 - self.hidden_boost_factor)
                        #v = (v + encoder_hidden[0][0][0][n]) / 2.0
                    else:
                        v = v * self.hidden_boost_factor
                    encoder_hidden[0][0][0][n] = v
                if self.cell_boost_factor != 0.0:
                    encoder_hidden[1][0][0][n] = cval * self.cell_boost_factor
        # If we want to directly add to thes hidden state/cell state vector
        if self.use_added_hidden_state:
            print('Adding hidden state vector')
            #print(self.added_hidden_state_vector)
            encoder_hidden[0][0][0] += self.added_hidden_state_vector
        if self.use_added_cell_state:
            print('Adding cell state vector')
            #print(self.added_cell_state_vector)
            encoder_hidden[1][0][0] += self.added_cell_state_vector

        # Reset whole encoder batch upon request -- for discriminator training
        if self.reset_hidden_state:
            encoder_hidden[0][0] = self.reset_hidden_state_value

        # If we predict real/fake text based on encoder hidden state, apply that here.
        # NOTE: If we manipulate encoder hidden state, do that first above.
        if self.discriminator_encoder_hidden:
            enc_hid = encoder_hidden[0][0]
            # We most likely don't want to backprop encoder from discriminator here. But we can.
            # Makes discrimination easier, but decoder has too many constaints...
            if self.disconnect_disc_enc_grad:
                enc_hid = enc_hid.detach()
            encoder_hidden_disc_out = self.disc_enc_transform(enc_hid)

            # Also compute partial for combined if needed
            if self.disc_combo_transform:
                encoder_hidden_disc_partial = self.disc_enc_partial_transform(enc_hid)
        else:
            encoder_hidden_disc_out = None

        emb = self.process_emb(encoder_hidden,
            use_latent_hidden=self.use_latent_hidden, transform_latent_hidden=self.transform_latent_hidden,
            use_cell_hidden=self.use_cell_hidden, transform_cell_hidden=self.transform_cell_hidden)
        if self.freeze_decoder:
            with torch.no_grad():
                decoder_output, decoder_hidden, sampled_out, all_hiddens = self.decode_out(input, emb, reset_mask,
                    temperature=temperature, highway_hidden=self.highway_hidden, beam=beam, variable_tf=variable_tf,
                    collect_hiddens=self.disc_collect_hiddens)
        else:
            decoder_output, decoder_hidden, sampled_out, all_hiddens = self.decode_out(input, emb, reset_mask,
                temperature=temperature, highway_hidden=self.highway_hidden, beam=beam, variable_tf=variable_tf,
                collect_hiddens=self.disc_collect_hiddens)

        # If we collect all decoder hidden states... what to do with it?
        # A. Last
        # B. Sum them all
        # C. Weighted sum -- encodes order
        # D. Cat the whole thing -- very long [might be good to reduce the dimension first in each step]
        if self.disc_collect_hiddens:
            # TODO: If we cat all hidden states... good to apply a single dim-reduction to all steps first.
            average_hiddens = torch.stack(all_hiddens).squeeze()
            # TODO: Apply position-encoding mask here [0.9^len, etc]
            seq_len = input.size(0)
            average_hiddens = torch.sum(average_hiddens, dim=0) / seq_len

        # Alternatively, don't sum/average the hidden states. Compress them and then cat to large input.
        # TODO: Perhaps we should CNN/RNN the small states? Would make more sense. But can a simple DNN find it? If information in the decoder
        if self.disc_collect_hiddens and self.decoder_hiddens_disc_transform:
            compressed_hiddens = [self.decoder_hiddens_disc_transform(h) for h in all_hiddens]
            # Now we just cat these results.
            # TODO: Keep em, and run a CNN/RNN over the results? CNN with max pool makes the most sense IMO
            #print(compressed_hiddens)
            #cat_hiddens = torch.stack(compressed_hiddens).squeeze()
            #print(cat_hiddens)

        # CNN over the (reduced size) decoder hidden state embeddings
        if self.cnn_out_channels:
            tmp_conv_maps = []
            layer_in = torch.stack(compressed_hiddens).squeeze()
            #print(layer_in)
            # Set batch as the first dimension
            layer_in = torch.transpose(layer_in, 0, 1)

            # Transpose dimensions, for 1D convolution
            layer_in = torch.transpose(layer_in, 1, 2)
            for layer_idx in range(self.disc_hidden_cnn_layers):
                #print('CNN layer %d' % layer_idx)
                if layer_idx > 0:
                    layer_in = enc_
                    # Zero out lower layer outputs, unless we connect all to output
                    #if True:
                    #    tmp_conv_maps = []
                enc_ = self.conv_layers[layer_idx](layer_in)
                enc_ = F.relu(enc_)
                tmp_conv_maps.append(enc_)

            # Now do a max pool
            #final_layer[i].append(F.max_pool1d(cmap, cmap.size(2)).squeeze(2))
            #print(tmp_conv_maps[-1])
            cnn_output = F.max_pool1d(tmp_conv_maps[-1], tmp_conv_maps[-1].size(2)).squeeze(2)
        else:
            # If we cat results -- kind of crazy
            cat_hiddens = torch.cat(compressed_hiddens, dim=0).squeeze()
            #print(cat_hiddens)


        # If we want to also predict real/fake from the decoder (final) hidden state, apply that here.
        # NOTE: overall this will be harder, and we can't (easily) freeze loss w/r/t decoder
        # But unrolled encoder should know some info like whether words were complete and made sense.
        if self.discriminator_decoder_hidden:
            #dec_hid = decoder_hidden[0][0]
            # If asked to collect all hiddens, use that average as input to the network -- rather than just last state.
            if self.disc_collect_hiddens:
                if self.decoder_hiddens_disc_transform:
                    if self.cnn_out_channels:
                        #print('using cnn channels')
                        dec_hid = cnn_output
                    else:
                        #print('using cat channels')
                        dec_hid = cat_hiddens
                else:
                    #print('using average hiddens')
                    dec_hid = average_hiddens
            else:
                dec_hid = decoder_hidden[0][0]

            if self.disconnect_disc_dec_grad:
                dec_hid = dec_hid.detach()
            decoder_hidden_disc_out = self.disc_dec_transform(dec_hid)

            # Also compute partial for combined if needed
            if self.disc_combo_transform:
                decoder_hidden_disc_partial = self.disc_dec_partial_transform(dec_hid)
        else:
            decoder_hidden_disc_out = None

        # TODO: As the (likely) best model, combine final encoder and decoder based discriminator layers
        # Why? Encoder better at some, decoder better at other discriminator tasks. Just print it.
        if self.disc_combo_transform:
            combo_input = torch.cat((encoder_hidden_disc_partial, decoder_hidden_disc_partial), 1)
            combo_disc_out = self.disc_combo_transform(combo_input)
        else:
            combo_disc_out = None

        # Reset encoer Hidden State... for the next step.
        self.encoder.set_hidden([(encoder_hidden[0][0], encoder_hidden[1][0])])
        return encoder_output, decoder_output, encoder_hidden, encoder_hidden_disc_out, decoder_hidden_disc_out, combo_disc_out, sampled_out

    def encode_in(self, input, reset_mask=None):
        out, (hidden, cell) = self.encoder(input, reset_mask=reset_mask)
        return out, (hidden, cell)

    def decode_out(self, input, hidden_output, reset_mask=None, temperature=0, highway_hidden=False, beam=None, variable_tf=0., collect_hiddens=False):
        # print(hidden_output)
        # print(hidden_output[0].size())
        self.decoder.set_hidden(hidden_output)
        # NOTE: change here to remove teacher forcing
        # TODO: pass flags to use internal state (no teacher forcing)
        out, (hidden, cell), sampled_out, all_hiddens = self.decoder(input, detach=False, reset_mask=reset_mask, context=hidden_output[0][1],
            temperature=temperature, highway_hidden=highway_hidden, beam=beam, variable_tf=variable_tf, collect_hiddens=collect_hiddens)
        return out, (hidden, cell), sampled_out, all_hiddens

    # placeholder
    def process_emb(self, emb, use_latent_hidden=True, transform_latent_hidden=True, use_cell_hidden=True, transform_cell_hidden=True):
        # Hidden state for RNN-layer0; cell state for RNN-layer0
        # [Transpose, of a sort]
        #emb = [[emb[0][0], emb[1][0]]]
        # Break out all possible choices -- do we pass cell and/or hidden state? Do we add trainable transform?
        # Why?? These transforms help for decoding. But potentially not necessary, and not good for style transfer.
        if use_latent_hidden:
            hid = emb[0][0]
        else:
            hid = torch.zeros_like(emb[0][0])
        if transform_latent_hidden:
            hid = self.latent_hidden_transform(hid)
        if use_cell_hidden:
            cell = emb[1][0]
        else:
            cell = torch.zeros_like(emb[1][0])
        if transform_cell_hidden:
            cell = self.latent_cell_transform(cell)

        #emb = [[self.latent_hidden_transform(emb[0][0]), self.latent_cell_transform(emb[1][0])]]
        #print((hid, cell))
        return [[hid, cell]]

    def get_text_from_outputs(self, out, temperature=0):
        """
        autoencoder = RNNAutoEncoder(...)
        out = autoencoder(batch)
        encoder_text, decoder_text = autoencoder.get_text_from_outputs(out)
        # each consists of batch_size number of strings with the text produced from the model
        """
        encoder_outs = out[0]
        decoder_outs = out[1]
        batch_size = encoder_outs.size(1)
        seq_len = encoder_outs.size(0)
        encoder_text = ['']*batch_size
        decoder_text = ['']*batch_size
        for t in range(seq_len):
            encoder_chars = sample(encoder_outs[t], temperature=temperature, cpu=True)
            decoder_chars = sample(decoder_outs[t], temperature=temperature, cpu=True)
            for b in range(batch_size):
                encoder_text[b] += chr(encoder_chars[b])
                decoder_text[b] += chr(decoder_chars[b])
        return encoder_text, decoder_text

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = {}
        if destination is not None:
            sd = destination
        sd['encoder'] = self.encoder.state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.tied:
            sd['decoder'] = sd['encoder']
        else:
            sd['decoder'] = self.decoder.state_dict(prefix=prefix, keep_vars=keep_vars)
        if self.use_latent_hidden and self.transform_latent_hidden:
            sd['hidden_transform'] = self.latent_hidden_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
        else:
            sd['hidden_transform'] = {}
        if self.use_cell_hidden and self.transform_cell_hidden:
            sd['cell_transform'] = self.latent_cell_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
        else:
            sd['cell_transform'] = {}
        # Discriminator transformations
        if self.discriminator_encoder_hidden:
            sd['disc_enc_transform'] = self.disc_enc_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
            sd['disc_enc_partial_transform'] = self.disc_enc_partial_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
        else:
            sd['disc_enc_transform'] = {}
            sd['disc_enc_partial_transform'] = {}
        if self.discriminator_decoder_hidden:
            sd['disc_dec_transform'] = self.disc_dec_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
            sd['disc_dec_partial_transform'] = self.disc_dec_partial_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
        else:
            sd['disc_dec_transform'] = {}
            sd['disc_dec_partial_transform'] = {}
        if self.disc_combo_transform:
            sd['disc_combo_transform'] = self.disc_combo_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
        else:
            sd['disc_combo_transform'] = {}
        # Collect hiddens (aggregate hidden by position, not just last) in discriminator if used
        if self.decoder_hiddens_disc_transform:
            sd['decoder_hiddens_disc_transform'] = self.decoder_hiddens_disc_transform.state_dict(prefix=prefix, keep_vars=keep_vars)
        else:
            sd['decoder_hiddens_disc_transform'] = {}
        # CNN in discriminator if used
        if self.conv_layers:
            sd['conv_layers'] = self.conv_layers.state_dict(prefix=prefix, keep_vars=keep_vars)
        else:
            sd['conv_layers'] = {}
        return sd

    def load_state_dict(self, sd, strict=True):
        self.encoder.load_state_dict(sd['encoder'], strict)
        if not self.tied:
            self.decoder.load_state_dict(sd['decoder'], strict)
        if self.latent_hidden_transform is not None and sd['hidden_transform']:
            self.latent_hidden_transform.load_state_dict(sd['hidden_transform'], strict)
        if self.latent_cell_transform is not None and sd['cell_transform']:
            self.latent_cell_transform.load_state_dict(sd['cell_transform'], strict)
        # Load discriminator elements -- if requeseted and exist in the loading model [keep random init if loading for older model]
        if self.disc_enc_transform is not None and 'disc_enc_transform' in sd and sd['disc_enc_transform']:
            self.disc_enc_transform.load_state_dict(sd['disc_enc_transform'], strict)
        if self.disc_enc_partial_transform is not None and 'disc_enc_partial_transform' in sd and sd['disc_enc_partial_transform']:
            self.disc_enc_partial_transform.load_state_dict(sd['disc_enc_partial_transform'], strict)
        if self.disc_dec_transform is not None and 'disc_dec_transform' in sd and sd['disc_dec_transform']:
            self.disc_dec_transform.load_state_dict(sd['disc_dec_transform'], strict)
        if self.disc_dec_partial_transform is not None and 'disc_dec_partial_transform' in sd and sd['disc_dec_partial_transform']:
            self.disc_dec_partial_transform.load_state_dict(sd['disc_dec_partial_transform'], strict)
        if self.disc_combo_transform is not None and 'disc_combo_transform' in sd and sd['disc_combo_transform']:
            self.disc_combo_transform.load_state_dict(sd['disc_combo_transform'], strict)
        # Optional discriminator features -- if we collect hiddens or use a CNN
        if self.decoder_hiddens_disc_transform is not None and 'decoder_hiddens_disc_transform' in sd and sd['decoder_hiddens_disc_transform']:
            self.decoder_hiddens_disc_transform.load_state_dict(sd['decoder_hiddens_disc_transform'], strict)
        if self.conv_layers is not None and 'conv_layers' in sd and sd['conv_layers']:
            self.conv_layers.load_state_dict(sd['conv_layers'], strict)

# Placeholder QRNN wrapper -- to support detach/reset/init RNN state
#class myQRNN(QRNN):
#    def __init__(self, *input, **kwargs):
#        super(myQRNN, self).__init__(*input, **kwargs)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)
        #self.rnn=getattr(myQRNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.rnn=getattr(RNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.decoder.bias.data.fill_(0)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, reset_mask=None, detach=True):
        emb = self.drop(self.encoder(input))
        if detach:
            self.rnn.detach_hidden()
        # if teacher forcing off == swap here [to RNNFeaturizer...]
        output, hidden = self.rnn(emb, reset_mask=reset_mask)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = destination if destination is not None else {}
        sd['encoder'] = self.encoder.state_dict(prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(prefix=prefix, keep_vars=keep_vars)
        sd = {'encoder': sd}
        sd['decoder'] = self.decoder.state_dict(prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        if 'decoder' in state_dict:
            self.decoder.load_state_dict(state_dict['decoder'], strict=strict)
        self.encoder.load_state_dict(state_dict['encoder']['encoder'], strict=strict)
        self.rnn.load_state_dict(state_dict['encoder']['rnn'], strict=strict)

    def set_hidden(self, hidden):
        self.rnn.set_hidden(hidden)

class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, all_layers=False,
        teacher_force=True, attention=False, transfer_model=None):
        super(RNNDecoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        if transfer_model is None:
            self.encoder = nn.Embedding(ntoken, ninp)
            self.rnn = getattr(RNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)
        else:
            self.encoder = transfer_model.encoder
            self.rnn = transfer_model.rnn
            self.decoder = transfer_model.decoder

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.all_layers = all_layers
        self.output_size = self.nhid if not self.all_layers else self.nhid * self.nlayers

        self.attention = attention

        self.teacher_force = teacher_force

    def forward(self, input, reset_mask=None, detach=True, context=None, temperature=0, highway_hidden=False, beam=None, variable_tf=0., collect_hiddens=False):
        """reset_mask and beam currently do not work together"""
        # TODO: init beam
        batch_size = input.size(1)
        if beam is not None:
            hidden_init = self.rnn.get_hidden()
            if context is not None:
                hidden_init = (hidden_init, context)
            sampled_out, hidden_init = beam.reset_beam_decoder(batch_size, hidden_init, input[0])
            if context is not None:
                hidden_init, context = hidden_init
            self.rnn.set_hidden(hidden_init)
            del hidden_init

        if detach:
            self.rnn.detach_hidden()
        outs = []
        context = context.view(1,context.size(-2), context.size(-1))
        seq_len = input.size(0)
        out_txt = [input[0].squeeze()]

        # If implementing a highway -- assume we copied initial hidden state, and += it to future states
        # HACK -- Highway assumes single-layer one-directional RNN
        # NOTE -- we += hidden state... after any linear/nonlinear transformation
        if highway_hidden:
            first_hidden = [self.rnn.rnns[0].hidden[0].clone(), self.rnn.rnns[0].hidden[1].clone()]

        # Optionally, collect all hidden states (each step) to pass back for discriminator
        # NOTE: We can get also this directly via "collectHidden" as well -- but then have to change other logic.
        all_hiddens = []

        for i in range(seq_len):
            # Use the next (true) character if teacher forcing
            # Flip variable control coin here (since it's per-char) [and forced_control over-rides self.teacher_force]
            # NOTE: This should be simpler -- backward compatible so that variable_tf=0.2 --> 20% we use generated character
            variable_control_coin = (random.random() < variable_tf)
            if beam is None and ((self.teacher_force and not variable_control_coin) or i == 0):
                x = input[i]
            else:
                x = sampled_out

            emb = self.drop(self.encoder(x))
            emb = emb.view(-1, emb.size(-1))

            # Do no highway the first step -- already there.
            if highway_hidden and i > 0:
                self.rnn.rnns[0].add_hidden(first_hidden)

            # Execute the RNN step
            _, hidden = self.rnn(emb.unsqueeze(0), reset_mask=reset_mask[i] if reset_mask is not None else None)

            cell = hidden[0]
            hidden = hidden[1]

            # Add to list of hiddens to return at the end.
            if collect_hiddens:
                all_hiddens.append(hidden.clone())

            decoder_in = hidden
            #decoder_in = decoder_in.view(-1, self.nhid).contiguous()
            if self.attention:
                assert context is not None
                decoder_in = decoder_in * context
            out = self.decoder(decoder_in)
            outs.append(out)

            sampled_out = sample(out, temperature, beam_step=None if beam is None else lambda x: beam.step(x, self.rnn.get_hidden())).squeeze()
            out_txt.append(sampled_out.data)

#        print([x.size() for x in out_txt])
        if beam is not None:
            out_txt = beam.get_hyp()
        else:
            out_txt = torch.stack(out_txt)
        outs = torch.cat(outs)
#        outs = torch.stack(outs)
#        outs.register_hook(lambda x:print('hooked'))
        return outs, (cell, hidden), out_txt, all_hiddens

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = destination if destination is not None else {}
        sd['encoder'] = self.encoder.state_dict(prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(prefix=prefix, keep_vars=keep_vars)
        sd = {'encoder': sd}
        sd['decoder'] = self.decoder.state_dict(prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        if 'decoder' in state_dict:
            self.decoder.load_state_dict(state_dict['decoder'], strict=strict)
        self.encoder.load_state_dict(state_dict['encoder']['encoder'], strict=strict)
        self.rnn.load_state_dict(state_dict['encoder']['rnn'], strict=strict)

    def set_hidden(self, hidden):
        self.rnn.set_hidden(hidden)

class RNNFeaturizer(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, all_layers=False):
        super(RNNFeaturizer, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn=getattr(RNN, rnn_type)(ninp, nhid, nlayers, dropout=dropout)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.all_layers = all_layers
        self.output_size = self.nhid if not self.all_layers else self.nhid * self.nlayers

    def forward(self, input, seq_len=None, get_hidden=False):
        self.rnn.detach_hidden()
        if seq_len is None:
            for i in range(input.size(0)):
                emb = self.drop(self.encoder(input[i]))
                _, hidden = self.rnn(emb.unsqueeze(0), collectHidden=True)
            cell = self.get_features(hidden)
            if get_hidden:
                cell = (self.get_features(hidden, get_hidden=True), cell)
        else:
            last_cell = 0
            last_hidden = 0
            for i in range(input.size(0)):
                emb = self.drop(self.encoder(input[i]))
                _, hidden = self.rnn(emb.unsqueeze(0), collectHidden=True)
                cell = self.get_features(hidden, get_hidden=False)
                if get_hidden:
                    hidden = self.get_features(hidden, get_hidden=True)
                if i > 0:
                    cell = get_valid_outs(i, seq_len, cell, last_cell)
                    if get_hidden:
                        hidden = get_valid_outs(i, seq_len, hidden, last_hidden)
                last_cell = cell
                if get_hidden:
                    last_hidden = hidden
            if get_hidden:
                cell = (hidden, cell)

        return cell

    # Passed a pair of hidden, cell -- choose the right one and get features (final values)
    def get_features(self, hidden, get_hidden=False):
        if get_hidden:
            cell = hidden[0]
        else:
            cell = hidden[1]
        #get cell state from layers
        if self.all_layers:
            cell = torch.cat(cell, -1)
        else:
            cell = cell[-1]
        return cell[-1]


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = destination if destination is not None else {}
        sd['encoder'] = self.encoder.state_dict(prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # For AutoEncoder module, need extra "encoder" path
        # TODO: Automate this switch.
        if 'encoder' in state_dict['encoder'].keys():
            self.encoder.load_state_dict(state_dict['encoder']['encoder'], strict=strict)
            self.rnn.load_state_dict(state_dict['encoder']['rnn'], strict=strict)
        else:
            self.encoder.load_state_dict(state_dict['encoder'], strict=strict)
            self.rnn.load_state_dict(state_dict['rnn'], strict=strict)

def get_valid_outs(timestep, seq_len, out, last_out):
    invalid_steps = timestep >= seq_len
    if (invalid_steps.long().sum() == 0):
        return out
    return selector_circuit(out, last_out, invalid_steps)

def selector_circuit(val0, val1, selections):
    selections = selections.type_as(val0.data).view(-1, 1).contiguous()
    return (val0*(1-selections)) + (val1*selections)
