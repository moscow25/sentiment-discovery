import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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
                tie_weights=True, freeze=False, teacher_force=True,
                attention=False, init_transform_id=False,
                use_latent_hidden=False, transform_latent_hidden=False, latent_tanh=False,
                use_cell_hidden=False, transform_cell_hidden=False, decoder_highway_hidden=False):
        super(RNNAutoEncoderModel, self).__init__()
        self.freeze = freeze
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

    def forward(self, input, reset_mask=None, temperature=0., beam=None):
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
                cval = self.emotion_cell_state[n]
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

        emb = self.process_emb(encoder_hidden,
            use_latent_hidden=self.use_latent_hidden, transform_latent_hidden=self.transform_latent_hidden,
            use_cell_hidden=self.use_cell_hidden, transform_cell_hidden=self.transform_cell_hidden)
<<<<<<< HEAD
        decoder_output, decoder_hidden, sampled_out = self.decode_out(input, emb, reset_mask, temperature, beam)
=======
        decoder_output, decoder_hidden, sampled_out = self.decode_out(input, emb, reset_mask,
            temperature=temperature, highway_hidden=self.highway_hidden)
>>>>>>> 1b7968d657cb7ee73a4f6b107d1f75568ddb7630
        self.encoder.set_hidden([(encoder_hidden[0][0], encoder_hidden[1][0])])
        return encoder_output, decoder_output, encoder_hidden, sampled_out

    def encode_in(self, input, reset_mask=None):
        out, (hidden, cell) = self.encoder(input, reset_mask=reset_mask)
        return out, (hidden, cell)

<<<<<<< HEAD
    def decode_out(self, input, hidden_output, reset_mask=None, temperature=0, beam=None):
=======
    def decode_out(self, input, hidden_output, reset_mask=None, temperature=0, highway_hidden=False):
>>>>>>> 1b7968d657cb7ee73a4f6b107d1f75568ddb7630
        # print(hidden_output)
        # print(hidden_output[0].size())
        self.decoder.set_hidden(hidden_output)
        # NOTE: change here to remove teacher forcing
        # TODO: pass flags to use internal state (no teacher forcing)
<<<<<<< HEAD
        out, (hidden, cell), sampled_out = self.decoder(input, detach=False, reset_mask=reset_mask, context=hidden_output[0][1], temperature=temperature, beam=beam)
=======
        out, (hidden, cell), sampled_out = self.decoder(input, detach=False, reset_mask=reset_mask,
            context=hidden_output[0][1], temperature=temperature, highway_hidden=highway_hidden)
>>>>>>> 1b7968d657cb7ee73a4f6b107d1f75568ddb7630
        return out, (hidden, cell), sampled_out

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
        return sd

    def load_state_dict(self, sd, strict=True):
        self.encoder.load_state_dict(sd['encoder'], strict)
        if not self.tied:
            self.decoder.load_state_dict(sd['decoder'], strict)
        if self.latent_hidden_transform is not None and sd['hidden_transform']:
            self.latent_hidden_transform.load_state_dict(sd['hidden_transform'], strict)
        if self.latent_cell_transform is not None and sd['cell_transform']:
            self.latent_cell_transform.load_state_dict(sd['cell_transform'], strict)

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
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, all_layers=False, teacher_force=True, attention=False, transfer_model=None):
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

<<<<<<< HEAD
    def forward(self, input, reset_mask=None, detach=True, context=None, temperature=0, beam=None):
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

=======
    def forward(self, input, reset_mask=None, detach=True, context=None, temperature=0, highway_hidden=False):
>>>>>>> 1b7968d657cb7ee73a4f6b107d1f75568ddb7630
        if detach:
            #print('detach')
            self.rnn.detach_hidden()
        outs = []
        context = context.view(1,context.size(-2), context.size(-1))
        seq_len = input.size(0)
        out_txt = [input[0].squeeze()]

        # If implementing a highway -- assume we copied initial hidden state, and += it to future states
        # HACK -- Highway assumes single-layer one-directional RNN
        # NOTE -- we += hidden state... after any linear/nonlinear transformation
        if highway_hidden:
            #print('copying for highway')
            first_hidden = [self.rnn.rnns[0].hidden[0].clone(), self.rnn.rnns[0].hidden[1].clone()]
            #print(first_hidden)

        for i in range(seq_len):
            if beam is None and (self.teacher_force or i == 0):
                x = input[i]
#                print(x.size())
            else:
#                print(i)
#                x = sample(out, temperature).squeeze()
                x = sampled_out
#                print(x)
#            out_txt.append(x)
            emb = self.drop(self.encoder(x))
            emb = emb.view(-1, emb.size(-1))

            # Do no highway the first step -- already there.
            if highway_hidden and i > 0:
                #print('using highway')
                #print('copy state')
                #print(first_hidden)
                #print('current state')
                #print(self.rnn.rnns[0].hidden)
                self.rnn.rnns[0].add_hidden(first_hidden)
                #print('after copy')
                #print(self.rnn.rnns[0].hidden)

#            if i>=seq_len-2:
                #print('rnn_hook', len(self.rnn._backward_hooks))
            _, hidden = self.rnn(emb.unsqueeze(0), reset_mask=reset_mask[i] if reset_mask is not None else None)
#            if i>=seq_len-2:
#                hidden[1].register_hook(lambda x: print('hook3'))
#                print(hidden[1].size())
    
            cell = hidden[0]
            hidden = hidden[1]
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
        return outs, (cell, hidden), out_txt

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
