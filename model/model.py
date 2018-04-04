import torch
import torch.nn as nn
from torch.autograd import Variable

from apex import RNN
#import QRNN

def sample(out, temperature=0.1):
    if temperature == 0:
        char_idx = torch.max(out.squeeze().data, 0)[1]
    else:
        char_weights = out.float().squeeze().data.div(temperature).exp().cpu()
        char_idx = torch.multinomial(char_weights, 1)
    return char_idx

def tie_params(module_src, module_dst):
    
    for name, p in module_src._parameters.items():
        setattr(module_dst, name, p)
    for mname, module in module_src._modules.items():
        tie_params(module, getattr(module_dst, mname))

class RNNAutoEncoderModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=True):
        super(RNNAutoEncoderModel, self).__init__()
        self.encoder = RNNModel(rnn_type=rnn_type, ntoken=ntoken, ninp=ninp, nhid=nhid,
            nlayers=nlayers, dropout=dropout)
        self.decoder = RNNModel(rnn_type=rnn_type, ntoken=ntoken, ninp=ninp, nhid=nhid,
            nlayers=nlayers, dropout=dropout)
        # Parameters from first to second.
        if tie_weights:
            tie_params(self.encoder, self.decoder)

        # Transform final hidden state (TanH so that it's in bounds)
        self.latent_hidden_transform = nn.Sequential(nn.Linear(nhid, nhid), nn.Tanh())
        # Transform cell state [maybe not necessary]
        self.latent_cell_transform = nn.Linear(nhid, nhid)

    def forward(self, input, reset_mask=None):
        encoder_output, emb = self.encode_in(input)
        emb = self.process_emb(emb)
        decoder_output, decoder_hidden = self.decode_out(input, emb)
        return encoder_output, decoder_output

    def encode_in(self, input):
        out, (hidden, cell) = self.encoder(input)
        return out, (hidden, cell)

    def decode_out(self, input, hidden_output):
        # print(hidden_output)
        # print(hidden_output[0].size())
        self.decoder.set_hidden(hidden_output)
        # NOTE: change here to remove teacher forcing
        # TODO: pass flags to use internal state (no teacher forcing)
        out, (hidden, cell) = self.decoder(input, detach=False)
        return out, (hidden, cell)

    # placeholder
    def process_emb(self, emb):
        # Hidden state for RNN-layer0; cell state for RNN-layer0
        # [Transpose, of a sort]
        #emb = [[emb[0][0], emb[1][0]]]
        # TODO -- add possible transformation?
        emb = [[self.latent_hidden_transform(emb[0][0]), self.latent_cell_transform(emb[1][0])]]
        return emb

    def get_text_from_outputs(self, out):
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
            encoder_chars = sample(encoder_outs[t])
            decoder_chars = sample(decoder_outs[t])
            for b in range(batch_size):
                encoder_text[b] += chr(encoder_chars[b])
                decoder_text[b] += chr(decoder_chars[b])
        return encoder_text, decoder_text

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
        sd = {}
        sd['encoder'] = self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd = {'encoder': sd}
        sd['decoder'] = self.decoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
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

    def forward(self, input, seq_len=None):
        self.rnn.detach_hidden()
        if seq_len is None:
            for i in range(input.size(0)):
                emb = self.drop(self.encoder(input[i]))
                _, hidden = self.rnn(emb.unsqueeze(0), collectHidden=True)
            cell = self.get_cell_features(hidden)
        else:
            last_cell = 0
            for i in range(input.size(0)):
                # encodes next character [teacher forcing]
                emb = self.drop(self.encoder(input[i]))
                # TODO: Run RNN on next gen character?
                logits, hidden = self.rnn(emb.unsqueeze(0), collectHidden=True)
                cell = self.get_cell_features(hidden)
                if i > 0:
                    cell = get_valid_outs(i, seq_len, cell, last_cell)
                last_cell = cell
                # Call sample(logits, temp)
                # current_ch = max(logits, -1)
                # Instead of input[i]
        return cell

    def get_cell_features(self, hidden):
        cell = hidden[1]
        #get cell state from layers
        if self.all_layers:
            cell = torch.cat(cell, -1)
        else:
            cell = cell[-1]
        return cell[-1]


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = {}
        sd['encoder'] = self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd['rnn'] = self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return sd

    def load_state_dict(self, state_dict, strict=True):
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
