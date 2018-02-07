import torch
import torch.nn as nn
from torch.autograd import Variable

# Creative dropouts -- from Salesforce/MoS repos
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        #m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        m = x.data.new(x.size(0), x.size(1)).bernoulli_(1 - dropout)
        # mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x




class StackedLSTM(nn.Module):
	"""
	Based on implementation of StackedLSTM from openNMT-py
	https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/StackedRNN.py
	Args:
		cell: LSTM/mLSTM cell u want to create. Callable of form `f(input, rnn_size)`
		num_layers: how many of these cells to stack
		input: The dimension of the input to the module
		rnn_size: The number of features in the hidden states of the lstm cells
		output_size: dimension of linear transformation layer on the output of the stacked rnn cells.
			If <=0 then no output layer is applied. Default: -1
		drop_out: probability of dropout layer (applied after rnn, but before output layer). Default: 0
		bias: If `False`, then the layer does not use bias weights for the linear transformation from
			multiplicative->hidden. Default: True
	Inputs: input, (h_0, c_0)
		- **input** (batch, input_size): tensor containing input features
		- **h_0** (num_layers, batch, hidden_size): tensor containing the initial hidden
		  state for each layer of each element in the batch.
		- **c_0** (num_layers, batch, hidden_size): tensor containing the initial cell state
		  for each layer of each element in the batch.
	Outputs: (h_1, c_1), output
		- **h_1** (num_layers, batch, hidden_size): tensor containing the next hidden state
		  for each layer of each element in the batch
		- **c_1** (num_layers, batch, hidden_size): tensor containing the next cell state
		  for each layer of each element in the batch
		- **output** (batch, output_size): tensor containing output of stacked rnn.
			If `output_size==-1` then this is equivalent to `h_1`.
	Examples:
		>>> rnn = nn.StackedLSTM(mLSTMCell, 1, 10, 20, 15, 0)
		>>> input = Variable(torch.randn(6, 3, 10))
		>>> hx, cx = hiddens = rnn.state0(3)
		>>> hx.size() # (1,3,20)
		>>> cx.size() # (1,3,20)
		>>> output = []
		>>> for i in range(6):
		...     hiddens, out = rnn(input[i], hiddens)
		...     output.append(out)
	"""
	def __init__(self, cell, num_layers, input_size, rnn_size,
				output_size=-1, dropout=0.0, n_experts=10, hidden_dim_reduce=4096,
				dropouth=0.0, dropouti=0.0, dropoute=0.0, ldropout=0.0):
		super(StackedLSTM, self).__init__()

		# Different dropouts?
		self.dropouti = dropouti
		self.dropouth = dropouth
		self.dropoute = dropoute
		self.ldropout = ldropout
		self.dropoutl = ldropout

		#self.add_module('dropout', nn.Dropout(dropout))
		self.dropout = dropout

		self.lockdrop = LockedDropout()
		self.num_layers = num_layers
		self.rnn_size = rnn_size
		if output_size > 0:
			self.add_module('h2o', nn.Linear(rnn_size, output_size))
		self.output_size = output_size
		self.add_module('layers', nn.ModuleList(
			[cell(input_size if x == 0 else rnn_size, rnn_size) for x in range(num_layers)]))

		# MoS upon request
		self.n_experts = n_experts
		if n_experts > 1:
			print('initializing %d MoS experts' % n_experts)
			nhidlast = rnn_size
			ninp = output_size
			# (Optionally) reduce the hidden state dimension before MoS -- helps with memory (but may reduce quality)
			if hidden_dim_reduce > 0 and hidden_dim_reduce < nhidlast:
				print('initializing hidden_dim_reduce %d to %d' % (nhidlast, hidden_dim_reduce))
				self.add_module('dim_reducer', nn.Linear(nhidlast, hidden_dim_reduce))
				nhidlast = hidden_dim_reduce
			self.hidden_dim_reduce = nhidlast
			self.add_module('prior', nn.Linear(nhidlast, n_experts, bias=False))
			# NOTE: works better without the nn.Tanh() from word-level MoS paper (could not get Tanh to converge)
			self.add_module('latent', nn.Sequential(nn.Linear(nhidlast, n_experts*ninp))) # , nn.Tanh()))

	def forward(self, input, hidden):
		x = input
		h_0, c_0 = hidden
		h_1, c_1 = [], []
		# iterate over layers and propagate hidden state through to top rnn
		for i, layer in enumerate(self.layers):
			h_1_i, c_1_i = layer(x, (h_0[i], c_0[i]))
			if i == 0:
				x = h_1_i
			else:
				x = x + h_1_i
			if i != len(self.layers):
				#x = self.dropout(x)
				x = self.lockdrop(x, self.dropouth)
			h_1 += [h_1_i]
			c_1 += [c_1_i]

		h_1 = torch.stack(h_1)
		c_1 = torch.stack(c_1)
		output = h_1
		if self.output_size > 0:
			# MoS -- softmax over N softmaxes
			# https://github.com/zihangdai/mos/blob/master/model.py
			# TODO: Can apply decoder layer as well? [see Salesforce model above it based on]
			if self.n_experts > 1:
				x = self.lockdrop(x, self.dropout)

				# Reduce dimensions, if requested [memory blowup]
				if self.hidden_dim_reduce < self.rnn_size:
					x = self.dim_reducer(x)

				# Compute the value and prior of the MoS
				latent = self.latent(x)

				# Dropout for latent...
				latent = self.lockdrop(latent, self.dropoutl)

				logit = latent.view(-1, self.output_size)
				prior_logit = self.prior(x).contiguous().view(-1, self.n_experts)
				prior = nn.functional.softmax(prior_logit)

				prob = nn.functional.softmax(logit.view(-1, self.output_size)).view(-1, self.n_experts, self.output_size)
				prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

				log_prob = torch.log(prob.add_(1e-8))
				output = log_prob
			else:
				# Otherwise, just hidden to output -- linear function
				output = self.h2o(x)

		return (h_1, c_1), output

	def state0(self, batch_size, volatile=False):
			"""
			Get initial hidden state tuple for mLSTMCell
			Args:
				batch_size: The minibatch size we intend to process
			Inputs: batch_size, volatile
				- **batch_size** : integer or scalar tensor representing the minibatch size
				- **volatile** : boolean whether to make hidden state volatile. (requires_grad=False)
			Outputs: h_0, c_0
				- **h_0** (num_layers, batch, hidden_size): tensor containing the next hidden state
				  for each element and layer in the batch
				- **c_0** (num_layers, batch, hidden_size): tensor containing the next cell state
				  for each element and layer in the batch
			"""
			h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size),
					requires_grad=False, volatile=volatile)
			c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.rnn_size),
					requires_grad=False, volatile=volatile)
			return (h_0, c_0)
