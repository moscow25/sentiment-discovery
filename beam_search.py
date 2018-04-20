"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch
from collections import Iterable


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab_pad=None, vocab_start=None, vocab_end=None, cuda=False, init_input=None):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = vocab_pad
        self.bos = vocab_start
        self.eos = vocab_end
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        # TODO: figure out what to do for padding substitute
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        # TODO: figure out what to do with first start
        self.nextYs[0][0] = self.bos if init_input is None else init_input.squeeze()

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_llk):
        """Advance the beam."""
        num_tokens = word_llk.size(-1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_llk = word_llk + self.scores.unsqueeze(1).expand_as(word_llk)
        else:
            beam_llk = word_llk[0]

        flat_beam_llk = beam_llk.view(-1)

        bestScores, bestScoresId = flat_beam_llk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # beam and token idx each score came from
        prev_k = bestScoresId / num_tokens
        self.prevKs.append(bestScoresId / num_tokens)
        self.nextYs.append(bestScoresId % num_tokens)

        # End condition is when top-of-beam is EOS.
        if self.eos is not None and self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]

class BeamDecoder(object):
    def __init__(self, beam_size, vocab_pad=None, vocab_start=None, vocab_end=None, cuda=False, n_best=1, hidden_batch_dim=0):
        self.beams = None
        self.beam_size = beam_size
        self.vocab_pad = vocab_pad
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.cuda = cuda
        self.layer_first = layer_first
        self.n_best = n_best
        # TODO: handle advanced/modular indexing later
        #self.hidden_batch_dim = hidden_batch_dim
        self.hidden_batch_dim = 0


    def bottle(self, m, batch_size):
        return m.view(batch_size * self.beam_size, -1)

    def unbottle(self, m, batch_size):
        return m.view(self.beam_size, batch_size, -1)

    def var(self, a): return Variable(a)

    def rvar(self, a):
        repeat_dims = [1] * a.dim()
        repeat_dims = [self.beam_size] + repeat_dims
        return self.var(a.data.repeat(*repeat_dims))

    def rstates(self, s):
        if not isinstance(s, Iterable):
            return self.rvar(s)
        return tuple([self.rstates(_s) for _s in s])

    def state_update(self, s, batch_idx, beam_positions):
        size = list(_s.size())
        reshape_size = [self.beam_size, int(size[0]//self.beam_size)]+size[1:]
        sentStates = _s.view(*reshape_size)[:, batch_idx]
        sentStates.data.copy_(sentStates.data.index_select(0, beam_positions))

    def beam_state_update(self, s, batch_idx, beam_positions):
        # state=[]
        if not isinstance(s, Iterable):
            return self.state_update(s, batch_idx, beam_positions)
        return [self.beam_state_update(_s, batch_idx, beam_positions) for _s in s]
        # for _s in s:
        #     size = list(_s.size())
        #     reshape_size = [self.beam_size, int(size[0]//self.beam_size)]+size[1:]
        #     #in place update of states, since we fill .data tensor in place
        #     sentStates = _s.view(*reshape_size)[:, idx]
        #     sentStates.data.copy_(sentStates.data.index_select(0,positions))
        #     # state.append(sentStates)
        # # return tuple(state)

    def reset_beam_decoder(self, batch_size, states, init_input=None):
        self.beams=[Beam(self.beam_size, n_best=self.n_best,
                         cuda=self.cuda,
                         vocab_pad=self.vocab_pad,
                         vocab_start=self.vocab_start,
                         vocab_end=self.vocab_end,
                         init_input=None if init_input is None else init_input[j])
                for j in range(batch_size)]
        return self.get_input(), self.rstates(states)

    def step(self,dec_out,state):
        unbottled=self.unbottle(dec_out,len(self.beams))
        for j, b in enumerate(self.beams):
            b.advance(unbottled[:, j])
            self.beam_state_update(state, j, b.get_current_origin())
        return self.get_input()

    def get_input(self):
        # tensor_list=[]
        # for j,b in enumerate(self.beams):
        #     tensor_list.append(b.get_current_state().view(-1))
        return self.var(torch.cat([b.get_current_state().view(-1) for b in self.beams]).contiguous().view(-1))

    def get_hyp(self):
        """
        returns k lists (where k = min(self.n_best, k))
        """
        rtn=[[]]
        n_best=self.n_best
        for b in self.beams:
            scores, ks = b.sort_best()
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times,k)
                rtn[i].append(hyp)
        return rtn
