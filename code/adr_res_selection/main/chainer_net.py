import chainer
from chainer import Chain, cuda
from chainer import function, functions, links, optimizer
from chainer import Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

from ..utils.evaluator import Evaluator

import numpy as np

# sentence Encoder
# GRU or CNN Encoder

# SentenceEncoderSum


def extract_last_vector():
    pass

to_cpu = chainer.cuda.to_cpu
to_gpu = chainer.cuda.to_gpu


class SentenceEncoderGRU(chainer.Chain):

    def __init__(self, n_vocab, emb_dim, hidden_dim, use_dropout):
        super(SentenceEncoderGRU, self).__init__(
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            gru=L.NStepGRU(n_layers=1, in_size=emb_dim,
                           out_size=hidden_dim, dropout=use_dropout)
        )
        self.use_dropout = use_dropout

    def __call__(self, x_data, lengths):
        batchsize = len(x_data)
        xp = self.xp
        hx = None

        # 1-D flatten
        xs = xp.concatenate(x_data, axis=0)
        lengths = xp.concatenate(lengths, axis=0)
        split_size = xp.cumsum(lengths)[:-1]

        xs = Variable(xs)
        xs = self.word_embed(xs)
        xs = F.dropout(xs, ratio=self.use_dropout)

        # split
        xs = F.split_axis(xs, to_cpu(split_size), axis=0)

        # GRU
        hy, ys = self.gru(hx=hx, xs=xs)
        # Extract Last Vector
        # last_idx = xp.cumsum(lengths).astype(xp.int32) - 1
        # last_vecs = F.embed_id(last_idx, F.concat(ys, axis=0))
        last_vecs = F.reshape(hy, (hy.shape[1], hy.shape[2]))

        last_vecs = F.dropout(last_vecs, ratio=self.use_dropout)
        return last_vecs


# Dynamic Model
class ConversationEncoderGRU(chainer.Chain):

    def __init__(self, emb_dim, hidden_dim, use_dropout, use_pad_unk):
        super(ConversationEncoderGRU, self).__init__(
            gru=L.NStepGRU(n_layers=1, in_size=emb_dim,
                           out_size=hidden_dim, dropout=use_dropout)
        )
        self.use_dropout = use_dropout
        self.use_pad_unk = use_pad_unk

    def __call__(self, x_data, n_agents):

        batchsize = len(x_data)
        hx = None
        xs = x_data
        x_size = len(xs)
        n_prev_sents = xs[0].shape[0]
        xp = self.xp

        _hy_f, ys = self.gru(hx=hx, xs=xs)

        # Extract Last Vector
        lengths = xp.full((x_size, ), n_prev_sents, dtype=xp.int32)
        last_idx = xp.cumsum(lengths).astype(xp.int32)
        last_idx = last_idx - 1
        agent_vecs = F.embed_id(last_idx, F.concat(ys, axis=0))
        agent_vecs = F.dropout(agent_vecs, ratio=self.use_dropout)

        # Extract First Agent (idx=0)
        cumsum_idx = xp.cumsum(n_agents).astype(xp.int32)
        first_agent_idx = xp.concatenate([xp.zeros((1, ), dtype=xp.int32), cumsum_idx[:-1]], axis=0)
        spk_agent_vecs = F.embed_id(first_agent_idx, agent_vecs)

        split_agent_vecs = F.split_axis(agent_vecs, to_cpu(cumsum_idx[:-1]), axis=0)
        pad_agent_vecs = F.pad_sequence(split_agent_vecs, padding=-1024.)
        # Max Pooling
        h_context = F.max(pad_agent_vecs, axis=1)

        return agent_vecs, h_context, spk_agent_vecs


class MultiLingualConv(chainer.Chain):

    def __init__(self, args, n_vocab):
        hidden_dim = args.dim_hidden
        if args.sentence_encoder_type == 'gru':
            sentence_encoder_context = SentenceEncoderGRU(
                n_vocab, args.dim_emb, hidden_dim, args.use_dropout)
            sentence_encoder_response = SentenceEncoderGRU(
                n_vocab, args.dim_emb, hidden_dim, args.use_dropout)
        conversation_encoder = ConversationEncoderGRU(
            hidden_dim, hidden_dim, args.use_dropout, use_pad_unk=args.use_pad_unk)

        super(MultiLingualConv, self).__init__(
            dammy_emb=L.EmbedID(1, hidden_dim, ignore_label=-1),
            sentence_encoder=sentence_encoder_context,
            sentence_encoder_response=sentence_encoder_response,
            conversation_encoder=conversation_encoder,
            layer_agent=L.Linear(hidden_dim * 2, hidden_dim, nobias=True),
            layer_response=L.Linear(hidden_dim * 2, hidden_dim, nobias=True),
        )
        self.args = args
        self.use_pad_unk = args.use_pad_unk
        self.n_prev_sents = args.n_prev_sents
        self.candidate_size = args.n_cands

    def predict_all(self, samples):
        batchsize = 1
        (dev_contexts, dev_contexts_length, dev_responses, dev_responses_length,
         dev_agents_ids, dev_n_agents, dev_binned_n_agents, dev_y_adr, dev_y_res) = samples
        evaluator = Evaluator()
        iteration_list = range(0, len(dev_contexts), batchsize)
        predict_lists = []
        for i_index, index in enumerate(iteration_list):

            contexts = dev_contexts[index:index + batchsize]
            responses = dev_responses[index:index + batchsize]
            agents_ids = dev_agents_ids[index:index + batchsize]
            contexts_length = dev_contexts_length[index:index + batchsize]
            contexts = [to_gpu(_i) for _i in contexts]
            responses = [to_gpu(_i) for _i in responses]
            agents_ids = [to_gpu(_i) for _i in agents_ids]
            contexts_length = [to_gpu(_i) for _i in contexts_length]

            responses_length = to_gpu(dev_responses_length[index:index + batchsize])
            n_agents = to_gpu(dev_n_agents[index:index + batchsize])
            binned_n_agents_cpu = dev_binned_n_agents[index:index + batchsize]
            binned_n_agents = to_gpu(binned_n_agents_cpu)
            y_adr_cpu = dev_y_adr[index:index + batchsize]
            y_adr = to_gpu(y_adr_cpu)
            y_res_cpu = dev_y_res[index:index + batchsize]
            y_res = to_gpu(y_res_cpu)

            sample = [contexts, contexts_length, responses, responses_length,
                      agents_ids, n_agents, binned_n_agents, y_adr, y_res]
            self.n_prev_sents = len(contexts_length[0])
            dot_r, dot_a, predict_r, predict_a = self.__call__(sample)
            evaluator.update(binned_n_agents_cpu, 0., 0., to_cpu(
                predict_a.data), to_cpu(predict_r.data), y_adr_cpu, y_res_cpu)
        #
        evaluator.show_results()

        self.n_prev_sents = self.args.n_prev_sents

        return evaluator.acc_both, evaluator.acc_adr, evaluator.acc_res

    def padding_offset(self, agents_ids, n_agents_list):
        xp = self.xp
        agents_ids = xp.concatenate(agents_ids, axis=0)
        padding_idx = -1
        if self.use_pad_unk:
            padding_idx = 0
        flag = agents_ids == -1
        n_prev_sents = self.n_prev_sents
        batchsize = len(n_agents_list)
        offset = xp.arange(0, batchsize * n_prev_sents, n_prev_sents).astype(xp.int32)
        offset = xp.repeat(offset, repeats=n_agents_list, axis=0)[..., None]
        offset = xp.broadcast_to(offset, agents_ids.shape)
        if self.use_pad_unk:
            offset += 1

        agents_ids = agents_ids + offset
        # where
        agents_ids = xp.where(flag, xp.full(
            agents_ids.shape, padding_idx, dtype=xp.int32), agents_ids)

        return agents_ids

    def __call__(self, samples):
        # Sentence Encoder
        xp = self.xp
        contexts, contexts_length, responses, responses_length, agents_ids, n_agents, binned_n_agents, y_adr, y_res = samples
        n_agents_list = n_agents.tolist()
        context_vecs = self.sentence_encoder(contexts, contexts_length)
        pad_context_vecs = context_vecs
        batchsize = n_agents.shape[0]
        if self.use_pad_unk:
            pad_context_vecs = F.concat([self.dammy_emb.W, context_vecs], axis=0)

        # TODO: use different GRU for responses?
        response_vecs = self.sentence_encoder(responses, responses_length)

        agents_ids = self.padding_offset(agents_ids, n_agents_list)
        split_size_cpu = np.arange(self.n_prev_sents, agents_ids.shape[0] * self.n_prev_sents,
                                   self.n_prev_sents).astype(np.int32)
        agent_input_vecs = F.embed_id(agents_ids, pad_context_vecs)
        agent_input_vecs = F.reshape(agent_input_vecs, (-1, agent_input_vecs.shape[-1]))
        agent_input_vecs = F.split_axis(agent_input_vecs, split_size_cpu, axis=0)

        agent_vecs, h_context, spk_agent_vecs = self.conversation_encoder(
            agent_input_vecs, n_agents)

        # predict
        a_h = F.concat([spk_agent_vecs, h_context], axis=1)

        response_o = self.layer_response(a_h)
        agent_o = self.layer_agent(a_h)

        r_shape = (batchsize, self.candidate_size, -1)
        response_vecs = F.reshape(response_vecs, r_shape)  # (batch, candidate_size, 256)
        response_o = F.reshape(response_o, (batchsize, 1, -1))  # (batch, 1, 256)

        dot_r = F.batch_matmul(response_vecs, response_o, transb=True)
        dot_r = F.reshape(dot_r, (batchsize, -1))
        dot_r_softmax = F.softmax(dot_r)
        predict_r = F.argmax(dot_r_softmax, axis=1)

        cumsum_idx = xp.cumsum(n_agents).astype(xp.int32)
        agent_vec_list = F.split_axis(agent_vecs, to_cpu(cumsum_idx[:-1]), axis=0)
        agent_vec_pad = F.pad_sequence(agent_vec_list, padding=-1024.)
        agent_o = F.reshape(agent_o, (batchsize, 1, -1))
        dot_a = F.batch_matmul(agent_vec_pad, agent_o, transb=True)
        dot_a = F.reshape(dot_a, (batchsize, -1))
        flag = agent_vec_pad.data != -1024.
        flag = flag[:, :, 0]
        dot_a = F.where(flag, dot_a, xp.full(dot_a.shape, -1024., dtype=xp.float32))
        dot_a_softmax = F.softmax(dot_a, axis=1)
        predict_a = F.argmax(dot_a_softmax, axis=1)

        return dot_r, dot_a, predict_r, predict_a
