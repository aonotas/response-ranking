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


class SentenceEncoderCNN(chainer.Chain):

    def __init__(self, emb_dim=100, window_size=3, hidden_dim=100,
                 n_vocab=None, use_dropout=0.50, add_n_vocab=0):
        dim = emb_dim
        self.hidden_dim = hidden_dim
        super(SentenceEncoderCNN, self).__init__(
            pad_emb=L.EmbedID(1, emb_dim, ignore_label=-1),
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            conv=L.Convolution2D(in_channels=1, out_channels=hidden_dim,
                                 ksize=(window_size, dim),
                                 stride=(1, dim), pad=0)
        )
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.dim = dim

        if add_n_vocab:
            add_word_embed = L.EmbedID(add_n_vocab, emb_dim, ignore_label=-1)
            self.add_link('add_word_embed', add_word_embed)

        self.add_n_vocab = add_n_vocab
        self.use_dropout = use_dropout
        self.train = True

    def set_train(self, train):
        self.train = train

    def __call__(self, x_data, lengths):

        xp = self.xp
        batchsize = len(x_data)

        # lengths = xp.concatenate(lengths, axis=0)
        lengths = np.concatenate(lengths, axis=0)
        # max_len = max(lengths)

        x_data = xp.concatenate(x_data, axis=0)
        # split_size = xp.cumsum(lengths)[:-1]
        # split_size = to_cpu(split_size)
        split_size = np.cumsum(lengths)[:-1]
        x_data = F.split_axis(x_data, split_size, axis=0)

        x_data = F.pad_sequence(x_data, padding=-1).data
        pad = xp.full((x_data.shape[0], self.window_size - 1), -1., dtype=xp.int32)
        x_data = xp.concatenate([pad, x_data, pad], axis=1)
        enable = x_data != -1

        # add offset (padding)
        x_data = x_data + 1

        if self.add_n_vocab:
            word_embW = F.concat([self.pad_emb.W, self.word_embed.W, self.add_word_embed.W], axis=0)
        else:
            word_embW = F.concat([self.pad_emb.W, self.word_embed.W], axis=0)
        # word_embW = F.concat([self.pad_emb.W, self.word_embed.W, self.add_word_embed.W], axis=0)
        word_embs = F.embed_id(x_data, word_embW, ignore_label=-1)
        word_embs = F.reshape(word_embs, (x_data.shape[0], 1, -1, self.dim))

        if self.use_dropout:
            word_embs = F.dropout(word_embs, ratio=self.use_dropout)

        word_embs = self.conv(word_embs)
        shape = word_embs.data.shape
        word_embs = F.reshape(word_embs, (shape[0], self.hidden_dim, -1))
        # Where Filter
        minus_inf_batch = self.xp.zeros(word_embs.data.shape,
                                        dtype=self.xp.float32) - 1024
        enable = enable[:, self.window_size - 1:]
        enable = xp.reshape(enable, (enable.shape[0], 1, -1))
        enable = xp.broadcast_to(enable, word_embs.shape)
        word_embs = F.where(enable, word_embs, minus_inf_batch)
        # max
        word_embs = F.max(word_embs, axis=2)
        return word_embs


class SentenceEncoderAverage(chainer.Chain):

    def __init__(self, n_vocab, emb_dim, hidden_dim, use_dropout, enc_type='avg', add_n_vocab=0):
        super(SentenceEncoderAverage, self).__init__(
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            output=L.Linear(emb_dim, hidden_dim),
        )
        self.use_dropout = use_dropout
        self.enc_type = enc_type

        if add_n_vocab:
            add_word_embed = L.EmbedID(add_n_vocab, emb_dim, ignore_label=-1)
            self.add_link('add_word_embed', add_word_embed)
        self.add_n_vocab = add_n_vocab

    def __call__(self, x_data, lengths):
        batchsize = len(x_data)
        xp = self.xp
        hx = None

        # 1-D flatten
        xs = xp.concatenate(x_data, axis=0)
        # lengths = xp.concatenate(lengths, axis=0)
        # split_size = xp.cumsum(lengths)[:-1]
        # split_size = to_cpu(split_size)
        lengths = np.concatenate(lengths, axis=0)
        split_size = np.cumsum(lengths)[:-1]

        xs = Variable(xs)

        if self.add_n_vocab:
            word_embW = F.concat([self.word_embed.W, self.add_word_embed.W], axis=0)
        else:
            word_embW = self.word_embed.W

        xs = F.embed_id(xs, word_embW, ignore_label=-1)
        # xs = self.word_embed(xs)
        if self.use_dropout > 0.0:
            xs = F.dropout(xs, ratio=self.use_dropout)

        # split
        xs = F.split_axis(xs, split_size, axis=0)
        xs = F.pad_sequence(xs, padding=0.0)
        sum_xs = F.sum(xs, axis=1)
        lengths = to_gpu(lengths)
        lengths_avg = xp.broadcast_to(lengths[..., None], sum_xs.shape)

        avg_xs = sum_xs / lengths_avg

        if self.enc_type == 'avg':
            last_vecs = avg_xs
        elif self.enc_type == 'sum':
            last_vecs = sum_xs

        last_vecs = self.output(last_vecs)

        if self.use_dropout > 0.0:
            last_vecs = F.dropout(last_vecs, ratio=self.use_dropout)
        return last_vecs


class SentenceEncoderGRU(chainer.Chain):

    def __init__(self, n_vocab, emb_dim, hidden_dim, use_dropout, add_n_vocab=0):
        super(SentenceEncoderGRU, self).__init__(
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            gru=L.NStepGRU(n_layers=1, in_size=emb_dim,
                           out_size=hidden_dim, dropout=use_dropout)
        )
        self.use_dropout = use_dropout
        if add_n_vocab:
            add_word_embed = L.EmbedID(add_n_vocab, emb_dim, ignore_label=-1)
            self.add_link('add_word_embed', add_word_embed)
        self.add_n_vocab = add_n_vocab

    def __call__(self, x_data, lengths):
        batchsize = len(x_data)
        xp = self.xp
        hx = None

        # 1-D flatten
        xs = xp.concatenate(x_data, axis=0)
        # lengths = xp.concatenate(lengths, axis=0)
        # split_size = xp.cumsum(lengths)[:-1]
        # split_size = to_cpu(split_size)
        lengths = np.concatenate(lengths, axis=0)
        split_size = np.cumsum(lengths)[:-1]

        xs = Variable(xs)
        if self.add_n_vocab:
            word_embW = F.concat([self.word_embed.W, self.add_word_embed.W], axis=0)
        else:
            word_embW = self.word_embed.W
        xs = F.embed_id(xs, word_embW, ignore_label=-1)
        # xs = self.word_embed(xs)
        if self.use_dropout > 0.0:
            xs = F.dropout(xs, ratio=self.use_dropout)

        # split
        xs = F.split_axis(xs, split_size, axis=0)

        # GRU
        hy, ys = self.gru(hx=hx, xs=xs)
        # Extract Last Vector
        # last_idx = xp.cumsum(lengths).astype(xp.int32) - 1
        # last_vecs = F.embed_id(last_idx, F.concat(ys, axis=0))
        # last_vecs = F.reshape(hy, (hy.shape[1], hy.shape[2]))
        last_vecs = hy[-1]
        if self.use_dropout > 0.0:
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

    def __call__(self, x_data, n_agents, n_agents_list=None):
        if n_agents_list is None:
            n_agents_list = n_agents.tolist()

        batchsize = len(x_data)
        hx = None
        xs = x_data
        x_size = len(xs)
        n_prev_sents = xs[0].shape[0]
        xp = self.xp

        hy, ys = self.gru(hx=hx, xs=xs)

        # Extract Last Vector
        # lengths = xp.full((x_size, ), n_prev_sents, dtype=xp.int32)
        # last_idx = xp.cumsum(lengths).astype(xp.int32)
        # last_idx = last_idx - 1
        # agent_vecs = F.embed_id(last_idx, F.concat(ys, axis=0))
        # agent_vecs = F.reshape(hy, (hy.shape[1], hy.shape[2]))
        agent_vecs = hy[-1]
        if self.use_dropout > 0.0:
            agent_vecs = F.dropout(agent_vecs, ratio=self.use_dropout)

        # Extract First Agent (idx=0)
        cumsum_idx = xp.cumsum(n_agents).astype(xp.int32)
        cumsum_idx_cpu = np.cumsum(n_agents_list).astype(np.int32)
        first_agent_idx = xp.concatenate([xp.zeros((1, ), dtype=xp.int32), cumsum_idx[:-1]], axis=0)
        spk_agent_vecs = F.embed_id(first_agent_idx, agent_vecs, ignore_label=-1)

        split_agent_vecs = F.split_axis(agent_vecs, cumsum_idx_cpu[:-1], axis=0)
        pad_agent_vecs = F.pad_sequence(split_agent_vecs, padding=-1024.)
        # Max Pooling
        h_context = F.max(pad_agent_vecs, axis=1)

        return agent_vecs, h_context, spk_agent_vecs


class ReverseGrad(function.Function):

    def __init__(self, reverse_flag=True):
        self.reverse_flag = reverse_flag
        flag = - 1.0
        if not reverse_flag:
            flag = 1.0
        self.flag = flag

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grad_outputs):
        # x, y = inputs
        gz, = grad_outputs

        gx = gz * inputs[0]

        return self.flag * gx,


class Critic(chainer.Chain):

    def __init__(self, input_dim, hidden_dim=512, output_dim=1, use_wgan=False):
        super(Critic, self).__init__(
            domain_layer=L.Linear(input_dim, hidden_dim),
            domain_final=L.Linear(hidden_dim, output_dim),
        )
        self.use_wgan = use_wgan

    def __call__(self, x):
        h = self.domain_layer(x)
        h = F.relu(h)
        h = self.domain_final(h)
        if self.use_wgan:
            h = F.sum(h) / h.size  # Mean
        return h


class MultiLingualConv(chainer.Chain):

    def __init__(self, args, n_vocab, init_emb=None, add_n_vocab=0, use_domain_adapt=0, n_domain=1):
        hidden_dim = args.dim_hidden
        if args.sentence_encoder_type == 'gru':
            sentence_encoder_context = SentenceEncoderGRU(
                n_vocab, args.dim_emb, hidden_dim, args.use_dropout, add_n_vocab=add_n_vocab)
        elif args.sentence_encoder_type in ['avg', 'sum']:
            sentence_encoder_context = SentenceEncoderAverage(
                n_vocab, args.dim_emb, hidden_dim, args.use_dropout, args.sentence_encoder_type, add_n_vocab=add_n_vocab)

        elif args.sentence_encoder_type == 'cnn':
            sentence_encoder_context = SentenceEncoderCNN(args.dim_emb, window_size=args.cnn_windows,
                                                          hidden_dim=hidden_dim, n_vocab=n_vocab, use_dropout=args.use_dropout, add_n_vocab=add_n_vocab)

            # sentence_encoder_response = SentenceEncoderGRU(
            #     n_vocab, args.dim_emb, hidden_dim, args.use_dropout)
        conversation_encoder = ConversationEncoderGRU(
            hidden_dim, hidden_dim, args.use_dropout, use_pad_unk=args.use_pad_unk)

        if init_emb is not None:
            print 'before:', sentence_encoder_context.word_embed.W.data
            sentence_encoder_context.word_embed.W.data[0, :] = 0.0
            sentence_encoder_context.word_embed.W.data[1:] = init_emb[:]
            print 'after:', sentence_encoder_context.word_embed.W.data
        super(MultiLingualConv, self).__init__(
            dammy_emb=L.EmbedID(1, hidden_dim, ignore_label=-1),
            sentence_encoder=sentence_encoder_context,
            conversation_encoder=conversation_encoder,
            layer_agent=L.Linear(hidden_dim * 2, hidden_dim, nobias=True),
            layer_response=L.Linear(hidden_dim * 2, hidden_dim, nobias=True),
        )

        if use_domain_adapt:
            critic = Critic(input_dim=hidden_dim * 2,
                            hidden_dim=hidden_dim, output_dim=n_domain)
            self.add_link('critic', critic)
        self.use_domain_adapt = use_domain_adapt
        self.n_domain = n_domain
        self.args = args
        self.use_pad_unk = args.use_pad_unk
        self.n_prev_sents = args.n_prev_sents
        self.candidate_size = args.n_cands

    def predict_all(self, samples):
        batchsize = self.args.batch
        (dev_contexts, dev_contexts_length, dev_responses, dev_responses_length,
         dev_agents_ids, dev_n_agents, dev_binned_n_agents, dev_y_adr, dev_y_res, max_idx_dev) = samples
        evaluator = Evaluator()

        def f(iteration_list, batchsize, start, end):
            for i_index, index in enumerate(iteration_list):
                contexts = dev_contexts[start:end][index:index + batchsize]

                responses = dev_responses[start:end][index:index + batchsize]
                agents_ids = dev_agents_ids[start:end][index:index + batchsize]
                contexts_length = dev_contexts_length[start:end][index:index + batchsize]
                contexts = [to_gpu(_i) for _i in contexts]
                responses = [to_gpu(_i) for _i in responses]
                agents_ids = [to_gpu(_i) for _i in agents_ids]
                contexts_length = [_i for _i in contexts_length]

                responses_length = dev_responses_length[start:end][index:index + batchsize]
                n_agents = to_gpu(dev_n_agents[start:end][index:index + batchsize])
                binned_n_agents_cpu = dev_binned_n_agents[start:end][index:index + batchsize]
                binned_n_agents = to_gpu(binned_n_agents_cpu)
                y_adr_cpu = dev_y_adr[start:end][index:index + batchsize]
                y_adr = to_gpu(y_adr_cpu)
                y_res_cpu = dev_y_res[start:end][index:index + batchsize]
                y_res = to_gpu(y_res_cpu)

                sample = [contexts, contexts_length, responses, responses_length,
                          agents_ids, n_agents, binned_n_agents, y_adr, y_res]
                self.n_prev_sents = len(contexts_length[0])
                dot_r, dot_a, predict_r, predict_a, _, _ = self.__call__(sample)

                # y_res_cpu = to_cpu(y_res)
                # y_adr_cpu = to_cpu(y_adr)
                evaluator.update(binned_n_agents_cpu, 0., 0., to_cpu(
                    predict_a.data), to_cpu(predict_r.data), y_adr_cpu, y_res_cpu)

        # print 'len(dev_contexts):', len(dev_contexts)
        # print 'max_idx_dev:', max_idx_dev
        iteration_list = range(0, max_idx_dev, batchsize)
        f(iteration_list, batchsize, start=0, end=max_idx_dev)

        iteration_list = range(max_idx_dev, len(dev_contexts), 1)
        f(iteration_list, 1, start=0, end=len(dev_contexts) + 1)

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

    def __call__(self, samples, y_domain=None):
        # Sentence Encoder
        xp = self.xp
        contexts, contexts_length, responses, responses_length, agents_ids, n_agents, binned_n_agents, y_adr, y_res = samples
        n_agents_list = to_cpu(n_agents).tolist()
        context_vecs = self.sentence_encoder(contexts, contexts_length)
        print ' context_vecs:', context_vecs.shape
        pad_context_vecs = context_vecs
        batchsize = n_agents.shape[0]
        if self.use_pad_unk:
            pad_context_vecs = F.concat([self.dammy_emb.W, context_vecs], axis=0)

        # TODO: use different GRU for responses?
        response_vecs = self.sentence_encoder(responses, responses_length)
        print ' response_vecs:', response_vecs.shape

        agents_ids = self.padding_offset(agents_ids, n_agents_list)
        split_size_cpu = np.arange(self.n_prev_sents, agents_ids.shape[0] * self.n_prev_sents,
                                   self.n_prev_sents).astype(np.int32)
        agent_input_vecs = F.embed_id(agents_ids, pad_context_vecs, ignore_label=-1)
        agent_input_vecs = F.reshape(agent_input_vecs, (-1, agent_input_vecs.shape[-1]))
        agent_input_vecs = F.split_axis(agent_input_vecs, split_size_cpu, axis=0)

        agent_vecs, h_context, spk_agent_vecs = self.conversation_encoder(
            agent_input_vecs, n_agents, n_agents_list)

        # predict
        a_h = F.concat([spk_agent_vecs, h_context], axis=1)

        response_o = self.layer_response(a_h)
        agent_o = self.layer_agent(a_h)

        self.domain_loss = 0.0
        if self.use_domain_adapt and y_domain is not None:
            h_domain = ReverseGrad(True)(a_h)
            h_domain = self.critic(h_domain)
            self.domain_loss = F.softmax_cross_entropy(h_domain, y_domain)

        r_shape = (batchsize, self.candidate_size, -1)
        response_vecs = F.reshape(response_vecs, r_shape)  # (batch, candidate_size, 256)
        response_o = F.reshape(response_o, (batchsize, 1, -1))  # (batch, 1, 256)

        dot_r = F.batch_matmul(response_vecs, response_o, transb=True)
        dot_r = F.reshape(dot_r, (batchsize, -1))
        dot_r_softmax = F.softmax(dot_r, axis=1)
        predict_r = F.argmax(dot_r_softmax, axis=1)
        # offset = xp.arange(0, batchsize * dot_r.shape[1], dot_r.shape[1]).astype(xp.int32)
        # y_res = y_res + offset

        # cumsum_idx = xp.cumsum(n_agents).astype(xp.int32)
        # cumsum_idx = to_cpu(cumsum_idx[:-1])
        cumsum_idx = np.cumsum(n_agents_list).astype(xp.int32)[:-1]
        agent_vec_list = F.split_axis(agent_vecs, cumsum_idx, axis=0)
        agent_vec_pad = F.pad_sequence(agent_vec_list, padding=-1024.)
        agent_vec_pad = agent_vec_pad[:, 1:, :]  # except speaker_agent
        agent_o = F.reshape(agent_o, (batchsize, 1, -1))
        dot_a = F.batch_matmul(agent_vec_pad, agent_o, transb=True)
        dot_a = F.reshape(dot_a, (batchsize, -1))
        flag = agent_vec_pad.data != -1024.
        flag = flag[:, :, 0]
        dot_a = F.where(flag, dot_a, xp.full(dot_a.shape, -1024., dtype=xp.float32))
        dot_a_softmax = F.softmax(dot_a, axis=1)
        predict_a = F.argmax(dot_a_softmax, axis=1)

        # offset labels
        # offset = xp.arange(0, batchsize * dot_a.shape[1], dot_a.shape[1]).astype(xp.int32)
        # y_adr = y_adr + offset
        # y_adr = y_adr + 1
        return dot_r, dot_a, predict_r, predict_a, y_res, y_adr
