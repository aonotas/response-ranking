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

import itertools


def extract_last_vector():
    pass

to_cpu = chainer.cuda.to_cpu
to_gpu = chainer.cuda.to_gpu


def binary_cross_entropy(y_p, n_p):
    return - (F.sum(F.log1p(y_p)) + F.sum(F.log1p(1. - n_p)))

class SentenceEncoderCNN(chainer.Chain):

    def __init__(self, emb_dim=100, window_size=3, hidden_dim=100,
                 n_vocab=None, use_dropout=0.50, add_n_vocab=0, domain_dim=0,
                 is_sum=0):
        dim = emb_dim + domain_dim
        self.hidden_dim = hidden_dim
        super(SentenceEncoderCNN, self).__init__(
            pad_emb=L.EmbedID(1, emb_dim, ignore_label=-1),
            word_embed=L.EmbedID(n_vocab, emb_dim, ignore_label=-1),
            conv=L.Convolution2D(in_channels=1, out_channels=hidden_dim,
                                 ksize=(window_size, dim),
                                 stride=(1, dim), pad=0)
        )
        self.is_sum = is_sum
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

    def __call__(self, x_data, lengths, y_domain=None, domain_embed=None):

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
        if y_domain is not None and domain_embed is not None:
            y_domain = self.xp.reshape(y_domain, (y_domain.shape[0], 1))
            y_domain_input = xp.repeat(y_domain, x_data.shape[0] / y_domain.shape[0], axis=0)
            y_domain_input = xp.broadcast_to(y_domain_input, x_data.shape)
            input_domain_vecs = domain_embed(y_domain_input)

            if self.is_sum:
                word_embs = word_embs + input_domain_vecs
            else:
                word_embs = F.concat([word_embs, input_domain_vecs], axis=2)

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

    def __call__(self, x_data, lengths, y_domain=None, domain_embed=None):
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
        if use_wgan:
            output_dim = 1
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


class MLP(chainer.Chain):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__(
            hidden_layer=L.Linear(input_dim, hidden_dim),
            output_layer=L.Linear(hidden_dim, output_dim),
        )

    def __call__(self, x):
        h = self.hidden_layer(x)
        h = F.relu(h)
        h = self.output_layer(h)
        return h


class MultiLingualConv(chainer.Chain):

    def __init__(self, args, n_vocab, init_emb=None, add_n_vocab=0,
                 use_domain_adapt=0, n_domain=1, use_wgan=0, use_wgan_for_both=1, use_mlp_layers=0, wgan_sep=0):
        hidden_dim = args.dim_hidden
        use_domain_input_emb = args.use_domain_input_emb
        use_domain_input_emb_sumver = args.use_domain_input_emb_sumver
        # use_wgan = args.use_wgan
        self.num_critic = args.num_critic
        self.use_wgan = use_wgan
        self.use_wgan_loss_decay = 1
        self.use_wgan_for_both = use_wgan_for_both

        self.use_mlp_layers = use_mlp_layers
        domain_dim = 0
        concat_domain_dim = 0

        if use_domain_input_emb:
            domain_dim = 256
            concat_domain_dim = 256
        elif use_domain_input_emb_sumver:
            domain_dim = args.dim_emb
            concat_domain_dim = 0

        if args.sentence_encoder_type == 'gru':
            sentence_encoder_context = SentenceEncoderGRU(
                n_vocab, args.dim_emb, hidden_dim, args.use_dropout, add_n_vocab=add_n_vocab)
        elif args.sentence_encoder_type in ['avg', 'sum']:
            sentence_encoder_context = SentenceEncoderAverage(
                n_vocab, args.dim_emb, hidden_dim, args.use_dropout, args.sentence_encoder_type, add_n_vocab=add_n_vocab)

        elif args.sentence_encoder_type == 'cnn':
            sentence_encoder_context = SentenceEncoderCNN(args.dim_emb, window_size=args.cnn_windows,
                                                          hidden_dim=hidden_dim, n_vocab=n_vocab, use_dropout=args.use_dropout, add_n_vocab=add_n_vocab,
                                                          domain_dim=concat_domain_dim, is_sum=args.use_domain_input_emb_sumver)

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

        if self.use_mlp_layers:
            # self.add_link('mlp_response', MLP(hidden_dim, hidden_dim, hidden_dim))
            # self.add_link('mlp_agent', MLP(hidden_dim, hidden_dim, hidden_dim))
            self.add_link('mlp_spk_agent', MLP(hidden_dim, hidden_dim, hidden_dim))
            self.add_link('mlp_context', MLP(hidden_dim, hidden_dim, hidden_dim))

        self.domain_loss_names = args.domain_loss_names.split(',')

        self.critic_names = []
        self.opt_list = []
        self.wgan_sep = wgan_sep
        if use_domain_adapt:
            critic_input_dim = hidden_dim * 2
            if self.wgan_sep:
                critic_input_dim = hidden_dim

            critic = Critic(input_dim=critic_input_dim, hidden_dim=hidden_dim,
                            output_dim=n_domain, use_wgan=use_wgan)
            self.add_link('critic', critic)
            self.critic_names = ['critic']
            if use_wgan:
                init_alpha_critic = 0.00005
                critic_opt = optimizers.Adam(alpha=init_alpha_critic)
                self.critic_opt = critic_opt
                self.critic_opt.setup(critic)
                self.opt_list = [critic_opt]
                self.opt_list_sep = []
            if self.wgan_sep:
                critic = Critic(input_dim=critic_input_dim, hidden_dim=hidden_dim,
                                output_dim=n_domain, use_wgan=use_wgan)
                self.add_link('double_critic', critic)
                # self.critic_names.append('double_critic')
                critic_opt = optimizers.Adam(alpha=init_alpha_critic)
                critic_opt.setup(critic)
                self.opt_list_sep.append(critic_opt)

            if use_wgan:

                if args.use_wgan_comb == 'comb':
                    # [(0, 1), (0, 2), (1, 2)]
                    wgan_comb_names = list(itertools.combinations(range(n_domain), 2))
                    self.wgan_comb_names = wgan_comb_names
                elif args.use_wgan_comb == 'multi_one':
                    # multi-souce, one-target
                    # [(0, 2), (1, 2)]
                    target_idx = n_domain - 1
                    wgan_comb_names = [(_, target_idx) for _ in range(0, target_idx)]
                    self.wgan_comb_names = wgan_comb_names
                elif args.use_wgan_comb == 'one_multi':
                    # one-source, multi-target,
                    # [(0, 1), (0, 2)]
                    wgan_comb_names = [(0, _) for _ in range(1, n_domain)]
                    self.wgan_comb_names = wgan_comb_names
                elif args.use_wgan_comb == 'target_only':
                    # one-source, one-target,
                    # [(0, 2)]
                    wgan_comb_names = [(0, n_domain - 1)]
                    self.wgan_comb_names = wgan_comb_names
                elif args.use_wgan_comb == 'concat':
                    # multi-concat-source, one-target,
                    # [(0, 1, 2)]
                    wgan_comb_names = [range(n_domain)]
                    self.wgan_comb_names = wgan_comb_names

                print 'args.use_wgan_comb:', args.use_wgan_comb
                print 'wgan_comb_names:', wgan_comb_names
                wgan_combs_iter = wgan_comb_names[1:]

                for tup in wgan_combs_iter:
                    i = '_'.join(map(str, tup)) if isinstance(tup, tuple) else tup
                    critic = Critic(input_dim=critic_input_dim, hidden_dim=hidden_dim,
                                    output_dim=n_domain, use_wgan=use_wgan)
                    name = 'critic_' + str(i)
                    self.add_link(name, critic)
                    self.critic_names.append(name)

                    if self.wgan_sep:
                        critic_sep = Critic(input_dim=critic_input_dim, hidden_dim=hidden_dim,
                                            output_dim=n_domain, use_wgan=use_wgan)
                        self.add_link('double_' + name, critic_sep)
                        # self.critic_names.append('double_' + name)

                        critic_opt = optimizers.Adam(alpha=init_alpha_critic)
                        critic_opt.setup(critic_sep)
                        self.opt_list_sep.append(critic_opt)

                    init_alpha_critic = 0.00005
                    critic_opt = optimizers.Adam(alpha=init_alpha_critic)
                    critic_opt.setup(critic)
                    self.opt_list.append(critic_opt)

            critic_response = Critic(input_dim=hidden_dim,
                                     hidden_dim=hidden_dim, output_dim=n_domain,
                                     use_wgan=use_wgan)
            critic_context = Critic(input_dim=hidden_dim,
                                    hidden_dim=hidden_dim, output_dim=n_domain,
                                    use_wgan=use_wgan)
            critic_agent = Critic(input_dim=hidden_dim,
                                  hidden_dim=hidden_dim, output_dim=n_domain,
                                  use_wgan=use_wgan)
            self.add_link('critic_response', critic_response)
            self.add_link('critic_context', critic_context)
            self.add_link('critic_agent', critic_agent)

        if use_domain_input_emb:
            domain_embed = L.EmbedID(n_domain, domain_dim, ignore_label=-1)
            self.add_link('domain_embed', domain_embed)
        else:
            self.domain_embed = None

        self.use_domain_adapt = use_domain_adapt
        self.n_domain = n_domain
        self.args = args
        self.use_pad_unk = args.use_pad_unk
        self.n_prev_sents = args.n_prev_sents
        self.candidate_size = args.n_cands
        self.compute_loss = True

    def get_layer(self, name):
        return self.__getitem__(name)

    def clip_discriminator_weights(self, link):
        upper = 0.01
        lower = -0.01
        for name, param in link.namedparams():
            if param.data is None:
                continue
            if self.use_wgan_loss_decay:
                # WGAN decay clip
                xp = self.xp
                ratio_lower = xp.amin(param.data) / lower
                ratio_upper = xp.amax(param.data) / upper
                ratio = max(ratio_lower, ratio_upper)
                if ratio > 1:
                    param.data /= ratio

            else:
                # Wgan clip
                param.data = self.xp.clip(param.data, lower, upper)

    def predict_all(self, samples, domain_index=0):
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

                y_domain = self.xp.full((len(responses), ), domain_index, self.xp.int32)

                responses_length = dev_responses_length[start:end][index:index + batchsize]
                # n_agents = to_gpu(dev_n_agents[start:end][index:index + batchsize])
                n_agents_cpu = dev_n_agents[start:end][index:index + batchsize]
                n_agents = to_gpu(n_agents_cpu)
                binned_n_agents_cpu = dev_binned_n_agents[start:end][index:index + batchsize]
                binned_n_agents = to_gpu(binned_n_agents_cpu)
                y_adr_cpu = dev_y_adr[start:end][index:index + batchsize]
                y_adr = to_gpu(y_adr_cpu)
                y_res_cpu = dev_y_res[start:end][index:index + batchsize]
                y_res = to_gpu(y_res_cpu)

                sample = [contexts, contexts_length, responses, responses_length,
                          agents_ids, n_agents, binned_n_agents, y_adr, y_res]
                self.n_prev_sents = len(contexts_length[0])
                dot_r, dot_a, predict_r, predict_a, _, _ = self.__call__(sample, y_domain=y_domain)

                # y_res_cpu = to_cpu(y_res)
                # y_adr_cpu = to_cpu(y_adr)
                # evaluator.update(binned_n_agents_cpu, 0., 0., to_cpu(
                #     predict_a.data), to_cpu(predict_r.data), y_adr_cpu, y_res_cpu)
                evaluator.update(n_agents_cpu, 0., 0., to_cpu(predict_a.data), to_cpu(predict_r.data), y_adr_cpu, y_res_cpu)

        # print 'len(dev_contexts):', len(dev_contexts)
        # print 'max_idx_dev:', max_idx_dev
        iteration_list = range(0, max_idx_dev, batchsize)
        f(iteration_list, batchsize, start=0, end=max_idx_dev)

        iteration_list = range(max_idx_dev, len(dev_contexts), 1)
        f(iteration_list, 1, start=0, end=len(dev_contexts) + 1)

        evaluator.show_results()
        self.adr_histry = evaluator.adr_histry
        self.res_histry = evaluator.res_histry
        self.both_histry = evaluator.both_histry

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

    def __call__(self, samples, y_domain=None, y_domain_count=None):
        # Sentence Encoder
        xp = self.xp
        self.domain_loss = 0.0
        contexts, contexts_length, responses, responses_length, agents_ids, n_agents, binned_n_agents, y_adr, y_res = samples
        n_agents_list = to_cpu(n_agents).tolist()
        context_vecs = self.sentence_encoder(
            contexts, contexts_length, y_domain=y_domain, domain_embed=self.domain_embed)

        if 'context' in self.domain_loss_names and self.use_domain_adapt and y_domain is not None and self.compute_loss:
            h_domain = ReverseGrad(True)(context_vecs)
            h_domain = self.critic_context(h_domain)
            y_domain_context = xp.repeat(y_domain, 15, axis=0)
            self.domain_loss += F.softmax_cross_entropy(h_domain, y_domain_context)

        pad_context_vecs = context_vecs
        batchsize = n_agents.shape[0]
        if self.use_pad_unk:
            pad_context_vecs = F.concat([self.dammy_emb.W, context_vecs], axis=0)

        # TODO: use different GRU for responses?
        response_vecs = self.sentence_encoder(
            responses, responses_length, y_domain=y_domain, domain_embed=self.domain_embed)

        # if self.use_mlp_layers:
        #     response_vecs = self.mlp_response(response_vecs)

        if 'response' in self.domain_loss_names and self.use_domain_adapt and y_domain is not None and self.compute_loss:
            h_domain = ReverseGrad(True)(response_vecs)
            h_domain = self.critic_response(h_domain)
            y_domain_response = xp.repeat(y_domain, self.candidate_size, axis=0)
            self.domain_loss += F.softmax_cross_entropy(h_domain, y_domain_response)

        agents_ids = self.padding_offset(agents_ids, n_agents_list)
        split_size_cpu = np.arange(self.n_prev_sents, agents_ids.shape[0] * self.n_prev_sents,
                                   self.n_prev_sents).astype(np.int32)
        agent_input_vecs = F.embed_id(agents_ids, pad_context_vecs, ignore_label=-1)
        agent_input_vecs = F.reshape(agent_input_vecs, (-1, agent_input_vecs.shape[-1]))
        agent_input_vecs = F.split_axis(agent_input_vecs, split_size_cpu, axis=0)

        agent_vecs, h_context, spk_agent_vecs = self.conversation_encoder(
            agent_input_vecs, n_agents, n_agents_list)

        if self.use_mlp_layers:
            # agent_vecs = self.mlp_agent(agent_vecs)
            h_context = self.mlp_context(h_context)
            spk_agent_vecs = self.mlp_spk_agent(spk_agent_vecs)

        if 'agent' in self.domain_loss_names and self.use_domain_adapt and y_domain is not None and self.compute_loss:
            h_domain = ReverseGrad(True)(agent_vecs)
            h_domain = self.critic_agent(h_domain)
            y_domain_agent = xp.repeat(y_domain, n_agents_list, axis=0)
            self.domain_loss += F.softmax_cross_entropy(h_domain, y_domain_agent)

        # predict
        a_h = F.concat([spk_agent_vecs, h_context], axis=1)

        response_o = self.layer_response(a_h)
        agent_o = self.layer_agent(a_h)

        if 'output' in self.domain_loss_names and self.use_domain_adapt and y_domain is not None and self.compute_loss:
            if self.use_wgan:
                # source_domain_idx = 0
                # sample data
                split_size = np.cumsum(y_domain_count)[:-1]
                if self.wgan_sep:
                    h_domain_list = F.split_axis(spk_agent_vecs, split_size, axis=0)
                    h_domain_list_double = F.split_axis(h_context, split_size, axis=0)
                else:
                    h_domain_list = F.split_axis(a_h, split_size, axis=0)

                sum_loss_critic = 0.0
                for k in xrange(self.num_critic):
                    loss_critic = 0.0
                    for wi, (tup, critic_name, critic_opt) in enumerate(zip(self.wgan_comb_names, self.critic_names, self.opt_list)):
                        if len(tup) == 2:
                            source_idx, target_idx = tup
                            h_source = h_domain_list[source_idx]
                            if self.wgan_sep:
                                h_source_double = h_domain_list_double[source_idx]
                        else:
                            target_idx = tup[-1]
                            h_source = F.concat([h_domain_list[_idx] for _idx in tup[:-1]], axis=0)
                            if self.wgan_sep:
                                h_source_double = F.concat([h_domain_list_double[_idx]
                                                            for _idx in tup[:-1]], axis=0)

                        bf_flag = self.args.mini_source_label != -1
                        keep_domain_idx = self.i_index % self.args.mini_source_label
                        if bf_flag:
                            if source_idx != keep_domain_idx or y_domain_count[target_idx] == 0:
                                continue

                        h_target = h_domain_list[target_idx]
                        h_source_data = Variable(h_source.data)  # unchain
                        h_target_data = Variable(h_target.data)  # unchain
                        critic_link = self.get_layer(critic_name)
                        self.clip_discriminator_weights(critic_link)

                        fw_source = critic_link(h_source_data)
                        fw_target = critic_link(h_target_data)
                        loss_critic = - F.sum(fw_source - fw_target)
                        sum_div = self.num_critic
                        sum_loss_critic += float(loss_critic.data) / sum_div
                        critic_link.cleargrads()
                        loss_critic.backward()
                        critic_opt.update()

                        if self.wgan_sep:
                            h_target = h_domain_list_double[target_idx]
                            h_source_data = Variable(h_source_double.data)  # unchain
                            h_target_data = Variable(h_target.data)  # unchain
                            critic_link = self.get_layer('double_' + critic_name)
                            self.clip_discriminator_weights(critic_link)

                            fw_source = critic_link(h_source_data)
                            fw_target = critic_link(h_target_data)
                            loss_critic = - F.sum(fw_source - fw_target)
                            sum_div = self.num_critic
                            sum_loss_critic += float(loss_critic.data) / sum_div
                            critic_link.cleargrads()
                            loss_critic.backward()
                            critic_opt_sep = self.opt_list_sep[wi]
                            critic_opt_sep.update()

                # generator loss
                domain_loss = 0.0
                domain_loss_sep = 0.0
                for tup, critic_name in zip(self.wgan_comb_names, self.critic_names):
                    if len(tup) == 2:
                        source_idx, target_idx = tup
                        h_source = h_domain_list[source_idx]
                        if self.wgan_sep:
                            h_source_double = h_domain_list_double[source_idx]
                    else:
                        target_idx = tup[-1]
                        h_source = F.concat([h_domain_list[_idx] for _idx in tup[:-1]], axis=0)
                        if self.wgan_sep:
                            h_source_double = F.concat([h_domain_list_double[_idx]
                                                        for _idx in tup[:-1]], axis=0)
                    if bf_flag:
                        if source_idx != keep_domain_idx or y_domain_count[target_idx] == 0:
                            continue
                    h_target = h_domain_list[target_idx]
                    critic_link = self.get_layer(critic_name)
                    fw_source = critic_link(h_source)
                    fw_target = critic_link(h_target)
                    if self.use_wgan_for_both:
                        domain_loss += F.sum(fw_source - fw_target)
                    else:
                        domain_loss += - F.sum(fw_target)

                    if self.wgan_sep:
                        h_target = h_domain_list_double[target_idx]
                        critic_link = self.get_layer('double_' + critic_name)
                        fw_source = critic_link(h_source_double)
                        fw_target = critic_link(h_target)
                        if self.use_wgan_for_both:
                            domain_loss_sep += F.sum(fw_source - fw_target)
                        else:
                            domain_loss_sep += - F.sum(fw_target)

                self.domain_loss_one = domain_loss
                self.domain_loss_sep = domain_loss_sep
                self.domain_loss = domain_loss + domain_loss_sep

                self.sum_loss_critic = sum_loss_critic
            else:
                h_domain = ReverseGrad(True)(a_h)
                h_domain = self.critic(h_domain)
                self.domain_loss += F.softmax_cross_entropy(h_domain, y_domain)

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
