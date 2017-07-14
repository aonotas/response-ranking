import chainer
from chainer import Chain, cuda
from chainer import function, functions, links, optimizer
from chainer import Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

from ..utils.evaluator import Evaluator


# sentence Encoder
# GRU or CNN Encoder

# SentenceEncoderSum


def extract_last_vector():
    pass


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
        xs = F.split_axis(xs, split_size, axis=0)

        # GRU
        _hy, ys = self.gru(hx=hx, xs=xs)
        # Extract Last Vector
        last_idx = xp.cumsum(lengths).astype(xp.int32) - 1
        last_vecs = F.embed_id(last_idx, F.concat(ys, axis=0))
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
        candidate_size = xs[0].shape[0]
        xp = self.xp

        _hy_f, ys = self.gru(hx=hx, xs=xs)

        # Extract Last Vector
        lengths = xp.full((x_size, ), candidate_size, dtype=xp.int32)
        last_idx = xp.cumsum(lengths).astype(xp.int32)
        last_idx = last_idx - 1
        agent_vecs = F.embed_id(last_idx, F.concat(ys, axis=0))
        agent_vecs = F.dropout(agent_vecs, ratio=self.use_dropout)

        # Extract First Agent (idx=0)
        cumsum_idx = xp.cumsum(n_agents).astype(xp.int32)
        first_agent_idx = xp.concatenate([xp.zeros((1, ), dtype=xp.int32), cumsum_idx[:-1]], axis=0)
        spk_agent_vecs = F.embed_id(first_agent_idx, agent_vecs)

        split_agent_vecs = F.split_axis(agent_vecs, cumsum_idx[:-1], axis=0)
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
            layer_a=L.Linear(hidden_dim * 2, hidden_dim, nobias=True),
            layer_r=L.Linear(hidden_dim * 2, hidden_dim, nobias=True),
        )
        self.args = args
        self.use_pad_unk = args.use_pad_unk
        self.candidate_size = args.n_prev_sents

    def predict_all(self, samples, batchsize=32):
        evaluator = Evaluator()
        iteration_list = range(0, len(samples), batchsize)
        #
        # for i, sample in enumerate(samples):
        #     if i != 0 and i % 100 == 0:
        #         say("  {}/{}".format(i, len(samples)))
        #
        #     contexts, responses, agents_vecs, n_agents, binned_n_agents, y_adr, y_res = sample
        #     # pred_a, pred_r = self.predict(c=x[0], r=x[1], a=x[2], y_r=x[3], y_a=x[4], n_agents=x[5])
        #
        #     evaluator.update(binned_n_agents, 0., 0., pred_a, pred_r, y_adr, y_res)
        #
        # evaluator.show_results()
        #
        # return evaluator.acc_both, evaluator.acc_adr, evaluator.acc_res

    def padding_offset(self, agents_ids, n_agents):
        xp = self.xp
        agents_ids = xp.concatenate(agents_ids, axis=0)
        padding_idx = -1
        if self.use_pad_unk:
            padding_idx = 0
        flag = agents_ids == -1
        candidate_size = self.candidate_size
        batchsize = n_agents.shape[0]
        offset = xp.arange(0, batchsize * candidate_size, candidate_size).astype(xp.int32)
        offset = xp.repeat(offset, repeats=n_agents)[..., None]
        offset = xp.broadcast_to(offset, agents_ids.shape)

        agents_ids = agents_ids + offset
        # where
        agents_ids = xp.where(flag, xp.full(
            agents_ids.shape, padding_idx, dtype=xp.int32), agents_ids)

        # print 'offset:', offset
        return agents_ids

    def __call__(self, samples):
        # Sentence Encoder
        xp = self.xp
        contexts, contexts_length, responses, responses_length, agents_ids, n_agents, binned_n_agents, y_adr, y_res = samples
        context_vecs = self.sentence_encoder(contexts, contexts_length)
        pad_context_vecs = context_vecs
        batchsize = n_agents.shape[0]
        if self.use_pad_unk:
            pad_context_vecs = F.concat([self.dammy_emb.W, context_vecs], axis=0)

        # TODO: use different GRU for responses?
        response_vecs = self.sentence_encoder(responses, responses_length)

        agents_ids = self.padding_offset(agents_ids, n_agents)
        split_size = xp.arange(self.candidate_size, agents_ids.shape[0] * self.candidate_size,
                               self.candidate_size).astype(xp.int32)
        agent_input_vecs = F.embed_id(agents_ids, pad_context_vecs)
        agent_input_vecs = F.reshape(agent_input_vecs, (-1, agent_input_vecs.shape[-1]))
        agent_input_vecs = F.split_axis(agent_input_vecs, split_size, axis=0)

        agent_vecs, h_context, spk_agent_vecs = self.conversation_encoder(
            agent_input_vecs, n_agents)

        # predict
        a_h = F.concat([spk_agent_vecs, h_context], axis=1)

        o_a = self.layer_a(a_h)
        o_r = self.layer_r(a_h)

        # broadcast
        response_idx = xp.repeat(xp.arange(batchsize), self.candidate_size).astype(xp.int32)
        response_o = F.embed_id(response_idx, o_a)
        # TODO: batch_matmul(response_o, response_vecs)

        agent_idx = xp.repeat(xp.arange(batchsize), n_agents).astype(xp.int32)
        agent_o = F.embed_id(agent_idx, o_r)
        # TODO: batch_matmul(agent_o, spk_agent_vecs)
