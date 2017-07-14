import os
os.environ["CHAINER_SEED"] = "1234"

import random
import numpy as np
random.seed(1234)
np.random.seed(1234)
import pickle

import chainer
from chainer import cuda
from chainer import optimizers
import chainer.functions as F

import logging
logger = logging.getLogger(__name__)

chainer.config.use_cudnn = 'always'
to_cpu = chainer.cuda.to_cpu

from chainer import serializers

from ..preprocess import convert_word_into_id, get_samples
from ..utils import say, load_dataset, load_init_emb, load_multi_ling_init_emb

import chainer_util as ch_util


def get_datasets(argv):
    say('\nSET UP DATASET\n')
    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id,
    # addressee_id, response1, ... , label)
    say('\nLoad dataset...')
    train_dataset, words = load_dataset(fn=argv.train_data, data_size=argv.data_size, test=False)
    dev_dataset, words = load_dataset(fn=argv.dev_data, vocab=words,
                                      data_size=argv.data_size, test=True)
    test_dataset, words = load_dataset(
        fn=argv.test_data, vocab=words, data_size=argv.data_size, test=True)
    return train_dataset, dev_dataset, test_dataset, words


def create_samples(argv, train_dataset, dev_dataset, test_dataset, vocab_word):
    ###########################
    # Task setting parameters #
    ###########################
    n_prev_sents = argv.n_prev_sents
    max_n_words = argv.max_n_words
    sample_size = argv.sample_size
    batch_size = argv.batch

    cands = train_dataset[0][0][3:-1]
    n_cands = len(cands)

    say('\n\nTASK  SETTING')
    say('\n\tResponse Candidates:%d  Contexts:%d  Max Word Num:%d\n' %
        (n_cands, n_prev_sents, max_n_words))

    ##########################
    # Convert words into ids #
    ##########################
    say('\n\nConverting words into ids...')
    # samples: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id,
    # addressee_id, response, ..., label)
    train_samples = convert_word_into_id(train_dataset, vocab_word)
    dev_samples = convert_word_into_id(dev_dataset, vocab_word)
    test_samples = convert_word_into_id(test_dataset, vocab_word)

    ##################
    # Create samples #
    ##################
    say('\n\nCreating samples...')
    # samples: 1D: n_samples; 2D: Sample
    train_samples = get_samples(threads=train_samples,
                                n_prev_sents=n_prev_sents,
                                max_n_words=max_n_words, pad=False)
    dev_samples = get_samples(threads=dev_samples, n_prev_sents=n_prev_sents,
                              max_n_words=max_n_words, test=True, pad=False)
    test_samples = get_samples(threads=test_samples, n_prev_sents=n_prev_sents,
                               max_n_words=max_n_words, test=True, pad=False)

    ###################################
    # Limit the used training samples #
    ###################################
    if sample_size > 1:
        np.random.shuffle(train_samples)
        train_samples = train_samples[: (len(train_samples) / sample_size)]

    ###################################
    # Create theano-formatted samples #
    ###################################
    # train_samples, n_train_batches, evalset = theano_shared_format(train_samples, batch_size)
    # dev_samples = numpy_format(dev_samples, batch_size, test=True)
    # test_samples = numpy_format(test_samples, batch_size, test=True)

    say('\n\nTRAIN SETTING\tBatch Size:%d  Epoch:%d  Vocab:%d  Max Words:%d' %
        (batch_size, argv.epoch, vocab_word.size(), max_n_words))
    say('\n\nTrain samples\tMini-Batch:%d' % n_train_batches)
    if dev_samples:
        say('\nDev samples\tMini-Batch:%d' % len(dev_samples))
    if test_samples:
        say('\nTest samples\tMini-Batch:%d' % len(test_samples))
    return train_samples, dev_samples, test_samples, n_train_batches, evalset


def train(argv, model_api, n_train_batches, evalset, dev_samples, test_samples):
    say('\n\nTRAINING START\n')

    acc_history = {}
    best_dev_acc_both = 0.
    unchanged = 0

    batch_indices = range(n_train_batches)

    for epoch in xrange(argv.epoch):
        ##############
        # Early stop #
        ##############
        unchanged += 1
        if unchanged > 5:
            say('\n\nEARLY STOP\n')
            break

        ############
        # Training #
        ############
        say('\n\n\nEpoch: %d' % (epoch + 1))
        say('\n  TRAIN  ')
        model_api.train_all(batch_indices, evalset)

        ##############
        # Validating #
        ##############
        if dev_samples:
            say('\n\n  DEV  ')
            dev_acc_both, dev_acc_adr, dev_acc_res = model_api.predict_all(dev_samples)

            if dev_acc_both > best_dev_acc_both:
                unchanged = 0
                best_dev_acc_both = dev_acc_both
                acc_history[epoch + 1] = [(best_dev_acc_both, dev_acc_adr, dev_acc_res)]

                if argv.save:
                    #                    model_api.save_model(argv.output_fn)
                    model_api.save_params(argv.output_fn + '_epoch' + str(epoch))

        if test_samples:
            say('\n\n\r  TEST  ')
            test_acc_both, test_acc_adr, test_acc_res = model_api.predict_all(test_samples)

            if unchanged == 0:
                if epoch + 1 in acc_history:
                    acc_history[epoch + 1].append((test_acc_both, test_acc_adr, test_acc_res))
                else:
                    acc_history[epoch + 1] = [(test_acc_both, test_acc_adr, test_acc_res)]

        #####################
        # Show best results #
        #####################
        say('\n\tBEST ACCURACY HISTORY')
        for k, v in sorted(acc_history.items()):
            text = '\n\tEPOCH-{:>3} | DEV  Both:{:>7.2%}  Adr:{:>7.2%}  Res:{:>7.2%}'
            text = text.format(k, v[0][0], v[0][1], v[0][2])
            if len(v) == 2:
                text += ' | TEST  Both:{:>7.2%}  Adr:{:>7.2%}  Res:{:>7.2%}'
                text = text.format(v[1][0], v[1][1], v[1][2])
            say(text)


def main(argv):
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  default='train', help='train/test')

    ##############
    # Input Data #
    ##############
    parser.add_argument('--train_data',  help='path to a training data')
    parser.add_argument('--dev_data',  help='path to a development data')
    parser.add_argument('--test_data',  help='path to a test data')

    ###############
    # Output Data #
    ###############
    parser.add_argument('--output_fn',  default=None, help='file name of the model to be saved')
    parser.add_argument('--output', type=int, default=0, help='output results or not')

    #################
    # Save and Load #
    #################
    parser.add_argument('--save', type=int, default=1, help='parameters to be saved or not')
    parser.add_argument('--load_param', default=None, help='model file to be loaded')

    ######################
    # Initial Embeddings #
    ######################
    parser.add_argument('--learn_emb', default='learn', help='learn/freeze')
    parser.add_argument('--load_skip_emb', type=int, default=0, help='learn/freeze')
    parser.add_argument('--emb_type', default='mono', help='mono/multi')
    parser.add_argument('--lang', default='en', help='en/it...')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')

    #############################
    # Neural Network parameters #
    #############################
    parser.add_argument('--dim_emb',    type=int, default=50, help='dimension of embeddings')
    parser.add_argument('--dim_hidden', type=int, default=50, help='dimension of hidden layer')
    parser.add_argument('--loss', default='nll', help='loss')
    parser.add_argument('--unit', default='gru', help='unit')

    #######################
    # Training parameters #
    #######################
    parser.add_argument('--data_size', type=int, default=10000000,
                        help='number of threads used for the task')
    parser.add_argument('--sample_size', type=int, default=1,
                        help='number of division of samples used for the task')

    parser.add_argument('--n_cands', type=int, default=2, help='number of candidate responses')
    parser.add_argument('--n_prev_sents', type=int, default=5,
                        help='number of the sentences used for the prediction')
    parser.add_argument('--max_n_words', type=int, default=20,
                        help='maximum number of words for context/response')

    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--opt', default='adam', help='optimization method')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--activation', default='tanh', help='activation')
    parser.add_argument('--reg', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--attention', type=int, default=0, help='attention')

    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int,
                        default=20, help='learning minibatch size')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=100, help='n_epoch')
    parser.add_argument('--use_dropout', dest='use_dropout',
                        type=float, default=0.33, help='use_dropout')
    parser.add_argument('--init_alpha', dest='init_alpha',
                        type=float, default=0.002, help='init_alpha')
    parser.add_argument('--normalize', dest='normalize',
                        type=int, default=1, help='normalize')
    parser.add_argument('--use_pad_unk', dest='use_pad_unk',
                        type=int, default=1, help='use_pad_unk')

    args = parser.parse_args()
    argv = args
    batchsize = args.batchsize
    print args

    if args.test is not '':
        use_test = True
    else:
        use_test = False

    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp.random.seed(1234)

    say('\nADDRESSEE AND RESPONSE SELECTION SYSTEM START\n')

    ###############
    # Set dataset #
    ###############
    train_dataset, dev_dataset, test_dataset, words = get_datasets(argv)

    ##########################
    # Set initial embeddings #
    ##########################
    from ..nn import initialize_weights

    if argv.emb_type == 'mono':
        vocab_words, init_emb = load_init_emb(argv.init_emb, words)
        init_emb = initialize_weights(vocab_words.size(), argv.dim_emb)
    elif argv.emb_type == 'multi':
        vocab_words, init_emb = load_multi_ling_init_emb(argv.init_emb, argv.lang)
    elif argv.emb_type == 'mono_multi':
        # vocab_words, init_emb = load_init_emb(None, words)
        pre_vocab_words, pre_init_emb = load_multi_ling_init_emb(argv.init_emb, argv.lang)

        from ..ling.vocab import Vocab, PAD, UNK
        vocab_words = Vocab()
        vocab_words.add_word(PAD)
        vocab_words.add_word(UNK)
        for w in words:
            vocab_words.add_word(w)

        init_emb = initialize_weights(len(words), argv.dim_emb)
        say('\nVocab Size: %d' % len(words))
        # replace embeddings
        for w_idx, w in enumerate(vocab_words.w2i.keys()):
            if w in pre_vocab_words.w2i:
                pre_w_idx = pre_vocab_words.w2i[w]
                init_emb[w_idx] = pre_init_emb[pre_w_idx]
        say('\nVocab Size: %d' % len(vocab_words.w2i))
        print 'init_emb:', init_emb.shape
    elif argv.emb_type == 'common_multi':
        # vocab_words, init_emb = load_init_emb(None, words)
        pre_vocab_words, pre_init_emb = load_multi_ling_init_emb(argv.init_emb, argv.lang)
        from ..ling.vocab import Vocab, PAD, UNK
        vocab_words = Vocab()
        vocab_words.add_word(PAD)
        vocab_words.add_word(UNK)
        for w in words:
            # common words
            if w in pre_vocab_words.w2i:
                vocab_words.add_word(w)
        init_emb = initialize_weights(len(vocab_words.w2i), argv.dim_emb)
        say('\nVocab Size: %d' % len(vocab_words.w2i))
        # replace embeddings
        for w_idx, w in enumerate(vocab_words.w2i.keys()):
            pre_w_idx = pre_vocab_words.w2i[w]
            init_emb[w_idx] = pre_init_emb[pre_w_idx]

    # write vocab files
    vocab_file = argv.output_fn + '_' + argv.emb_type + '.vocab'
    _vcb_f = open(vocab_file, 'w')
    for word, word_id in sorted(vocab_words.w2i.items(), key=lambda x: x[1]):
        try:
            _vcb_f.write(word.encode('utf-8') + '\n')
        except:
            print 'error:', word
    _vcb_f.close()

    ###############
    # Set samples #
    ###############
    train_samples, dev_samples, test_samples, n_train_batches, evalset =\
        create_samples(argv, train_dataset, dev_dataset, test_dataset, vocab_words)
    del train_dataset
    del dev_dataset
    del test_dataset

    (train_contexts, train_contexts_length, train_responses,
     train_responses_length, train_agents_ids, train_n_agents,
     train_binned_n_agents, train_y_adr, train_y_res) = ch_util.pre_process(train_samples, xp)

    from chainer_net import MultiLingualConv
    n_vocab = vocab_words.size()
    model = MultiLingualConv(args, n_vocab)

    for epoch in xrange(args.n_epoch):
        # train
        model.cleargrads()
        chainer.config.train = True
        iteration_list = range(0, len(x_train), batchsize)
        perm = np.random.permutation(len(x_train))
        predict_lists = []
        sum_loss = 0.0
        for i_index, index in enumerate(iteration_list):
            xp_index = perm[index:index + batchsize]

            contexts = train_contexts[xp_index]
            responses = train_responses[xp_index]
            contexts_length = train_contexts_length[xp_index]
            responses_length = train_responses_length[xp_index]
            agents_ids = train_agents_ids[xp_index]
            n_agents = train_n_agents[xp_index]
            binned_n_agents = train_binned_n_agents[xp_index]
            y_adr = train_y_adr[xp_index]
            y_res = train_y_res[xp_index]

            sample = [contexts, contexts_length, responses, responses_length,
                      agents_ids, n_agents, binned_n_agents, y_adr, y_res]

            result = model(sample)

if __name__ == '__main__':
    main()