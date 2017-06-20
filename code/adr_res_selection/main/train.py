import numpy as np
import theano

from ..preprocess import convert_word_into_id, get_samples, numpy_format, theano_shared_format
from ..model import ModelAPI
from ..utils import say, load_dataset, load_init_emb, load_multi_ling_init_emb


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
                                n_prev_sents=n_prev_sents, max_n_words=max_n_words)
    dev_samples = get_samples(threads=dev_samples, n_prev_sents=n_prev_sents,
                              max_n_words=max_n_words, test=True)
    test_samples = get_samples(threads=test_samples, n_prev_sents=n_prev_sents,
                               max_n_words=max_n_words, test=True)

    ###################################
    # Limit the used training samples #
    ###################################
    if sample_size > 1:
        np.random.shuffle(train_samples)
        train_samples = train_samples[: (len(train_samples) / sample_size)]

    ###################################
    # Create theano-formatted samples #
    ###################################
    train_samples, n_train_batches, evalset = theano_shared_format(train_samples, batch_size)
    dev_samples = numpy_format(dev_samples, batch_size, test=True)
    test_samples = numpy_format(test_samples, batch_size, test=True)

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
        vocab_words, init_emb = load_init_emb(None, words)
        pre_vocab_words, pre_init_emb = load_multi_ling_init_emb(argv.init_emb, argv.lang)
        init_emb = initialize_weights(len(words), argv.dim_emb)
        say('\nVocab Size: %d' % len(words))
        # replace embeddings
        for w in vocab_words.w2i.keys():
            if w in pre_vocab_words.w2i:
                w_idx = vocab_words.w2i[w]
                pre_w_idx = pre_vocab_words.w2i[w]
                init_emb[w_idx, :] = pre_init_emb[pre_w_idx, :]
        init_emb = np.asarray(init_emb, dtype=theano.config.floatX)
    elif argv.emb_type == 'common_multi':
        # vocab_words, init_emb = load_init_emb(None, words)
        pre_vocab_words, pre_init_emb = load_multi_ling_init_emb(argv.init_emb, argv.lang)
        from ..ling.vocab import Vocab, PAD, UNK
        common_words = []
        vocab_words = Vocab()
        vocab_words.add_word(PAD)
        vocab_words.add_word(UNK)
        for w in words:
            if w in pre_vocab_words.w2i:
                common_words.append(w)
                vocab_words.add_word(w)

        init_emb = initialize_weights(len(common_words), argv.dim_emb)
        say('\nVocab Size: %d' % len(common_words))
        # replace embeddings
        for w_idx, w in enumerate(common_words):
            pre_w_idx = pre_vocab_words.w2i[w]
            init_emb[w_idx] = pre_init_emb[pre_w_idx]
        print 'init_emb:', init_emb.shape
        print 'len(init_emb[0]):', len(init_emb[0])
        init_emb = np.asarray(init_emb, dtype=theano.config.floatX)

        print 'init_emb:', init_emb

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

    ###############
    # Set a model #
    ###############
    model_api = ModelAPI(argv, init_emb, vocab_words, argv.n_prev_sents)
    model_api.set_model()
    if argv.load_param is not None:
        model_api.load_params(load_skip_emb=argv.load_skip_emb)

    model_api.set_train_f(train_samples)
    model_api.set_test_f()

    train(argv, model_api, n_train_batches, evalset, dev_samples, test_samples)
