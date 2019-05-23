import numpy as np
import os

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

class Config():

    # general config
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 50
    dim_char = 50

    filename_embedding = "../data/korean_news_100MB_word2vec.txt".format(dim_word)
    filename_trimmed = "../data/korean_embedding.trimmed.npz".format(dim_word)

    use_pretrained = False
    use_chars = True

    # dataset
    filename_dev = "../data/NER_dev.txt"
    filename_test = "../data/NER_test.txt"
    filename_train = "../data/NER_train.txt"

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "../data/words.txt"
    filename_tags = "../data/tags.txt"
    filename_chars = "../data/chars.txt"

    # training
    nepochs          = 10
    dropout          = 0.5
    batch_size       = 20
    lr               = 0.005 #learning rate
    lr_decay         = 0.9
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 25 # lstm on chars
    hidden_size_lstm = 100 # lstm on word embeddings


    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # load if requested (default)
        if load:
            """Loads vocabulary, processing functions and embeddings

                   Supposes that build_data.py has been run successfully and that
                   the corresponding files have been created 

          """
            # 1. vocabulary
            self.vocab_words = load_vocab(self.filename_words)
            self.vocab_tags = load_vocab(self.filename_tags)
            self.vocab_chars = load_vocab(self.filename_chars)

            self.nwords = len(self.vocab_words)
            self.ntags = len(self.vocab_tags)
            self.nchars = len(self.vocab_chars)

            # 2. get processing functions that map str -> id
            self.processing_word = get_processing_word(self.vocab_words,
                                                       self.vocab_chars, chars=self.use_chars)
            self.processing_tag = get_processing_word(self.vocab_tags,
                                                      allow_unk=False)

            # 3. get pre-trained embeddings
            data = np.load(self.filename_trimmed)
            self.embeddings = data["embeddings"]

class data_read(object):

    def __init__(self, filename, processing_word=None, processing_tag=None,
                 max_iter=None):

        self.filename = filename    #file path
        self.processing_word = processing_word  #input word
        self.processing_tag = processing_tag    #input tag
        self.max_iter = max_iter    #maximum number of sentence
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0  :
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    # try:
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx

            # except IOError:
            # raise MyIOError(filename)
    return d

def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """

    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))

def data_build():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word()

    # Generators
    dev   = data_read(config.filename_dev, processing_word)
    test  = data_read(config.filename_test, processing_word)
    train = data_read(config.filename_train, processing_word)

    # Build Word and Tag vocab #-> get_
    print("Building vocab(data vocabulary)...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in [train, dev, test]:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))

    print("Building vocab(embedding vocabulary)...")
    vocab_embed = set()
    with open(config.filename_embedding) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab_embed.add(word)
    print("- done. {} tokens".format(len(vocab_embed)))

    vocab = vocab_words & vocab_embed
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    print("Writing vocab(# of covered words with pre-trained embedding)...")
    write_vocab(vocab, config.filename_words)
    print("Writing vocab(# of NEtag)...")
    write_vocab(vocab_tags, config.filename_tags)

    vocab_for_embed = load_vocab(config.filename_words)
    embeddings = np.zeros([len(vocab_for_embed), config.dim_word])
    with open(config.filename_embedding) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab_for_embed:
                word_idx = vocab_for_embed[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(config.filename_trimmed, embeddings=embeddings)

    # Build and save char vocab
    train = data_read(config.filename_train)
    vocab_chars = set()
    for words, _ in train:
        for word in words:
            vocab_chars.update(word)
    #vocab_chars = get_char_vocab(train)
    print("Writing vocab(# of char)...")
    write_vocab(vocab_chars, config.filename_chars)

def get_processing_word(vocab_words=None, vocab_chars=None,
                      chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 2. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f
