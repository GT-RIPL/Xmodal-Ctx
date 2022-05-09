# coding: utf8
from collections import Counter, OrderedDict
from torch.utils.data.dataloader import default_collate
from itertools import chain
import six
import torch
import numpy as np
import h5py
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool

from .dataset import Dataset
from .vocab import Vocab
from .utils import get_tokenizer


class RawField(object):
    """ Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """ Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [batch, ]
        else:
            batch = list(zip(*batch))

        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out


class TextField(RawField):
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    def __init__(self, use_vocab=True, init_token=None, eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False, tokenize=(lambda s: s.split()),
                 remove_punctuation=False, include_lengths=False, batch_first=True, pad_token="<pad>",
                 unk_token="<unk>", pad_first=False, truncate_first=False, vectors=None, nopoints=True):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.lower = lower
        self.tokenize = get_tokenizer(tokenize)
        self.remove_punctuation = remove_punctuation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors
        if nopoints:
            self.punctuations.append("..")

        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        if six.PY2 and isinstance(x, six.string_types) and not isinstance(x, six.text_type):
            x = six.text_type(x, encoding='utf-8')
        if self.lower:
            x = six.text_type.lower(x)
        x = self.tokenize(x.rstrip('\n'))
        if self.remove_punctuation:
            x = [w for w in x if w not in self.punctuations]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(OrderedDict.fromkeys([
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None]))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.init_token, self.eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x)) +
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token]) +
                    list(x[-max_len:] if self.truncate_first else x[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat([torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr])

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode([word_idxs, ], join_words)[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs, ], join_words)[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []
        for wis in word_idxs:
            caption = []
            for wi in wis:
                word = self.vocab.itos[int(wi)]
                if word == self.eos_token:
                    break
                caption.append(word)
            if join_words:
                caption = ' '.join(caption)
            captions.append(caption)
        return captions


class ImageDetectionsField(RawField):
    def __init__(self, obj_file=None, max_detections=50, preload=False, preprocessing=None, postprocessing=None):
        self.max_detections = max_detections

        self.preload = preload
        if preload:
            self.obj = self.load(obj_file)
        else:
            self.obj = h5py.File(obj_file, "r")

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)
    
    @staticmethod
    def __load(obj_file, k):
        return obj_file[f"{k}/obj_features"][:]

    def load(self, obj_file):
        print(f"Preload features from {str(obj_file)}...")
        obj_file = h5py.File(obj_file, "r")
        pool = Pool(128)
        results = {
            k: pool.apply_async(self.__load, args=(obj_file, k))
            for k in obj_file.keys()
        }
        obj = {k: v.get() for k, v in tqdm(results.items())}
        obj_file.close()

        return obj
    
    def preprocess(self, image_id):
        if self.preload:
            obj = self.obj[str(image_id)]
        else:
            obj = self.obj[f"{image_id}/obj_features"][:]
        n, d = obj.shape

        delta = self.max_detections - n
        if delta > 0:
            p = np.zeros((delta, d), dtype=obj.dtype)
            obj = np.concatenate([obj, p], axis=0)
        elif delta < 0:
            obj = obj[:self.max_detections]
        
        return torch.FloatTensor(obj)  # obj.shape = (50, 2054)
    
    def process(self, batch, *args, **kwargs):
        batch = torch.stack(batch)
        max_len = torch.max(torch.sum(torch.sum(batch, dim=-1) != 0, dim=-1))

        return batch[:, :max_len]


class TxtCtxField(RawField):
    def __init__(self, ctx_file, k=4, preload=False, preprocessing=None, postprocessing=None):
        self.k = k

        self.preload = preload
        if preload:
            self.ctx = self.load(ctx_file)
        else:
            self.ctx = ctx_file

        super(TxtCtxField, self).__init__(preprocessing, postprocessing)
    
    def __load(self, ctx, k):
        ctx_whole_f = ctx[f"{k}/whole/features"][:self.k]

        ctx_five_f = ctx[f"{k}/five/features"][:, :self.k]
        ctx_five_p = np.tile(np.arange(5)[:, None], (1, self.k))
        ctx_five_f = ctx_five_f.reshape((5*self.k, -1))
        ctx_five_p = ctx_five_p.reshape((5*self.k, ))

        ctx_nine_f = ctx[f"{k}/nine/features"][:, :self.k]
        ctx_nine_p = np.tile(np.arange(9)[:, None], (1, self.k))
        ctx_nine_f = ctx_nine_f.reshape((9*self.k, -1))
        ctx_nine_p = ctx_nine_p.reshape((9*self.k, ))

        return {
            "whole": {"features": ctx_whole_f},
            "five": {"features": ctx_five_f, "positions": ctx_five_p},
            "nine": {"features": ctx_nine_f, "positions": ctx_nine_p}
        }
    
    def load(self, ctx_file):
        print(f"Preload features from {str(ctx_file)}...")
        ctx_file = h5py.File(ctx_file, "r")
        pool = Pool(128)
        results = {
            k: pool.apply_async(self.__load, args=(ctx_file, k))
            for k in ctx_file.keys()
        }
        ctx = {k: v.get() for k, v in tqdm(results.items())}
        ctx_file.close()

        return ctx
    
    def preprocess(self, x):
        if self.preload:
            ctx_whole_f = self.ctx[str(x)]["whole"]["features"]
            ctx_five_f = self.ctx[str(x)]["five"]["features"]
            ctx_five_p = self.ctx[str(x)]["five"]["positions"]
            ctx_nine_f = self.ctx[str(x)]["nine"]["features"]
            ctx_nine_p = self.ctx[str(x)]["nine"]["positions"]
        else:
            ctx = h5py.File(self.ctx, "r")
            data = self.__load(ctx, x)
            ctx_whole_f = data["whole"]["features"]
            ctx_five_f = data["five"]["features"]
            ctx_five_p = data["five"]["positions"]
            ctx_nine_f = data["nine"]["features"]
            ctx_nine_p = data["nine"]["positions"]
            ctx.close()

        ctx_whole_f = torch.FloatTensor(ctx_whole_f)
        ctx_five_f = torch.FloatTensor(ctx_five_f)
        ctx_five_p = torch.LongTensor(ctx_five_p)
        ctx_nine_f = torch.FloatTensor(ctx_nine_f)
        ctx_nine_p = torch.LongTensor(ctx_nine_p)

        return ctx_whole_f, ctx_five_f, ctx_five_p, ctx_nine_f, ctx_nine_p
            
    def process(self, batch, *args, **kwargs):
        ctx_whole_f, ctx_five_f, ctx_five_p, ctx_nine_f, ctx_nine_p = list(zip(*batch))
        
        ctx_whole_f = torch.stack(ctx_whole_f)
        ctx_five_f = torch.stack(ctx_five_f)
        ctx_nine_f = torch.stack(ctx_nine_f)

        ctx_whole_p = torch.zeros((len(ctx_whole_f), len(ctx_whole_f[0])), dtype=torch.long)
        ctx_five_p = torch.stack(ctx_five_p)
        ctx_nine_p = torch.stack(ctx_nine_p)

        return {
            "whole": {"embed": ctx_whole_f, "pos": ctx_whole_p},
            "five": {"embed": ctx_five_f, "pos": ctx_five_p},
            "nine": {"embed": ctx_nine_f, "pos": ctx_nine_p},
        }


class VisCtxField(RawField):
    def __init__(self, ctx_file, preload=False, preprocessing=None, postprocessing=None):
        self.preload = preload
        if preload:
            self.ctx = self.load(ctx_file)
        else:
            self.ctx = h5py.File(ctx_file, "r")
        
        f = h5py.File(ctx_file, "r")
        self.fdim = f.attrs["fdim"]
        f.close()

        super(VisCtxField, self).__init__(preprocessing, postprocessing)

    def __load(self, ctx, k):
        return ctx[str(k)][:]

    def load(self, ctx_file):
        print(f"Preload features from {str(ctx_file)}...")
        ctx_file = h5py.File(ctx_file, "r")
        pool = Pool(128)
        results = {
            k: pool.apply_async(self.__load, args=(ctx_file, k))
            for k in ctx_file.keys()
        }
        ctx = {k: v.get() for k, v in tqdm(results.items())}
        ctx_file.close()

        return ctx

    def preprocess(self, image_id):
        if self.preload:
            ctx = self.ctx[str(image_id)]
        else:
            ctx = self.ctx[str(image_id)][:]

        return torch.FloatTensor(ctx)

    def process(self, batch, *args, **kwargs):
        return torch.stack(batch)
