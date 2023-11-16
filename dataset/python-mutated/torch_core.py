from __future__ import annotations
from .imports import *
from .torch_imports import *
from packaging.version import parse
__all__ = ['norm_types', 'setup_cuda', 'subplots', 'show_image', 'show_titled_image', 'show_images', 'ArrayBase', 'ArrayImageBase', 'ArrayImage', 'ArrayImageBW', 'ArrayMask', 'tensor', 'set_seed', 'get_random_states', 'set_random_states', 'no_random', 'unsqueeze', 'unsqueeze_', 'apply', 'maybe_gather', 'to_detach', 'to_half', 'to_float', 'default_device', 'to_device', 'to_cpu', 'to_np', 'to_concat', 'TensorBase', 'TensorImageBase', 'TensorImage', 'TensorImageBW', 'TensorMask', 'TensorFlowField', 'TensorCategory', 'TensorMultiCategory', 'TitledTensorScalar', 'concat', 'Chunks', 'show_title', 'ShowTitle', 'TitledInt', 'TitledFloat', 'TitledStr', 'TitledTuple', 'get_empty_df', 'display_df', 'get_first', 'one_param', 'item_find', 'find_device', 'find_bs', 'np_func', 'Module', 'get_model', 'one_hot', 'one_hot_decode', 'params', 'trainable_params', 'norm_bias_params', 'batch_to_samples', 'logit', 'num_distrib', 'rank_distrib', 'distrib_barrier', 'base_doc', 'doc', 'nested_reorder', 'flatten_check', 'make_cross_image', 'show_image_batch', 'requires_grad', 'init_default', 'cond_init', 'apply_leaf', 'apply_init', 'script_use_ctx', 'script_save_ctx', 'script_fwd', 'script_bwd', 'grad_module', 'ismin_torch', 'notmax_torch', 'progress_bar', 'master_bar']
_all_ = ['progress_bar', 'master_bar']
defaults.benchmark = True

def setup_cuda(benchmark=defaults.benchmark):
    if False:
        return 10
    'Sets the main cuda device and sets `cudnn.benchmark` to `benchmark`'
    if torch.cuda.is_available():
        if torch.cuda.current_device() == 0:
            def_gpu = int(os.environ.get('DEFAULT_GPU') or 0)
            if torch.cuda.device_count() >= def_gpu:
                torch.cuda.set_device(def_gpu)
        torch.backends.cudnn.benchmark = benchmark

@delegates(plt.subplots, keep=True)
def subplots(nrows: int=1, ncols: int=1, figsize: tuple=None, imsize: int=3, suptitle: str=None, **kwargs) -> (plt.Figure, plt.Axes):
    if False:
        return 10
    'Returns a figure and set of subplots to display images of `imsize` inches'
    if figsize is None:
        h = nrows * imsize if suptitle is None or imsize > 2 else nrows * imsize + 0.6
        figsize = (ncols * imsize, h)
    (fig, ax) = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None:
        fig.suptitle(suptitle)
    if nrows * ncols == 1:
        ax = array([ax])
    return (fig, ax)

def _fig_bounds(x):
    if False:
        for i in range(10):
            print('nop')
    r = x // 32
    return min(5, max(1, r))

@delegates(plt.Axes.imshow, keep=True, but=['shape', 'imlim'])
def show_image(im, ax=None, figsize=None, title=None, ctx=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Show a PIL or PyTorch image on `ax`.'
    if hasattrs(im, ('data', 'cpu', 'permute')):
        im = im.data.cpu()
        if im.shape[0] < 5:
            im = im.permute(1, 2, 0)
    elif not isinstance(im, np.ndarray):
        im = array(im)
    if im.shape[-1] == 1:
        im = im[..., 0]
    ax = ifnone(ax, ctx)
    if figsize is None:
        figsize = (_fig_bounds(im.shape[0]), _fig_bounds(im.shape[1]))
    if ax is None:
        (_, ax) = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    return ax

@delegates(show_image, keep=True)
def show_titled_image(o, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Call `show_image` destructuring `o` to `(img,title)`'
    show_image(o[0], title=str(o[1]), **kwargs)

@delegates(subplots)
def show_images(ims, nrows=1, ncols=None, titles=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Show all images `ims` as subplots with `rows` using `titles`.'
    if ncols is None:
        ncols = int(math.ceil(len(ims) / nrows))
    if titles is None:
        titles = [None] * len(ims)
    axs = subplots(nrows, ncols, **kwargs)[1].flat
    for (im, t, ax) in zip(ims, titles, axs):
        show_image(im, ax=ax, title=t)

class ArrayBase(ndarray):
    """An `ndarray` that can modify casting behavior"""

    @classmethod
    def _before_cast(cls, x):
        if False:
            i = 10
            return i + 15
        return x if isinstance(x, ndarray) else array(x)

class ArrayImageBase(ArrayBase):
    """Base class for arrays representing images"""
    _show_args = {'cmap': 'viridis'}

    def show(self, ctx=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return show_image(self, ctx=ctx, **{**self._show_args, **kwargs})

class ArrayImage(ArrayImageBase):
    """An array representing an image"""
    pass

class ArrayImageBW(ArrayImage):
    """An array representing an image"""
    _show_args = {'cmap': 'Greys'}

class ArrayMask(ArrayImageBase):
    """An array representing an image mask"""
    _show_args = {'alpha': 0.5, 'cmap': 'tab20', 'interpolation': 'nearest'}

@patch
def __array_eq__(self: Tensor, b):
    if False:
        while True:
            i = 10
    return torch.equal(self, b) if self.dim() else self == b

def _array2tensor(x, requires_grad=False, pin_memory=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if x.dtype == np.uint16:
        x = x.astype(np.float32)
    if sys.platform == 'win32' and x.dtype == int:
        x = x.astype(np.int64)
    t = torch.as_tensor(x, **kwargs)
    t.requires_grad_(requires_grad)
    if pin_memory:
        t.pin_memory()
    return t

@use_kwargs_dict(dtype=None, device=None, requires_grad=False, pin_memory=False)
def tensor(x, *rest, **kwargs):
    if False:
        return 10
    'Like `torch.as_tensor`, but handle lists too, and can pass multiple vector elements directly.'
    if len(rest):
        x = (x,) + rest
    res = x if isinstance(x, Tensor) else torch.tensor(x, **kwargs) if isinstance(x, (tuple, list, numbers.Number)) else _array2tensor(x, **kwargs) if isinstance(x, ndarray) else as_tensor(x.values, **kwargs) if isinstance(x, (pd.Series, pd.DataFrame)) else _array2tensor(array(x), **kwargs)
    if res.dtype is torch.float64:
        return res.float()
    return res

def set_seed(s, reproducible=False):
    if False:
        i = 10
        return i + 15
    'Set random seed for `random`, `torch`, and `numpy` (where available)'
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2 ** 32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_random_states():
    if False:
        while True:
            i = 10
    'Gets states for `random`, `torch`, and `numpy` random number generators'
    return {'random_state': random.getstate(), 'numpy_state': np.random.get_state(), 'torch_state': torch.get_rng_state(), 'torch_cuda_state': torch.cuda.get_rng_state_all(), 'torch_deterministic': torch.backends.cudnn.deterministic, 'torch_benchmark': torch.backends.cudnn.benchmark}

def set_random_states(random_state, numpy_state, torch_state, torch_cuda_state, torch_deterministic, torch_benchmark):
    if False:
        return 10
    'Set states for `random`, `torch`, and `numpy` random number generators'
    random.setstate(random_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state_all(torch_cuda_state)
    torch.backends.cudnn.deterministic = torch_deterministic
    torch.backends.cudnn.benchmark = torch_benchmark

@contextmanager
def no_random(seed=42, reproducible=True):
    if False:
        return 10
    'Stores and retrieves state of random number generators. Sets random seed for `random`, `torch`, and `numpy`.'
    states = get_random_states()
    set_seed(seed, reproducible=reproducible)
    try:
        yield
    finally:
        set_random_states(**states)

def unsqueeze(x, dim=-1, n=1):
    if False:
        for i in range(10):
            print('nop')
    'Same as `torch.unsqueeze` but can add `n` dims'
    for _ in range(n):
        x = x.unsqueeze(dim)
    return x

def unsqueeze_(x, dim=-1, n=1):
    if False:
        while True:
            i = 10
    'Same as `torch.unsqueeze_` but can add `n` dims'
    for _ in range(n):
        x.unsqueeze_(dim)
    return x

def _fa_rebuild_tensor(cls, *args, **kwargs):
    if False:
        return 10
    return cls(torch._utils._rebuild_tensor_v2(*args, **kwargs))

def _fa_rebuild_qtensor(cls, *args, **kwargs):
    if False:
        while True:
            i = 10
    return cls(torch._utils._rebuild_qtensor(*args, **kwargs))

def apply(func, x, *args, **kwargs):
    if False:
        return 10
    'Apply `func` recursively to `x`, passing on args'
    if is_listy(x):
        return type(x)([apply(func, o, *args, **kwargs) for o in x])
    if isinstance(x, (dict, MutableMapping)):
        return {k: apply(func, v, *args, **kwargs) for (k, v) in x.items()}
    res = func(x, *args, **kwargs)
    return res if x is None else retain_type(res, x)

def maybe_gather(x, axis=0):
    if False:
        while True:
            i = 10
    'Gather copies of `x` on `axis` (if training is distributed)'
    if num_distrib() <= 1:
        return x
    ndim = x.ndim
    res = [x.new_zeros(*(x.shape if ndim > 0 else (1,))) for _ in range(num_distrib())]
    torch.distributed.all_gather(res, x.contiguous() if ndim > 0 else x[None])
    return torch.cat(res, dim=axis) if ndim > 0 else torch.cat(res, dim=axis).mean()

def to_detach(b, cpu=True, gather=True):
    if False:
        while True:
            i = 10
    'Recursively detach lists of tensors in `b `; put them on the CPU if `cpu=True`.'

    def _inner(x, cpu=True, gather=True):
        if False:
            while True:
                i = 10
        if not isinstance(x, Tensor):
            return x
        x = x.detach()
        if gather:
            x = maybe_gather(x)
        return x.cpu() if cpu else x
    return apply(_inner, b, cpu=cpu, gather=gather)

def to_half(b):
    if False:
        print('Hello World!')
    'Recursively map floating point tensors in `b ` to FP16.'
    return apply(lambda x: x.half() if torch.is_floating_point(x) else x, b)

def to_float(b):
    if False:
        i = 10
        return i + 15
    'Recursively map floating point tensors in `b ` to float.'
    return apply(lambda x: x.float() if torch.is_floating_point(x) else x, b)
defaults.use_cuda = None

def _has_mps():
    if False:
        while True:
            i = 10
    if nested_attr(torch, 'backends.mps.is_available', noop)():
        return True
    return getattr(torch, 'has_mps', False)

def default_device(use=-1):
    if False:
        return 10
    'Return or set default device; `use_cuda`: -1 - CUDA/mps if available; True - error if not available; False - CPU'
    if use == -1:
        use = defaults.use_cuda
    else:
        defaults.use_cuda = use
    if use is None:
        if torch.cuda.is_available() or _has_mps():
            use = True
    if use:
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        if _has_mps():
            return torch.device('mps')
    return torch.device('cpu')

def to_device(b, device=None, non_blocking=False):
    if False:
        print('Hello World!')
    'Recursively put `b` on `device`.'
    if defaults.use_cuda == False:
        device = 'cpu'
    elif device is None:
        device = default_device()

    def _inner(o):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(o, Tensor):
            return o.to(device, non_blocking=non_blocking)
        return o
    return apply(_inner, b)

def to_cpu(b):
    if False:
        while True:
            i = 10
    'Recursively map tensors in `b ` to the cpu.'
    return to_device(b, 'cpu')

def to_np(x):
    if False:
        i = 10
        return i + 15
    'Convert a tensor to a numpy array.'
    return apply(lambda o: o.data.cpu().numpy(), x)

def to_concat(xs, dim=0):
    if False:
        print('Hello World!')
    'Concat the element in `xs` (recursively if they are tuples/lists of tensors)'
    if not xs:
        return xs
    if is_listy(xs[0]):
        return type(xs[0])([to_concat([x[i] for x in xs], dim=dim) for i in range_of(xs[0])])
    if isinstance(xs[0], dict):
        return {k: to_concat([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
    try:
        return retain_type(torch.cat(xs, dim=dim), xs[0])
    except:
        return sum([L((retain_type(o_.index_select(dim, tensor(i)).squeeze(dim), xs[0]) for i in range_of(o_))) for o_ in xs], L())
_torch_version = parse(torch.__version__)
_torch_20 = parse('2.0')
_torch_113 = parse('1.13')
_torch_112 = parse('1.12')

@patch
def set_meta(self: Tensor, x, as_copy=False):
    if False:
        return 10
    'Set all metadata in `__dict__`'
    if not hasattr(x, '__dict__'):
        return
    self.__dict__ = copy(x.__dict__) if as_copy else x.__dict__
if not hasattr(torch, 'as_subclass'):
    torch.as_subclass = torch.Tensor.as_subclass

@patch
def as_subclass(self: Tensor, typ):
    if False:
        for i in range(10):
            print('nop')
    'Cast to `typ` and include `__dict__` and meta'
    return retain_meta(self, torch.as_subclass(self, typ))

def _torch_handled(args, opt, func):
    if False:
        while True:
            i = 10
    if func not in opt:
        return False
    for oks in opt[func]:
        if all((isinstance(arg, ok) for (arg, ok) in zip(args, oks) if ok)):
            return True

def _rebuild_from_type(func, type, args, dict):
    if False:
        while True:
            i = 10
    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret

def _find_args(x):
    if False:
        return 10
    x0 = x[0] if is_listy(x[0]) and x[0] else x
    return [a for a in x0 if hasattr(a, '__dict__')]

class TensorBase(Tensor):
    """A `Tensor` which support subclass pickling, and maintains metadata when casting or after methods"""
    (debug, _opt) = (False, defaultdict(list))

    def __new__(cls, x, **kwargs):
        if False:
            print('Hello World!')
        res = cast(tensor(x), cls)
        for (k, v) in kwargs.items():
            setattr(res, k, v)
        return res

    @classmethod
    def _before_cast(cls, x):
        if False:
            print('Hello World!')
        return tensor(x)

    def __repr__(self):
        if False:
            print('Hello World!')
        return re.sub('tensor', self.__class__.__name__, super().__repr__())

    def __reduce_ex__(self, proto):
        if False:
            i = 10
            return i + 15
        if _torch_version >= _torch_20:
            return super().__reduce_ex__(proto)
        else:
            torch.utils.hooks.warn_if_has_hooks(self)
            args = (self.storage(), self.storage_offset(), tuple(self.size()), self.stride())
            if self.is_quantized:
                args = args + (self.q_scale(), self.q_zero_point())
            args = args + (self.requires_grad, OrderedDict())
            f = torch._utils._rebuild_qtensor if self.is_quantized else torch._utils._rebuild_tensor_v2
            return (_rebuild_from_type, (f, type(self), args, self.__dict__))

    @classmethod
    def register_func(cls, func, *oks):
        if False:
            return 10
        cls._opt[func].append(oks)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        if cls.debug and func.__name__ not in ('__str__', '__repr__'):
            print(func, types, args, kwargs)
        if _torch_handled(args, cls._opt, func):
            types = (torch.Tensor,)
        res = super().__torch_function__(func, types, args, ifnone(kwargs, {}))
        dict_objs = _find_args(args) if args else _find_args(list(kwargs.values()))
        if issubclass(type(res), TensorBase) and dict_objs:
            res.set_meta(dict_objs[0], as_copy=True)
        elif dict_objs and is_listy(res):
            [r.set_meta(dict_objs[0], as_copy=True) for r in res if issubclass(type(r), TensorBase)]
        return res

    def new_tensor(self, size, dtype=None, device=None, requires_grad=False):
        if False:
            print('Hello World!')
        cls = type(self)
        return self.as_subclass(Tensor).new_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

    def new_ones(self, data, dtype=None, device=None, requires_grad=False):
        if False:
            return 10
        cls = type(self)
        return self.as_subclass(Tensor).new_ones(data, dtype=dtype, device=device, requires_grad=requires_grad).as_subclass(cls)

    def new(self, x=None):
        if False:
            for i in range(10):
                print('nop')
        cls = type(self)
        res = self.as_subclass(Tensor).new() if x is None else self.as_subclass(Tensor).new(x)
        return res.as_subclass(cls)

    def requires_grad_(self, requires_grad=True):
        if False:
            return 10
        self.requires_grad = requires_grad
        return self

    def clone(self, *, memory_format=None):
        if False:
            return 10
        cls = type(self)
        return self.as_subclass(Tensor).clone(memory_format=memory_format).as_subclass(cls)

    def new_empty(self, size, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        if False:
            i = 10
            return i + 15
        cls = type(self)
        if _torch_version < _torch_113 and layout is None:
            layout = torch.strided
        if _torch_version < _torch_112:
            return super().new_empty(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
        return self.as_subclass(Tensor).new_empty(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(cls)

    def new_empty(self, *size, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False):
        if False:
            while True:
                i = 10
        cls = type(self)
        if _torch_version < _torch_113 and layout is None:
            layout = torch.strided
        if _torch_version < _torch_112:
            return super().new_empty(*size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad)
        return self.as_subclass(Tensor).new_empty(*size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory, requires_grad=requires_grad).as_subclass(cls)

class TensorImageBase(TensorBase):
    _show_args = ArrayImageBase._show_args

    def show(self, ctx=None, **kwargs):
        if False:
            while True:
                i = 10
        return show_image(self, ctx=ctx, **{**self._show_args, **kwargs})

class TensorImage(TensorImageBase):
    pass

class TensorImageBW(TensorImage):
    _show_args = ArrayImageBW._show_args

class TensorMask(TensorImageBase):
    _show_args = ArrayMask._show_args

    def show(self, ctx=None, **kwargs):
        if False:
            i = 10
            return i + 15
        codes = getattr(self, 'codes', None)
        if codes is not None:
            kwargs = merge({'vmin': 0, 'vmax': len(codes)}, kwargs)
        return super().show(ctx=ctx, **kwargs)
for o in (Tensor.__getitem__, Tensor.__ne__, Tensor.__eq__, Tensor.add, Tensor.sub, Tensor.mul, Tensor.div, Tensor.__rsub__, Tensor.__radd__, Tensor.matmul, Tensor.bmm):
    TensorBase.register_func(o, TensorMask, TensorImageBase)
    TensorBase.register_func(o, TensorImageBase, TensorMask)
TensorMask.register_func(torch.einsum, str, TensorImageBase, TensorMask)
TensorMask.register_func(torch.einsum, str, TensorMask, TensorImageBase)

class TensorFlowField(TensorBase):
    pass
TensorImage.register_func(F.grid_sample, TensorImageBase, TensorFlowField)

class TensorCategory(TensorBase):
    pass
TensorBase.register_func(Tensor.__getitem__, TensorImageBase, TensorCategory)

class TensorMultiCategory(TensorCategory):
    pass

class TitledTensorScalar(TensorBase):
    """A tensor containing a scalar that has a `show` method"""

    def show(self, **kwargs):
        if False:
            i = 10
            return i + 15
        show_title(self.item(), **kwargs)

@patch
def tensored(self: L):
    if False:
        return 10
    '`mapped(tensor)`'
    return self.map(tensor)

@patch
def stack(self: L, dim=0):
    if False:
        for i in range(10):
            print('nop')
    'Same as `torch.stack`'
    return torch.stack(list(self.tensored()), dim=dim)

@patch
def cat(self: L, dim=0):
    if False:
        print('Hello World!')
    'Same as `torch.cat`'
    return torch.cat(list(self.tensored()), dim=dim)

def concat(*ls):
    if False:
        while True:
            i = 10
    'Concatenate tensors, arrays, lists, or tuples'
    if not len(ls):
        return []
    it = ls[0]
    if isinstance(it, torch.Tensor):
        res = torch.cat(ls)
    elif isinstance(it, ndarray):
        res = np.concatenate(ls)
    else:
        res = itertools.chain.from_iterable(map(L, ls))
        if isinstance(it, (tuple, list)):
            res = type(it)(res)
        else:
            res = L(res)
    return retain_type(res, it)

class Chunks:
    """Slice and int indexing into a list of lists"""

    def __init__(self, chunks, lens=None):
        if False:
            return 10
        self.chunks = chunks
        self.lens = L(map(len, self.chunks) if lens is None else lens)
        self.cumlens = np.cumsum(0 + self.lens)
        self.totlen = self.cumlens[-1]

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        if isinstance(i, slice):
            return retain_type(self.getslice(i), old=self.chunks[0])
        (di, idx) = self.doc_idx(i)
        return retain_type(self.chunks[di][idx], old=self.chunks[0])

    def getslice(self, i):
        if False:
            while True:
                i = 10
        (st_d, st_i) = self.doc_idx(ifnone(i.start, 0))
        (en_d, en_i) = self.doc_idx(ifnone(i.stop, self.totlen + 1))
        res = [self.chunks[st_d][st_i:en_i if st_d == en_d else sys.maxsize]]
        for b in range(st_d + 1, en_d):
            res.append(self.chunks[b])
        if st_d != en_d and en_d < len(self.chunks):
            res.append(self.chunks[en_d][:en_i])
        return concat(*res)

    def doc_idx(self, i):
        if False:
            print('Hello World!')
        if i < 0:
            i = self.totlen + i
        docidx = np.searchsorted(self.cumlens, i + 1) - 1
        cl = self.cumlens[docidx]
        return (docidx, i - cl)

def show_title(o, ax=None, ctx=None, label=None, color='black', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Set title of `ax` to `o`, or print `o` if `ax` is `None`'
    ax = ifnone(ax, ctx)
    if ax is None:
        print(o)
    elif hasattr(ax, 'set_title'):
        t = ax.title.get_text()
        if len(t) > 0:
            o = t + '\n' + str(o)
        ax.set_title(o, color=color)
    elif isinstance(ax, pd.Series):
        while label in ax:
            label += '_'
        ax = pd.concat([ax, pd.Series({label: o})])
    return ax

class ShowTitle:
    """Base class that adds a simple `show`"""
    _show_args = {'label': 'text'}

    def show(self, ctx=None, **kwargs):
        if False:
            while True:
                i = 10
        'Show self'
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledInt(Int, ShowTitle):
    _show_args = {'label': 'text'}

    def show(self, ctx=None, **kwargs):
        if False:
            print('Hello World!')
        'Show self'
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledFloat(Float, ShowTitle):
    _show_args = {'label': 'text'}

    def show(self, ctx=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Show self'
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledStr(Str, ShowTitle):
    _show_args = {'label': 'text'}

    def show(self, ctx=None, **kwargs):
        if False:
            return 10
        'Show self'
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))

class TitledTuple(fastuple, ShowTitle):
    _show_args = {'label': 'text'}

    def show(self, ctx=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Show self'
        return show_title(str(self), ctx=ctx, **merge(self._show_args, kwargs))
add_docs(TitledInt, 'An `int` with `show`')
add_docs(TitledStr, 'An `str` with `show`')
add_docs(TitledFloat, 'A `float` with `show`')
add_docs(TitledTuple, 'A `fastuple` with `show`')

@patch
def truncate(self: TitledStr, n):
    if False:
        for i in range(10):
            print('nop')
    'Truncate self to `n`'
    words = self.split(' ')[:n]
    return TitledStr(' '.join(words))
if not hasattr(pd.DataFrame, '_old_init'):
    pd.DataFrame._old_init = pd.DataFrame.__init__

@patch
def __init__(self: pd.DataFrame, data=None, index=None, columns=None, dtype=None, copy=None):
    if False:
        while True:
            i = 10
    if data is not None and isinstance(data, Tensor):
        data = to_np(data)
    self._old_init(data, index=index, columns=columns, dtype=dtype, copy=copy)

def get_empty_df(n):
    if False:
        print('Hello World!')
    'Return `n` empty rows of a dataframe'
    df = pd.DataFrame(index=range(n))
    return [df.iloc[i] for i in range(n)]

def display_df(df):
    if False:
        while True:
            i = 10
    'Display `df` in a notebook or defaults to print'
    try:
        from IPython.display import display, HTML
    except:
        return print(df)
    display(HTML(df.to_html()))

def get_first(c):
    if False:
        i = 10
        return i + 15
    'Get the first element of c, even if c is a dataframe'
    return getattr(c, 'iloc', c)[0]

def one_param(m):
    if False:
        for i in range(10):
            print('nop')
    'First parameter in `m`'
    return first(m.parameters())

def item_find(x, idx=0):
    if False:
        i = 10
        return i + 15
    'Recursively takes the `idx`-th element of `x`'
    if is_listy(x):
        return item_find(x[idx])
    if isinstance(x, dict):
        key = list(x.keys())[idx] if isinstance(idx, int) else idx
        return item_find(x[key])
    return x

def find_device(b):
    if False:
        while True:
            i = 10
    'Recursively search the device of `b`.'
    return item_find(b).device

def find_bs(b):
    if False:
        print('Hello World!')
    'Recursively search the batch size of `b`.'
    res = item_find(b)
    if not hasattr(res, 'shape'):
        return len(b)
    return res.shape[0]

def np_func(f):
    if False:
        for i in range(10):
            print('nop')
    'Convert a function taking and returning numpy arrays to one taking and returning tensors'

    def _inner(*args, **kwargs):
        if False:
            print('Hello World!')
        nargs = [to_np(arg) if isinstance(arg, Tensor) else arg for arg in args]
        return tensor(f(*nargs, **kwargs))
    functools.update_wrapper(_inner, f)
    return _inner

class Module(nn.Module, metaclass=PrePostInitMeta):
    """Same as `nn.Module`, but no need for subclasses to call `super().__init__`"""

    def __pre_init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass
from torch.nn.parallel import DistributedDataParallel

def get_model(model):
    if False:
        return 10
    'Return the model maybe wrapped inside `model`.'
    return model.module if isinstance(model, (DistributedDataParallel, nn.DataParallel)) else model

def one_hot(x, c):
    if False:
        while True:
            i = 10
    'One-hot encode `x` with `c` classes.'
    res = torch.zeros(c, dtype=torch.uint8)
    if isinstance(x, Tensor) and x.numel() > 0:
        res[x] = 1.0
    else:
        res[list(L(x, use_list=None))] = 1.0
    return res

def one_hot_decode(x, vocab=None):
    if False:
        i = 10
        return i + 15
    return L((vocab[i] if vocab else i for (i, x_) in enumerate(x) if x_ == 1))

def params(m):
    if False:
        return 10
    'Return all parameters of `m`'
    return [p for p in m.parameters()]

def trainable_params(m):
    if False:
        for i in range(10):
            print('nop')
    'Return all trainable parameters of `m`'
    return [p for p in m.parameters() if p.requires_grad]
norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm)

def norm_bias_params(m, with_bias=True):
    if False:
        print('Hello World!')
    'Return all bias and BatchNorm parameters'
    if isinstance(m, norm_types):
        return L(m.parameters())
    res = L(m.children()).map(norm_bias_params, with_bias=with_bias).concat()
    if with_bias and getattr(m, 'bias', None) is not None:
        res.append(m.bias)
    return res

def batch_to_samples(b, max_n=10):
    if False:
        return 10
    "'Transposes' a batch to (at most `max_n`) samples"
    if isinstance(b, Tensor):
        return retain_types(list(b[:max_n]), [b])
    else:
        res = L(b).map(partial(batch_to_samples, max_n=max_n))
        return retain_types(res.zip(), [b])

@patch
def interp_1d(x: Tensor, xp, fp):
    if False:
        return 10
    'Same as `np.interp`'
    slopes = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    incx = fp[:-1] - slopes * xp[:-1]
    locs = (x[:, None] >= xp[None, :]).long().sum(1) - 1
    locs = locs.clamp(0, len(slopes) - 1)
    return slopes[locs] * x + incx[locs]

@patch
def pca(x: Tensor, k=2):
    if False:
        return 10
    'Compute PCA of `x` with `k` dimensions.'
    x = x - torch.mean(x, 0)
    (U, S, V) = torch.svd(x.t())
    return torch.mm(x, U[:, :k])

def logit(x):
    if False:
        for i in range(10):
            print('nop')
    'Logit of `x`, clamped to avoid inf.'
    x = x.clamp(1e-07, 1 - 1e-07)
    return -(1 / x - 1).log()

def num_distrib():
    if False:
        print('Hello World!')
    'Return the number of processes in distributed training (if applicable).'
    return int(os.environ.get('WORLD_SIZE', 0))

def rank_distrib():
    if False:
        for i in range(10):
            print('nop')
    'Return the distributed rank of this process (if applicable).'
    return int(os.environ.get('RANK', 0))

def distrib_barrier():
    if False:
        print('Hello World!')
    'Place a synchronization barrier in distributed training'
    if num_distrib() > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
try:
    import tables
except:
    pass

def _comp_filter(lib='lz4', lvl=3):
    if False:
        print('Hello World!')
    return tables.Filters(complib=f'blosc:{lib}', complevel=lvl)

@patch
def save_array(p: Path, o, complib='lz4', lvl=3):
    if False:
        for i in range(10):
            print('nop')
    'Save numpy array to a compressed `pytables` file, using compression level `lvl`'
    if isinstance(o, Tensor):
        o = to_np(o)
    with tables.open_file(p, mode='w', filters=_comp_filter(lib=complib, lvl=lvl)) as f:
        f.create_carray('/', 'data', obj=o)

@patch
def load_array(p: Path):
    if False:
        i = 10
        return i + 15
    'Save numpy array to a `pytables` file'
    with tables.open_file(p, 'r') as f:
        return f.root.data.read()

def base_doc(elt):
    if False:
        for i in range(10):
            print('nop')
    'Print a base documentation of `elt`'
    name = getattr(elt, '__qualname__', getattr(elt, '__name__', ''))
    print(f'{name}{inspect.signature(elt)}\n{inspect.getdoc(elt)}\n')
    print('To get a prettier result with hyperlinks to source code and documentation, install nbdev: pip install nbdev')

def doc(elt):
    if False:
        i = 10
        return i + 15
    'Try to use doc form nbdev and fall back to `base_doc`'
    try:
        from nbdev.showdoc import doc
        doc(elt)
    except:
        base_doc(elt)

def nested_reorder(t, idxs):
    if False:
        i = 10
        return i + 15
    'Reorder all tensors in `t` using `idxs`'
    if isinstance(t, (Tensor, L)):
        return t[idxs]
    elif is_listy(t):
        return type(t)((nested_reorder(t_, idxs) for t_ in t))
    if t is None:
        return t
    raise TypeError(f'Expected tensor, tuple, list or L but got {type(t)}')

def flatten_check(inp, targ):
    if False:
        return 10
    'Check that `inp` and `targ` have the same number of elements and flatten them.'
    (inp, targ) = (TensorBase(inp.contiguous()).view(-1), TensorBase(targ.contiguous()).view(-1))
    test_eq(len(inp), len(targ))
    return (inp, targ)

def make_cross_image(bw=True):
    if False:
        while True:
            i = 10
    'Create a tensor containing a cross image, either `bw` (True) or color'
    if bw:
        im = torch.zeros(5, 5)
        im[2, :] = 1.0
        im[:, 2] = 1.0
    else:
        im = torch.zeros(3, 5, 5)
        im[0, 2, :] = 1.0
        im[1, :, 2] = 1.0
    return im

def show_image_batch(b, show=show_titled_image, items=9, cols=3, figsize=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Display batch `b` in a grid of size `items` with `cols` width'
    if items < cols:
        cols = items
    rows = (items + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 3, rows * 3)
    (fig, axs) = plt.subplots(rows, cols, figsize=figsize)
    for (*o, ax) in zip(*to_cpu(b), axs.flatten()):
        show(o, ax=ax, **kwargs)

def requires_grad(m):
    if False:
        print('Hello World!')
    'Check if the first parameter of `m` requires grad or not'
    ps = list(m.parameters())
    return ps[0].requires_grad if len(ps) > 0 else False

def init_default(m, func=nn.init.kaiming_normal_):
    if False:
        print('Hello World!')
    'Initialize `m` weights with `func` and set `bias` to 0.'
    if func:
        if hasattr(m, 'weight'):
            func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    return m

def cond_init(m, func):
    if False:
        for i in range(10):
            print('nop')
    "Apply `init_default` to `m` unless it's a batchnorm module"
    if not isinstance(m, norm_types) and requires_grad(m):
        init_default(m, func)

def apply_leaf(m, f):
    if False:
        while True:
            i = 10
    'Apply `f` to children of `m`.'
    c = m.children()
    if isinstance(m, nn.Module):
        f(m)
    for l in c:
        apply_leaf(l, f)

def apply_init(m, func=nn.init.kaiming_normal_):
    if False:
        print('Hello World!')
    'Initialize all non-batchnorm layers of `m` with `func`.'
    apply_leaf(m, partial(cond_init, func=func))

def script_use_ctx(f):
    if False:
        print('Hello World!')
    'Decorator: create jit script and pass everything in `ctx.saved_variables to `f`, after `*args`'
    sf = torch.jit.script(f)

    def _f(ctx, *args, **kwargs):
        if False:
            return 10
        return sf(*args, *ctx.saved_variables, **kwargs)
    return update_wrapper(_f, f)

def script_save_ctx(static, *argidx):
    if False:
        i = 10
        return i + 15
    'Decorator: create jit script and save args with indices `argidx` using `ctx.save_for_backward`'

    def _dec(f):
        if False:
            return 10
        sf = torch.jit.script(f)

        def _f(ctx, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if argidx:
                save = [args[o] for o in argidx]
                ctx.save_for_backward(*save)
            if not argidx:
                args = [ctx] + args
            return sf(*args, **kwargs)
        if static:
            _f = staticmethod(_f)
        return update_wrapper(_f, f)
    return _dec

def script_fwd(*argidx):
    if False:
        return 10
    'Decorator: create static jit script and save args with indices `argidx` using `ctx.save_for_backward`'
    return script_save_ctx(True, *argidx)

def script_bwd(f):
    if False:
        i = 10
        return i + 15
    'Decorator: create static jit script and pass everything in `ctx.saved_variables to `f`, after `*args`'
    return staticmethod(script_use_ctx(f))

def grad_module(cls):
    if False:
        while True:
            i = 10
    'Decorator: convert `cls` into an autograd function'

    class _c(nn.Module):

        def forward(self, *args, **kwargs):
            if False:
                return 10
            return cls.apply(*args, **kwargs)
    return _c

def ismin_torch(min_version):
    if False:
        print('Hello World!')
    'Check if `torch.__version__` >= `min_version` using packaging.version'
    return _torch_version >= parse(min_version)

def notmax_torch(max_version):
    if False:
        for i in range(10):
            print('nop')
    'Check if `torch.__version__` < `max_version` using packaging.version'
    return _torch_version < parse(max_version)
if ismin_torch('1.13') and notmax_torch('1.14'):
    from torch.overrides import has_torch_function_unary, handle_torch_function

    @patch
    def __format__(self: Tensor, format_spec):
        if False:
            i = 10
            return i + 15
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
        if self.dim() == 0 and (not self.is_meta) and issubclass(type(self), Tensor):
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)