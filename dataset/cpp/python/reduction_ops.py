import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.torch.func_wrapper import (
    to_ivy_arrays_and_back,
    numpy_to_torch_style_args,
)
from collections import namedtuple
import ivy.functional.frontends.torch as torch_frontend


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def all(input, dim=None, keepdim=False, *, out=None):
    input_dtype = ivy.as_ivy_dtype(input.dtype)
    ret = ivy.all(input, axis=dim, keepdims=keepdim, out=out)
    if ivy.is_uint_dtype(input_dtype):
        ret = ivy.astype(ret, input_dtype, out=out)
    return ret


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def amax(input, dim=None, keepdim=False, *, out=None):
    return ivy.max(input, axis=dim, keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def amin(input, dim=None, keepdim=False, *, out=None):
    return ivy.min(input, axis=dim, keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.1.0 and below": ("float16", "bfloat16")}, "torch")
def aminmax(input, *, dim=None, keepdim=False, out=None):
    minmax_tuple = namedtuple("minmax", ["min", "max"])
    return minmax_tuple(
        ivy.min(input, axis=dim, keepdims=keepdim, out=out),
        ivy.max(input, axis=dim, keepdims=keepdim, out=out),
    )


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def any(input, dim=None, keepdim=False, *, out=None):
    input_dtype = ivy.as_ivy_dtype(input.dtype)
    ret = ivy.any(input, axis=dim, keepdims=keepdim, out=out)
    if ivy.is_uint_dtype(input_dtype):
        ret = ivy.astype(ret, input_dtype, out=out)
    return ret


@with_unsupported_dtypes({"2.1.0 and below": ("complex", "bool")}, "torch")
@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def argmax(input, dim=None, keepdim=False):
    return ivy.argmax(input, axis=dim, keepdims=keepdim)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def argmin(input, dim=None, keepdim=False):
    return ivy.argmin(input, axis=dim, keepdims=keepdim).astype(ivy.int64)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.1.0 and below": ("uint8", "int8")},
    "torch",
)
def count_nonzero(input, dim=None):
    return ivy.count_nonzero(input, axis=dim).astype(ivy.int64)


@to_ivy_arrays_and_back
def dist(input, other, p=2):
    return ivy.vector_norm(ivy.subtract(input, other), ord=p)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def logsumexp(input, dim, keepdim=False, *, out=None):
    c = ivy.max(input, axis=dim, keepdims=True)
    if ivy.get_num_dims(c) > 0:
        c = ivy.where(ivy.isinf(c), ivy.zeros_like(c), c)
    elif not ivy.isinf(c):
        c = 0
    exponential = ivy.exp(input - c)
    sum = ivy.sum(exponential, axis=dim, keepdims=keepdim)
    ret = ivy.log(sum)
    if not keepdim:
        c = ivy.squeeze(c, axis=dim)
    ret = ivy.add(ret, c, out=out)
    return ret


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def max(*input, dim=None, keepdim=False, out=None):
    if len(input) == 1:
        input = input[0]
    elif len(input) == 2:
        return torch_frontend.maximum(*input)
    if dim is None:
        return ivy.max(input, axis=dim, keepdims=keepdim, out=out)
    elif out is not None:
        ivy.max(input, axis=dim, keepdims=keepdim, out=out[0])
        ivy.argmax(input, axis=dim, keepdims=keepdim, out=out[1])
        return out
    else:
        max_tuple = namedtuple("max", ["values", "indices"])
        return max_tuple(
            ivy.max(input, axis=dim, keepdims=keepdim),
            ivy.argmax(input, axis=dim, keepdims=keepdim),
        )


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is not None:
        input = input.astype(dtype)
        if out is not None:
            out = out.astype(dtype)
    return ivy.mean(input, axis=dim, keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def median(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        input = ivy.reshape(input, (-1,))
        sorted_input = ivy.sort(input)
        return sorted_input[(sorted_input.shape[0] - 1) // 2]

    median_tuple = namedtuple("median", ["values", "indices"])

    if input.ndim == 0:
        result = median_tuple(input, ivy.array(0))
    else:
        sorted_indices = ivy.argsort(input, axis=dim)
        median_indices = ivy.gather(
            sorted_indices, (sorted_indices.shape[dim] - 1) // 2, axis=dim
        )
        median_values = ivy.take_along_axis(
            input, ivy.expand_dims(median_indices, axis=dim), dim
        ).squeeze(axis=dim)

        if keepdim:
            median_values = ivy.expand_dims(median_values, axis=dim)
            median_indices = ivy.expand_dims(median_indices, axis=dim)

        result = median_tuple(median_values, median_indices)
    if out is not None:
        ivy.inplace_update(out[0], result.values)
        ivy.inplace_update(out[1], result.indices)
        return out
    return result


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"2.1.0 and below": ("complex64", "complex128")},
    "torch",
)
def min(*input, dim=None, keepdim=False, out=None):
    if len(input) == 1:
        input = input[0]
    elif len(input) == 2:
        return torch_frontend.minimum(*input)
    if dim is None:
        return ivy.min(input, axis=dim, keepdims=keepdim, out=out)
    elif out is not None:
        ivy.min(input, axis=dim, keepdims=keepdim, out=out[0])
        ivy.argmin(input, axis=dim, keepdims=keepdim, out=out[1])
        return out
    else:
        min_tuple = namedtuple("min", ["values", "indices"])
        return min_tuple(
            ivy.min(input, axis=dim, keepdims=keepdim),
            ivy.argmin(input, axis=dim, keepdims=keepdim),
        )


@to_ivy_arrays_and_back
def moveaxis(input, source, destination):
    return ivy.moveaxis(input, source, destination)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def nanmean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    return ivy.nanmean(input, axis=dim, keepdims=keepdim, dtype=dtype, out=out)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def nanmedian(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        flattened_input = ivy.flatten(input)
        sorted_input = ivy.sort(flattened_input)
        nonnan_index = int(sorted_input.shape[0] - ivy.isnan(sorted_input).sum())
        return sorted_input[(nonnan_index - 1) // 2]

    nanmedian_tuple = namedtuple("nanmedian", ["values", "indices"])

    if input.ndim == 0:
        result = nanmedian_tuple(input, ivy.array(0))
    else:
        sorted_indices = ivy.argsort(input, axis=dim)
        nonnan_index = (
            sorted_indices.shape[dim] - ivy.isnan(input).sum(axis=1) - 1
        ) // 2
        nonnan_index = ivy.expand_dims(nonnan_index, axis=1)
        nanmedian_indices = ivy.gather_nd(sorted_indices, nonnan_index, batch_dims=1)
        nanmedian_values = ivy.take_along_axis(
            input, ivy.expand_dims(nanmedian_indices, axis=dim), dim
        ).squeeze(axis=dim)

        if keepdim:
            nanmedian_values = ivy.expand_dims(nanmedian_values, axis=dim)
            nanmedian_indices = ivy.expand_dims(nanmedian_tuple, axis=dim)

        result = nanmedian_tuple(nanmedian_values, nanmedian_indices)
    if out is not None:
        ivy.inplace_update(out[0], result.values)
        ivy.inplace_update(out[1], result.indices)
        return out
    return result


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.1.0 and below": ("float", "int")},
    "torch",
)
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    input = ivy.where(ivy.isnan(input), ivy.zeros_like(input), input)
    return ivy.sum(input, axis=dim, dtype=dtype, keepdims=keepdim, out=None)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {"2.1.0 and below": ("float", "complex")},
    "torch",
)
def norm(input, p="fro", dim=None, keepdim=False, out=None, dtype=None):
    if dtype is None or not ivy.is_float_dtype(dtype):
        dtype = "float64" if "128" in str(dtype) else "float32"
    if (
        p == "fro" and (dim is None or isinstance(dim, int) or len(dim) <= 2)
    ) or p is None:
        p = 2
    if isinstance(p, str):
        if dim is None:
            dim = tuple(range(input.dim()))
        return ivy.matrix_norm(
            input, ord=p, axis=dim, keepdims=keepdim, out=out
        ).astype(dtype)
    else:
        return ivy.vector_norm(
            input, ord=p, axis=dim, keepdims=keepdim, dtype=dtype, out=out
        )


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def prod(input, dim=None, keepdim=False, *, dtype=None):
    if not dtype:
        if "int" in input.dtype:
            dtype = ivy.int64
    return ivy.prod(input, axis=dim, dtype=dtype, keepdims=keepdim)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.1.0 and below": ("float16", "bfloat16")}, "torch")
def quantile(input, q, dim=None, keepdim=False, *, interpolation="linear", out=None):
    return ivy.quantile(
        input, q, axis=dim, keepdims=keepdim, interpolation=interpolation, out=out
    )


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.1.0 and below": ("float16",)}, "torch")
def std(input, dim=None, unbiased=True, keepdim=False, *, out=None):
    return ivy.std(input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes({"2.1.0 and below": ("bfloat16",)}, "torch")
def std_mean(input, dim, unbiased, keepdim=False, *, out=None):
    temp_std = ivy.std(
        input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out
    )
    temp_mean = ivy.mean(input, axis=dim, keepdims=keepdim, out=out)
    return temp_std, temp_mean


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
def sum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    return ivy.sum(input, axis=dim, dtype=dtype, keepdims=keepdim, out=out)


@to_ivy_arrays_and_back
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if dim is not None:
        sorted = True
    results = ivy.unique_all(input, axis=dim, by_value=sorted)

    fields = ["output"]
    if return_inverse:
        fields.append("inverse_indices")
    if return_counts:
        fields.append("counts")

    Results = namedtuple("Results", fields)

    values = [results.values]
    if return_inverse:
        inverse_indices = results.inverse_indices

        if dim is None:
            inverse_indices = inverse_indices.reshape(input.shape)

        values.append(inverse_indices)
    if return_counts:
        values.append(results.counts)

    return Results(*values)


@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
            "float16",
            "complex",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def unique_consecutive(input, return_inverse, return_counts, dim):
    output, inverse_indices, counts = ivy.unique_consecutive(input, axis=dim)
    ret = (output,)
    if return_inverse:
        ret += (inverse_indices,)
    if return_counts:
        ret += (counts,)
    return ret


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def var(input, dim, unbiased, keepdim=False, *, out=None):
    return ivy.var(input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out)


@numpy_to_torch_style_args
@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "2.1.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def var_mean(input, dim, unbiased, keepdim=False, *, out=None):
    temp_var = ivy.var(
        input, axis=dim, correction=int(unbiased), keepdims=keepdim, out=out
    )
    temp_mean = ivy.mean(input, axis=dim, keepdims=keepdim, out=out)
    return (temp_var, temp_mean)
