from collections import namedtuple
import torch
SymmArrowhead = namedtuple('SymmArrowhead', ['top', 'bottom_diag'])
TriuArrowhead = namedtuple('TriuArrowhead', ['top', 'bottom_diag'])

def sqrt(x):
    if False:
        i = 10
        return i + 15
    '\n    EXPERIMENTAL Computes the upper triangular square root of an\n    symmetric arrowhead matrix.\n\n    :param SymmArrowhead x: an symmetric arrowhead matrix\n    :return: the square root of `x`\n    :rtype: TriuArrowhead\n    '
    assert isinstance(x, SymmArrowhead)
    head_size = x.top.size(0)
    if head_size == 0:
        return TriuArrowhead(x.top, x.bottom_diag.sqrt())
    (A, B) = (x.top[:, :head_size], x.top[:, head_size:])
    Dsqrt = x.bottom_diag.sqrt()
    num_attempts = 6
    for i in range(num_attempts):
        B_Dsqrt = B / Dsqrt.unsqueeze(-2)
        schur_complement = A - B_Dsqrt.matmul(B_Dsqrt.t())
        try:
            top_left = torch.flip(torch.linalg.cholesky(torch.flip(schur_complement, (-2, -1))), (-2, -1))
            break
        except RuntimeError:
            B = B / 2
            continue
        raise RuntimeError('Singular schur complement in computing Cholesky of the input arrowhead matrix')
    top_right = B_Dsqrt
    top = torch.cat([top_left, top_right], -1)
    bottom_diag = Dsqrt
    return TriuArrowhead(top, bottom_diag)

def triu_inverse(x):
    if False:
        return 10
    '\n    EXPERIMENTAL Computes the inverse of an upper-triangular arrowhead matrix.\n\n    :param TriuArrowhead x: an upper-triangular arrowhead matrix.\n    :return: the inverse of `x`\n    :rtype: TriuArrowhead\n    '
    assert isinstance(x, TriuArrowhead)
    head_size = x.top.size(0)
    if head_size == 0:
        return TriuArrowhead(x.top, x.bottom_diag.reciprocal())
    (A, B) = (x.top[:, :head_size], x.top[:, head_size:])
    B_Dinv = B / x.bottom_diag.unsqueeze(-2)
    identity = torch.eye(head_size, dtype=A.dtype, device=A.device)
    top_left = torch.linalg.solve_triangular(A, identity, upper=True)
    top_right = -top_left.matmul(B_Dinv)
    top = torch.cat([top_left, top_right], -1)
    bottom_diag = x.bottom_diag.reciprocal()
    return TriuArrowhead(top, bottom_diag)

def triu_matvecmul(x, y, transpose=False):
    if False:
        print('Hello World!')
    '\n    EXPERIMENTAL Computes matrix-vector product of an upper-triangular\n    arrowhead matrix `x` and a vector `y`.\n\n    :param TriuArrowhead x: an upper-triangular arrowhead matrix.\n    :param torch.Tensor y: a 1D tensor\n    :return: matrix-vector product of `x` and `y`\n    :rtype: TriuArrowhead\n    '
    assert isinstance(x, TriuArrowhead)
    head_size = x.top.size(0)
    if transpose:
        z = x.top.transpose(-2, -1).matmul(y[:head_size])
        top = z[:head_size]
        bottom = z[head_size:] + x.bottom_diag * y[head_size:]
    else:
        top = x.top.matmul(y)
        bottom = x.bottom_diag * y[head_size:]
    return torch.cat([top, bottom], 0)

def triu_gram(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    EXPERIMENTAL Computes the gram matrix `x.T @ x` from an upper-triangular\n    arrowhead matrix `x`.\n\n    :param TriuArrowhead x: an upper-triangular arrowhead matrix.\n    :return: the square of `x`\n    :rtype: TriuArrowhead\n    '
    assert isinstance(x, TriuArrowhead)
    head_size = x.top.size(0)
    if head_size == 0:
        return x.bottom_diag.pow(2)
    (A, B) = (x.top[:, :head_size], x.top[:, head_size:])
    top = A.t().matmul(x.top)
    bottom_left = top[:, head_size:].t()
    bottom_right = B.t().matmul(B) + x.bottom_diag.pow(2).diag()
    return torch.cat([top, torch.cat([bottom_left, bottom_right], -1)], 0)