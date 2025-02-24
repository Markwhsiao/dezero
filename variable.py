#/* ===============================================
#     Copyright(c) All rights reserved
#     Author: markxiao@tencent.com
#     Time: 20250219-09:50
#o==================================================*/

import numpy as np
import weakref
import contextlib

class Config:
  enable_backprop = True

#这里shape与x.shape满足np.broadcast的规则：1.从右往左，遇1扩展；2. 维度少的部分补齐
def utils_sum_to(x, shape):
  """Sum elements along axes to output an array of a given shape.

  Args:
      x (ndarray): Input array.
      shape:

  Returns:
      ndarray: Output array of the shape.
  """
  ndim = len(shape)
  lead = x.ndim - ndim
  lead_axis = tuple(range(lead))

  axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
  y = x.sum(lead_axis + axis, keepdims=True)
  if lead > 0:
      y = y.squeeze(lead_axis)
  return y

def utils_reshape_sum_backward(y, shape, axis, keepdims):
  k = 0
  if keepdims:
    output_shape = y.shape
  else:
    output_shape = []
    for i,sx in enumerate(shape):
      if axis is None or i == axis[k]:
        output_shape.append(1)
      else:
        output_shape.append(sx)
      k = k + 1
  return y.reshape(output_shape)

@contextlib.contextmanager
def using_config(name, value):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)

def no_grad():
  return using_config('enable_backprop', False)

def as_array(x):
  if np.isscalar(x):
    return np.array(x)
  return x

def as_variable(x):
  if isinstance(x, Variable):
    return x
  return Variable(x)

class Variable:
  __array_priority__ = 200
  def __init__(self, data, name=None):
    if data is not None:
        if not isinstance(data, np.ndarray):
           raise TypeError('{} is not supported'.format(type(data)))

    self.data = data
    self.name = name
    self.grad = None
    self.creator = None
    self.generation = 0

  @property
  def shape(self):
    return self.data.shape
  @property
  def ndim(self):
    return self.data.ndim
  @property
  def size(self):
    return self.data.size
  @property
  def dtype(self):
    return self.data.dtype

  def __len__(self):
    return len(self.data)

  def __repr__(self):
    if self.data is None:
      return 'variable(None)'
    p = str(self.data).replace('\n', '\n' + ' '*9)
    return 'variable(' + p + ')'

  def __mul__(self, other):
    return mul(self, as_array(other))
  def __add__(self, other):
    return add(self, as_array(other))

  def __rmul__(self, other):
    return mul(self, as_array(other))
  def __radd__(self, other):
    return add(self, as_array(other))
  def __neg__(self):
    return neg(self)
  def __sub__(self, other):
    return sub(self, as_array(other))
  def __rsub__(self, other):
    return sub(as_array(other), self)
  def __truediv__(self, other):
    return div(self, as_array(other))
  def __rtruediv__(self, other):
    return div(as_array(other), self)
  def __pow__(self, other):
    return pow(self, as_array(other))

  def reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
      shape = shape[0]
    return reshape(self, shape)
  @property
  def T(self):
    return transpose(self)

  def sum(self, axis=None, keepdims=False):
    return sum(self, axis, keepdims)

  def set_creator(self, func):
    self.creator = func
    self.generation = func.generation + 1

  def cleargrad(self):
    self.grad = None

  def backward(self, retain_grad=False, create_graph=False):
    if self.grad is None:
      self.grad = Variable(np.ones_like(self.data))

    funcs = []
    seen_set = set()

    def add_func(f):
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        funcs.sort(key=lambda x: x.generation)
    add_func(self.creator)

    while funcs:
      f = funcs.pop()
      ygs = [output().grad for output in f.outputs]
      with using_config('enable_backprop', create_graph):
        xgs = f.backward(*ygs)
        if not isinstance(xgs, tuple):
            xgs = (xgs,)
        for x, xg in zip(f.inputs, xgs):
          if x.grad is None:
            x.grad = xg
          else:
            x.grad = x.grad + xg
          if x.creator is not None:
            add_func(x.creator)
      if not retain_grad:
        for y in f.outputs:
          y().grad = None


class Function:
  def __call__(self, *inputs):
    inputs = [as_variable(x) for x in  inputs]
    xs = [input.data for input in inputs]
    ys = self.forward(*xs)
    if not isinstance(ys, tuple):
        ys = (ys,)
    outputs = [Variable(as_array(y)) for y in ys]

    if Config.enable_backprop:
      self.generation = max([input.generation for input in inputs])
      for output in outputs:
        output.set_creator(self)
      self.inputs = inputs
    else:
      self.inputs = None
    self.outputs = [weakref.ref(output) for output in outputs]
    return outputs if len(outputs) > 1 else outputs[0]
  def forward(self, *xs):
    raise NotImplementedError()
  def backward(self, *gy):
    raise NotImplementedError()

class Add(Function):
  def forward(self, x0, x1):
    self.x0_shape,self.x1_shape = x0.shape, x1.shape
    return x0 + x1
  def backward(self, gy):
    gx0, gx1 = gy, gy
    if self.x0_shape != self.x1_shape:
      gx0 = sum_to(gx0, self.x0_shape)
      gx1 = sum_to(gx1, self.x1_shape)
    return gx0, gx1

class Sum(Function):
  def __init__(self, axis, keepdims):
    self.axis = axis
    self.keepdims = keepdims
  def forward(self, x):
    self.x_shape = x.shape
    return np.sum(x, self.axis, keepdims=self.keepdims)
  def backward(self, gy):
    gy = utils_reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
    return broadcast_to(gy, self.x_shape)

class Tanh(Function):
  def forward(self, x):
    y = np.tanh(x)
    return y
  def backward(self, gy):
    y = self.outputs[0]
    gy = gy * (1 - y*y)
    return gx

class BroadCastTo(Function):
  def __init__(self, shape):
    self.shape = shape
  def forward(self, x):
    self.x_shape = x.shape
    return np.broadcast_to(x, self.shape)
  def backward(self, gy):
    return sum_to(gy, self.x_shape)

class SumTo(Function):
  def __init__(self, shape):
    self.shape = shape
  def forward(self, x):
    self.x_shape = x.shape
    y = utils_sum_to(x, self.shape)
    return y
  def backward(self, gy):
    return broadcast_to(gy, self.x_shape)

class Sin(Function):
  def forward(self, x):
    return np.sin(x)
  def backward(self, gy):
    x, = self.inputs
    gx = gy*cos(x)
    return gx

class Cos(Function):
  def forward(self, x):
    return np.cos(x)
  def backward(self, gy):
    x, = self.inputs
    return gy*(-sin(x))

class Mul(Function):
  def forward(self, x0, x1):
    return x0 * x1
  def backward(self, gy):
    x0, x1 = self.inputs
    return gy*x1, gy*x0

class Sub(Function):
  def forward(self, x0, x1):
    return x0 - x1
  def backward(self, gy):
    return gy,-gy

class Div(Function):
  def forward(self, x0, x1):
    return x0/x1
  def backward(self, gy):
    x0,x1 = self.inputs
    return gy/x1, gy*(-x0/x1**2)

class Pow(Function):
  def __init__(self, c):
    self.c = c
  def forward(self, x):
    return x**self.c
  def backward(self, gy):
    x = self.inputs[0]
    return  gy * self.c * x ** (self.c-1)

class Neg(Function):
  def forward(self, x):
    return -x
  def backward(self, gy):
    return -gy

class Reshape(Function):
  def __init__(self, shape):
    self.shape = shape
  def forward(self, x):
    self.x_shape = x.shape
    return np.reshape(x, self.shape)
  def backward(self, gy):
    return reshape(gy, self.x_shape)

class Transpose(Function):
  def forward(self, x):
    return np.transpose(x)
  def backward(self, gy):
    return transpose(gy)

class MeanSquaredError(Function):
  def forward(self, x0, x1):
    diff = x0 - x1
    y = np.sum(diff**2)/len(diff)
    return y
  def backward(self, gy):
    x0, x1 = self.inputs
    diff = x0 - x1
    gx0 = gy*diff*(2./len(diff))
    gx1 = -gx0
    return gx0, gx1

class MatMul(Function):
  def forward(self, x0, x1):
    if x0.shape[-1] != x1.shape[0]:
      raise NotImplementedError("shape not match")
    return np.dot(x0, x1)
  def backward(self, gy):
    x0,x1 = self.inputs
    gx0 = matmul(gy, x1.T)
    gx1 = matmul(x0.T, gy)
    return gx0, gx1

def add(x0, x1):
  return Add()(x0, x1)

def mul(x0, x1):
  return Mul()(x0, x1)

def sub(x0, x1):
  return Sub()(x0, x1)
def div(x0, x1):
  return Div()(x0, x1)
def pow(x, c):
  return Pow(c)(x)
def neg(x):
  return Neg()(x)
def sin(x):
  return Sin()(x)
def cos(x):
  return Cos()(x)
def tanh(x):
  return Tanh()(x)

def reshape(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return Reshape(shape)(x)

def transpose(x):
  return Transpose()(x)

def sum(x, axis=None, keepdims=False):
  return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return BroadCastTo(shape)(x)

def sum_to(x, shape):
  return SumTo(shape)(x)

def matmul(x0, x1):
  return MatMul()(x0, x1)

def mean_squared_error(x0, x1):
  return MeanSquaredError()(x0, x1)

def numerical_diff(f, x, eps=1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data)/(2*eps)

def numerical_diff(f, xa, xb, eps=1e-4):
  xa0 = Variable(as_array(xa.data - eps))
  xa1 = Variable(as_array(xa.data + eps))
  y0 = f(xa0, xb)
  y1 = f(xa1, xb)

  xa_g = (y1.data - y0.data)/(2*eps)

  xb0 = Variable(as_array(xb.data - eps))
  xb1 = Variable(as_array(xb.data + eps))
  y0 = f(xa, xb0)
  y1 = f(xa, xb1)
  xb_g = (y1.data - y0.data)/(2*eps)
  return (xa_g, xb_g)


#######





