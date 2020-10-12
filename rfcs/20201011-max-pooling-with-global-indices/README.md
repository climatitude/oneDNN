# RFC: Max pooling to support global indices as one of the outputs

## Motivation

There are some
[neural networks for semantic segmentation](Learning_Deconvolution_Network_for_Semantic_Segmentation.pdf)
which implement the network as two chains: half is a convolutional chain with
pooling operations in between, another half is a deconvolutional chain with
**unpooling** operations in between. An unpooling operation does the opposite
to pooling, it upsamples the image based on the indices obtained from "paired"
pooling and the output from a previous operation. Currently, oneDNN does not
give an opportunity to an end user to obtain global indices of local max values.
The request from PyTorch framework is to add such ability to the max pooling
operation. The similar request came from ONNX Runtime Team to support their
feature of running forward and backward pooling in different sub-graphs which
may execute on different vendor hardware and/or software.

We are not requested to implement an unpooling primitive so far as it is of low
priority and used in not very popular models (it's not in PyTorch's Model Zoo).
But even if we were to implement unpooling, we can't rely on workspace memory
from pooling since in general case the library expect that pooling to be called
using only oneDNN implementation but not native framework's or user's code. But
even with the call to oneDNN pooling, using the same workspace would be really
complicated since the memory format of unpooling may easily differ from pooling
one, and parameters of pooling and unpooling do not have to coincide - they may
have different kernel, padding, strides and output sizes.

## Proposal

### Option 1 (recommended) - new algorithm.

Introduce a new max pooling algorithm:
~~~c
/* dnnl_types.h */

typedef enum {
    ...
    /// Average pooling exclude padding
    dnnl_pooling_avg_exclude_padding = 0x3ff,
    /// Max pooling with second output of global indices
    dnnl_pooling_max_with_global_indices = 0x4ff,
    ...
} dnnl_alg_kind_t;
~~~
It will instruct the implementation to dump global indices in a second output
marked as `DNNL_ARG_DST_1`. By introducing a new algorithm we preserve
performance for the case when output indices are not desired, since it will
double the amount of memory to write for a kernel.

### Option 2 - re-use existing algorithm

This option makes `dnnl_pooling_max` algorithm to dump global indices
unconditionally if the second output memory is not empty. The drawback is
sacrificing performance for slightly easier enabling the feature as no need to
differentiate on a user side when to use one algorithm over the other. However,
the library always seem to rely on a primitive descriptor when it comes to
whether the memory should be read or written in `execute()` (e.g. `with_bias()`
method for convolutions), so run-time check is less intuitive from API
perspective at the same time.

## Testing

Seems transparent. Reference benchdnn implementation to dump global indices to
compare with the second output in addition to max pooling results comparison.

EOD.
