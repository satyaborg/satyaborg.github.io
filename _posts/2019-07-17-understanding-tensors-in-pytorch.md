---
layout: post
title: Understanding Tensors in PyTorch
description: In this first post, 
tags: [pytorch, deep-learning]
---


Let's try and deconstruct tensors which are the fundamental data structures behind neural networks.
Starting with what a tensor looks like in different dimensions,

![tensors]({{site.baseurl}}/assets/images/documentation/post-1/tensors.jpg)

For smaller dimensions, we can think in geometric analogies as shown above, like for e.g.

- A 0-d Tensor or Scalar would be a point in space.
- A 1-d Tensor or Vector would be a line in that space.
- A 2-d Tensor or Matrix would be a square (or rectangle), and
- A 3-d Tensor would be a cube (or cuboid)

Similarly we can go further,

- A 4-d Tensor would be a vector of cubes.
- A 5-d Tensor would be a matrix of cubes.
- A 6-d Tensor would be a cube made up of cubes.
- A 7-d Tensor would be a vector of cubes made up of cubes!

<!-- 8-d matrix of cubes made up of cubes
9-d cube made up of cubes made up of cubes
10-d vector of 9-d tensors -->

You get the idea.

As you can see, tensors are nothing but a generalisation of things we know and love : scalars, vectors and matrices. Specifically it is a *N-dimensional* data structure (or container) where $$N$$ is the number of dimensions. For people familiar with arrays, its just a multi-dimensional array.

![tensor image]({{site.baseurl}}/assets/images/documentation/post-1/scalar-vector-matrix-tensor.jpg)

As the world around us is three dimensional, we have a hard time visualising tensors, especially as the dimensions go beyond our own. Like imagine a 10-dimensional tensor in your head. What does it look like? Use the analogy of cubes and let me know in the comments below!

But why are they important for deep learning, you ask? Well they are central to linear algebra and also the fundamental data structures of a neural network. Its all N-dimensional tensors under the hood, following some predefined rules (dot products and applying some non-linearity) to compute some output given certain inputs.

Let's see how we can create tensors in PyTorch.

## Import Libraries

```python
import torch # the core module of PyTorch
import seaborn as sns # For visualisation
torch.manual_seed(42); # set a seed for reproducibility
```

A good practice before starting any experiment is to set a fixed seed.
This makes things _deterministic_, meaning every time we run the experiment certain non-determinisitic (i.e. probabilistic) processes won't result in different values. And yes, 42 is a reference to _The Hitchhiker's Guide to the Galaxy_ :)

## Generating Tensors

There are various ways to generate tensors in PyTorch. Let's try a few and also check the resulting tensors' datatypes,


```python
print(torch.tensor(1), '->', torch.tensor(1).dtype)
print(torch.empty(1), '->', torch.empty(1).dtype)
print(torch.FloatTensor(1), '->', torch.FloatTensor(1).dtype)
print(torch.Tensor(1), '->', torch.Tensor(1).dtype)
```
```

    tensor(1) -> torch.int64
    tensor([1.5283e-35]) -> torch.float32
    tensor([1.5283e-35]) -> torch.float32
    tensor([2.4548e-37]) -> torch.float32
```

> Note : Here `torch.tensor()` with a single number actually returns a scalar value
with int64 as the data type, whereas the others return a tensor. In general, `torch.tensor()`
accepts a list of numbers and the dimensions/size of the tensor is determined from it.

Lets try passing a list of lists to `torch.tensor()` as follows,

```python
z = torch.tensor([[1,2,3],[4,5,6],[6,7,8]]);
z.size()
```

```
    torch.Size([3, 3])
```
So its a 2-d tensor of 3x3 size. We can peek into its contents just to be sure,

```python
z
```
```
    tensor([[1, 2, 3],
            [4, 5, 6],
            [6, 7, 8]])
```

So we can pass lists like,

- $$[..]$$ which is a list (or an array). It would be a rank 1 or a 1-d tensor.
- $$[[..],[..],[..]]$$ which is a list of lists. It would be a rank 2 or a 2-d tensor.
- $$[[[..],[..]],[[..],[..]]]$$ which is a list of lists containing lists. It would be a rank 3 or a 3-d tensor.
and so on.

The notion of **Rank** is very important when dealing with tensors. In a rigorous mathematical setting, it would have a slightly different meaning, but here I am using the term interchangeably with dimensions.

`torch.empty()` can be used for randomly initialising a tensor with a given size,

```python
torch.empty([2,3,4])
```
```




    tensor([[[1.3515e-36, 0.0000e+00, 6.8664e-44, 7.9874e-44],
             [6.3058e-44, 6.7262e-44, 7.7071e-44, 6.3058e-44],
             [6.8664e-44, 7.1466e-44, 1.1771e-43, 6.8664e-44]],

            [[7.0065e-44, 8.1275e-44, 6.7262e-44, 7.5670e-44],
             [8.1275e-44, 7.1466e-44, 7.0065e-44, 6.4460e-44],
             [6.8664e-44, 7.9874e-44, 7.5670e-44, 7.1466e-44]]])

```

Similarly, `torch.ones()` is for creating a tensor with all the values as ones.

```python
torch.ones([2,3,4])
```

```

    tensor([[[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]],

            [[1., 1., 1., 1.],
             [1., 1., 1., 1.],
             [1., 1., 1., 1.]]])
```

Both `torch.empty()` and `torch.ones()` accept dimensions of the tensor you want to create.

> Note : We can use the GPU variant of tensors by just adding `.cuda()` at the end. This allows for faster computations by leveraging GPU memory.

## Specifying data types and type coercion

One way to force a PyTorch tensor to be of a specific datatype is to use the `dtype` argument as follows,

```python
x = torch.tensor(1, dtype=torch.float32)
print(x.dtype)
y = torch.tensor(1., dtype=torch.int32)
print(y.dtype)
```

```

    torch.float32
    torch.int32
```
> Note : The datatypes are specified with `torch.<something>`.

However, we cannot do this with `torch.Tensor()` since it is actually an alias for `torch.FloatTensor()` and fixes a default datatype of float rather than inferring it from the tensor passed. On the other hand, datatypes can be explicitly stated with `torch.tensor()` (as we saw above).

Now let's check `torch.tensor()`'s default datatype,


```python
print(torch.Tensor([1,2]).dtype)
```
```

    torch.float32

```

Another cool thing is that if we just use a *dot* at the end of a number,
the tensor becomes of type float,


```python
print(torch.tensor([1.,2]).dtype)
print(torch.tensor([1, 2]).dtype)
```
```

    torch.float32
    torch.int64
```

Next up, we can check the sizes or shapes of tensors with
`.size()`,


```python
torch.tensor([1,1]).size()
```
```

    torch.Size([2])

```

> Note : Both `.size()` and `.shape` works.

Did I say tensors can only contain number? They can contain boolean values (or strings) as well.
Using `dtype=torch.bool` we can convert a tensors of ones into a tensor of True(s) as follows,

```python
z = torch.ones((16,16,16), dtype=torch.bool)
```

The tensors should only contain elements of homogeneous datatypes.

## Generating tensors containing random numbers

Now let's generate a tensor with `torch.rand()`, which will be composed of numbers randomly drawn from a uniform distribution $$U \backsim [0,1)$$

```python
torch.rand([3,5])
```
```


    tensor([[0.4294, 0.8854, 0.5739, 0.2666, 0.6274],
            [0.2696, 0.4414, 0.2969, 0.8317, 0.1053],
            [0.2695, 0.3588, 0.1994, 0.5472, 0.0062]])

```

If U is a random variable uniformly distributed on $$[0, 1]$$, then $$(r1 - r2) * U + r2$$ is uniformly distributed on $$[r1, r2]$$. We can also use `uniform_` to perform operations inplace (anything with a
underscore at the end means inplace). Let's see them in action,

```python
# Define a few variables
a = 1; b = 2; r1 = -1; r2 = 1

# To sample from a uniform distribution
# Approach 1
print((r1 - r2) * torch.rand(a, b) + r2)

# Approach 2
print(torch.FloatTensor([a, b]).uniform_(r1, r2))
```
```

    tensor([[0.9953, 0.2270]])
    tensor([-0.5995, -0.0875])
```

## Indexing and slicing

Both indexing and slicing of PyTorch's tensors work similar to numpy's ndarrays and python's lists. For e.g.

```python
z = torch.rand([3,4,8])
```

We can get a copy of the entire tensor like so,

```python
z[:,:,:]
```
```


    tensor([[[0.5898, 0.7489, 0.3316, 0.0840, 0.3186, 0.7509, 0.2768, 0.4062],
             [0.4274, 0.6052, 0.3167, 0.0132, 0.9384, 0.7179, 0.9822, 0.8424],
             [0.7407, 0.6645, 0.7467, 0.4408, 0.3952, 0.2945, 0.7976, 0.9999],
             [0.9323, 0.4777, 0.6843, 0.7982, 0.5203, 0.1099, 0.9234, 0.9767]],

            [[0.5355, 0.6715, 0.8545, 0.1427, 0.5750, 0.3447, 0.2765, 0.4843],
             [0.3656, 0.5375, 0.0905, 0.6682, 0.1834, 0.0282, 0.0847, 0.8121],
             [0.5522, 0.7084, 0.9103, 0.8601, 0.5659, 0.1395, 0.5961, 0.4317],
             [0.7865, 0.6097, 0.0239, 0.6577, 0.6302, 0.1751, 0.2286, 0.8689]],

            [[0.3085, 0.6109, 0.7863, 0.0473, 0.8031, 0.4685, 0.4898, 0.8933],
             [0.9218, 0.3830, 0.0900, 0.1459, 0.8806, 0.6364, 0.6556, 0.3507],
             [0.7947, 0.8174, 0.7804, 0.9511, 0.3414, 0.0311, 0.4173, 0.0569],
             [0.7231, 0.4320, 0.8551, 0.9223, 0.3884, 0.5857, 0.5061, 0.5856]]])

```

We can specify indices for each dimension of the tensor to get the specific element as follows,


```python
z[0,0,1]
```
```




    tensor(0.7489)

```

We can also slice the tensors along a specific dimension as follows,

```python
z[1,:,:]
```
```




    tensor([[0.5355, 0.6715, 0.8545, 0.1427, 0.5750, 0.3447, 0.2765, 0.4843],
            [0.3656, 0.5375, 0.0905, 0.6682, 0.1834, 0.0282, 0.0847, 0.8121],
            [0.5522, 0.7084, 0.9103, 0.8601, 0.5659, 0.1395, 0.5961, 0.4317],
            [0.7865, 0.6097, 0.0239, 0.6577, 0.6302, 0.1751, 0.2286, 0.8689]])

```

In case you forgot how slicing worked in python, here's a quick refresher :

- z[start:stop] - Get all items from *start* index all the way upto *stop* index-1
- z[start:] - Get all items from *start* index all the way to the end
- z[:stop] - Get all items from the beginning all the way to the *stop* index-1
- z[:] - Get the copy of the entire list

The only difference here is that we are dealing with multiple dimensions for tensors.

## 0-d Tensor or a Scalar

We can have scalars in PyTorch as follows,

```python
z = torch.tensor([[3]]);
z, z.size()
```
```
    (tensor([[3]]), torch.Size([1, 1]))
```
or

```python
z = torch.tensor(3);
z, z.size()
```
```
    (tensor(3), torch.Size([]))
```

Note that although their sizes are slightly different, however a 2-d tensor of size 1x1 contains just one number. Similarly a 3-d tensor of size 1x1x1 will also contain a single element and so on.

And since its just a single number we can retrieve it by using `.item()`

```python
z.item()
```
```
    3
```
## 1-d Tensor or a Vector

Similarly for vectors,

```python
z = torch.rand(5);
z, z.size()
```
```

    (tensor([0.9008, 0.1170, 0.2945, 0.1563, 0.6122]), torch.Size([5]))
```

or we can manually define as,

```python
z = torch.tensor([1,2,3,4,5])
```


## 2-d Tensor or a Matrix

Now let's generate a 16x16 matrix as follows,

```python
z = torch.rand(16,16);
z.size()
```
```

    torch.Size([16, 16])
```

We can visualise the matrix by plotting it as a heatmap. This allow us to quickly get a sense
of the numbers contained within visually.


```python
ax = sns.heatmap(z, cmap="gray")
```


![Every 16x16 slice]({{site.baseurl}}/assets/images/documentation/post-1/1.png)


The intensity of each block corresponds to the values in the matrix. Doesn't it look like the
pixels of very low resolution grayscale image?

## 3-d Tensor

Now let's create a 3-d tensor of size 16x16x16 as follows,

```python
z = torch.rand(16,16,16);
z.size()
```
```

    torch.Size([16, 16, 16])
```

Now imagine a cube with all the faces having a size of 16. We can visualise each *slice* of this
cube along its depth. For the 1st slice,

```python
ax = sns.heatmap(z[0,:,:], cmap="gray")
```

![Every 16x16 slice]({{site.baseurl}}/assets/images/documentation/post-1/2.png)


Similarly, the second slice,

```python
ax = sns.heatmap(z[1,:,:], cmap="gray")
```

![Every 16x16 slice]({{site.baseurl}}/assets/images/documentation/post-1/3.png)


and so on. In fact, our whole 16x16x16 cube is made up of 16 of these
16x16 slices (makes sense doesn't it?).

![Heat maps of the slices]({{site.baseurl}}/assets/images/documentation/post-1/3_d_tensor.png)


Just for fun, let's examine the cross-section i.e. all the 16 slices of our cube,

```python
from matplotlib import pyplot as plt
from celluloid import Camera

fig, ax = plt.subplots(1);
camera = Camera(fig);

for i in range(z.size(0)):
    ax.imshow(z[i,:,:], interpolation='nearest', cmap="gray");
    ax.text(0, 0, 'slice :'+str(i+1), bbox={'facecolor': 'white', 'pad': 10});
    camera.snap();

animation = camera.animate();
animation.save('tensors.gif', writer = 'imagemagick');
```

![Every 16x16 slice]({{site.baseurl}}/assets/images/documentation/post-1/tensors.gif)


Our 16x16x16 3-d tensor turned out to be composites of 16 different grayscale images in way!
Looking at matrices and tensors in this way is helpful, especially in the context of image processing since
images have depth in the form of channels (like RGB) and are composites of 2-d tensors.

> Note : The numbers on the axis in all the heatmaps represent indices in the 1st and 2nd dimensions.

A very neat feature in PyTorch is the seamless conversion between PyTorch and numpy data structures (tensors to ndarray) by just calling `.numpy()`.


```python
type(z.numpy())
```
```
    numpy.ndarray
```

This means we can leverage all functions/methods which rely on a numpy array.

Now let's jump a few dimensions and revisit what a 6-d tensor looked like (remember?),

![Every 16x16 slice]({{site.baseurl}}/assets/images/documentation/post-1/5.png)


Let the sizes along each dimension for the tensor be 16x16x16x16x16x16. Think about it :
- The larger cube represents a 3-d tensor of size 16x16x16.
- The smaller cubes within also represents a 3-d tensor of size 16x16x16.
- If the larger cube contains 16x16x16 smaller cubes with each having a size of 16x16x16 we get our 6-d tensor.

Tensors ain't so scary anymore, right? We will go deeper next time with concepts such as broadcasting and tensor operations. In case you find a typo or something didn't quite click, please leave a comment below!

## References
- [PyTorch Documentation](https://pytorch.org/docs/stable/tensors.html)
- [fastai course V3](https://course.fast.ai/videos/?lesson=2)
- [How to get a uniform distribution in a range r1-r2 in pytorch](https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch)
- [Celluloid](https://github.com/jwkvam/celluloid)