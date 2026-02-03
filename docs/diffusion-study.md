# 扩散模型学习记录
这是和哈基米一起学习的第一次内容，主要关于扩散模型的原理和应用。
## 1. 数学原理

扩散模型的前向过程（Forward Process）是一个马尔可夫链。在时刻 $t$，图像 $x_t$ 的分布由 $x_{t-1}$ 决定：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})
$$

通过重参数化技巧，我们可以直接从 $x_0$ 得到任意时刻 $x_t$：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 是高斯噪声。

## 2. 代码实现

这是我在 PyTorch 中的实现代码：

```python
import torch

def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
```