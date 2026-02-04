# 扩散模型学习记录
这是和哈基米一起学习的第一次内容，主要关于扩散模型的原理和应用。
## 1. 数学原理
### 简述
扩散模型的前向过程（Forward Process）是一个马尔可夫链。在时刻 $t$，图像 $x_t$ 的分布由 $x_{t-1}$ 决定：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})
$$

通过重参数化技巧，我们可以直接从 $x_0$ 得到任意时刻 $x_t$：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ 是高斯噪声。

首先，我们明确一个概念，图像生成模型想做的一件事是学习真实图像的概率分布即$p_{data}(x|c}$,其中x是真实世界图像，c是文本条件。第一次看到真是图像的概率分布时，大家可能都有点懵逼。打个比方，一张224*224*3的图像，每个像素点上的RBG通道值都可以容易取值，那么可以得到的图像是天文级别，但是真实世界是不可能的，你无法看到黑色的雪花、弯腰的钢背兽。所以，真实世界的所有图像其实是满足一个概率分布的。

但是这个分布太复杂了，根本无法直接采样来还原，所以diffusion做了一个巧妙的操作:换一个好采样的点。我无法知道我想要复杂自然图像的概率分布，但是我将复杂分布慢慢抹平为高斯噪声，那么生成的时候只需要与抹平的操作反过来就行了，对应的就是加噪和去噪，我们学习的是这一过程，而非原始分布。那么为什么现在图像生成的输入都是噪声，而我们的提示词只是约束条件呢？打个比方：如果我们直接用提示词来生成图片，“一只在雪地里的白色哈士奇”对应现实世界会有无穷多张合理的图像，如果我们要拟合这样一个一对多的模型太复杂了，而且压根无法建立损失和训练方向。但是如果我们用噪声来提供随机性，决定“是哪一种可能世界”，就要容易得多！！！

在深度学习中Diffusion Model的前向和反向的过程如下：
### 去噪与加噪
![alt text](notebook_images\image.png)

这个前向加噪-反向去噪的过程可以理解为另类的encoder-decoder模型。

在加噪的过程中，我们选定了一组参数$\alpha_1, \alpha_2, \cdots$作为权重参数，代表不同步的加噪程度。通过公式$\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$得到加噪后的模糊图像（注意，$hat{\alpha}_t = \sum_0^t \alpha_i$, 相当于累计每一步的加噪程度来计算单步加噪程度，逐渐变大，代表图片加噪越来越多，图片越来越模糊）。而训练这是通过加噪图像和步骤$t$来预测加入噪声的形状。公式如下：$min|\epsilon - \hat{\epsilon}(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)|$

![alt text](notebook_images\image1.png)

这里加噪过程需要注意一个点，实际上训练的过程不是理想中的马尔科夫链那样，由此状态$x_t$预测上一状态$x_{t-1}$(一步步加噪)；我们是通过原本的图像混入严重噪声的图像和此时的步骤$t$去预测加入的噪声。

![alt text](notebook_images\image2.png)


这是因为，实际上前后两次加噪可以等同于一次加噪：

![alt text](image.png)

然后我们用$\alpha$来替换这里的$\beta$。

在去噪（生成图像）过程中，同样有一个非常有意思的点。理论上噪声图像减去预测的噪声形状应该等于我们想要的上一步加噪图像，但是我们最终还加上了一个高斯噪声。为什么？

![alt text](notebook_images\image3.png)

还是那个问题，我们虽然学习的是加噪和去噪的过程，但是我们目标是学习真实世界的分布，真实世界是随机的。如果我们不采样噪声，就是在假装这个分布是“退化成一个点”的。从数学角度上来讲，我们的前向扩散过程是$q(x_t|x_{t-1})=\mathcal{N}(\sqrt{\bar{\alpha}_t} x_{t-1},(1-\bar{\alpha}_t)\mathbf{I})$,这是一个随机映射，所以反向分布也一定是随机的。

### KL散度
KL散度是图像生成模型常用的损失函数，回到最初的问题，我们学习的是加噪和去噪的过程，而非原始分布，那么我如何定义一个函数来评估生成图像和实际图像的差异，或者说如何评估我的模型根据输入能够生成出对应图像的概率？这就是KL散度做的事，它的含义可以理解为如果真实世界按我们的数据集$P_{data}$来说话，而我用$P_{\zeta}$来解释，会浪费多少信息。推导公式如下：

![alt text](notebook_images\image4.png)

对于diffusion来说：

![alt text](notebook_images\image5.png)

也就是每个时间步，我如何把图像往真实方向推一点点，于是目标变成了:$\sum_{t=1}^T E[D_{KL}(q(x_{t-1}|x_t),x_0)||p_{\zeta}(x_{t-1}|x_t)]$

在合适的参数化下，上面的目标等价于MSE:$E_{x_0,\epsilon,t}[||\epsilon - \epsilon_{\zeta}(x_t,t)||^2]$

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