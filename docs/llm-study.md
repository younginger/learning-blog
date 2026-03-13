# Transformer的深度学习
## 1.从 Transformer 到 LLaMA
传统的Transformerk Block和现代LLM Block(LLaMA)对比如下:
```
x                               x
↓                               ↓
LayerNorm                       RMSNorm
↓                               ↓
Multi-Head Attention            Self Attention(RoPE)
↓                               ↓
Residual Add                    Residual Add
↓                               ↓
LayerNorm                       RMSNorm
↓                               ↓
Feed Forward Network            SwiGLU FFN
↓                               ↓
Residual Add                    Residual Add
```
主要的四个变化在于：
- 输入正则化有LN变为了RMSN
- 位置编码从决定位置嵌入（Sin/Cos PE）变为了旋转位置嵌入（RoPE）
- 激活函数由ReLU/GELU变为了SwiGLU
- 从普通的attention变为了KV cache优化

注意，总体框架没有改变，只是组件升级，接下来我们仔细讲讲变化的原因。
### 1.1RMSNorm（Root Mean Square Normalization）
第一个方面是训练稳定性：当模型参数由100M变为100B时，由于超大模型计算量过大且梯度不稳定，所以我们从LayerNorm升级为了RMSNorm。

回顾一下LayerNorm,LayerNorm对每个token的隐藏向量x计算均值和方差，再做标准化并缩放。而RMSNorm提出了一个新问题：减去均值有必要吗？作者发现对于Transformer来说只需要控制向量的大小就够了，于是公式简化为：
$$
RMS(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$
再进行归一化：
$$
y_i = \gamma \frac{x_i}{RMS(x) + \epsilon}
$$

这样一来的好处是计算更快、参数更少，示例代码如下：
```
rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)

y = gamma * x / rms
```

任务实践1-实现一个Transformer Block
```
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MutiheadAttention(dim,num_heads=8)

        self.ln2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
    
    def forward(self,x):
        attn_out,_ = self.atten(
            self.ln1(x),
            self.ln1(x),
            self.ln1(x)
        )

        x = x+attn_out

        x = x+self.ffn(self.ln2(x))

        return(x)

```
任务实践2-LLaMA版
```
class LLaMABlock(nn.Module):
    def __init__(self.dim):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim)

        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
```

思考：
为什么Transformer需要残差连接？
- 防止梯度消失
- 保留原始信息：模型学习的是信息修正而不是完全重写
- 残差还有一个好处是让模型可以学习f(x)=0，让模型退化为identity,训练更加稳定。
为什么attention要除以$\sqrt{d}$?

Attention计算的是$QK^T$,如果向量维度很大，这个数值会非常大，这会导致softmax接近one-hot，梯度几乎为0。

为什么Transformer需要Attention+FFN?

如果只有attention就会变为token之间的交流，但每个token的表示能力不足。FFN作用是进行token内部的非线性变换，即信息交流（Attention）+特征变换（FFN）。

### 1.2RoPE
为什么我们需要Position Encoding?因为attention只能计算token相似度，它没有顺序关系，所以我们需要位置编码来解决这个问题。

传统方法都使用Sin/Cos来进行位置编码，但是这样带来的缺点是不能很好表达相对位置关系。RoPE的核心思想就是旋转变量：$$q'=R_{\theta}q$$$$k'=R_{\theta}k$$,其中，$R_{\theta}$是旋转矩阵。

RoPE让$q_i^Tk_j$变为了$q_i^TR_{i-j}k_j$,注意到，R至于i-j有关，所以RoPE天然表示了相对位置。再实际过程中，RoPE会把向量的每两个维度组成一组，每个位置的token会对这些组进行不同角度的旋转，相关代码实现如下：
```
def apply_rotray(q,k,cos,sin):
    q1,q2 = q[...,::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    q_rot =torch.cat(
        [q1 * cos - q2 * sin,
         q1 * sin + q2 * cos],
        dim=-1
    ).flatten(-2)

    k_rot = torch.cat(
        [k1 * cos - k2 * sin,
         k1 * sin + k2 * cos],
        dim=-1
    ).flatten(-2)

    return q_rot, k_rot
```
注意，RoPE只作用在Q和K上。回到最开始的思路，RoPE是将位置编码融入到token之间的相似度中，而注意力机制中token相似度的计算只与Q和K相关，V只是键值。而为什么RoPE支持更长的context呢？这里有两个原因，一是sin/cos的绝对编码表示的绝对位置信息，而attention更需要相对位置信息；二是extrapolation不稳定，一旦推理时模型没有见过某些位置就会出错，二RoPE中i与j的计算时的位置编码只与i-j有关。

### 1.3SwiGLU
很多人学习Transformer时可能认为Attention进行的注意力交互是最重要的,其实在LLM中FFN参数大于Attention参数,所以FFN的设计非常关键。

原始的Transformer FFN采用$FFN(x) = max(0,xW_1)W_2$,采用线性层（升维）-非线性激活函数-线性层（降维）的设计。而GLU提出一个思想：让网络学会控制信息流$GLU(x)=(xW_1)\otimes \sigma (xW_2)$，也就是value*gate，类似于LSTM的门控通道。Google在论文中进一步提出了SwiGLU，公式为$SwiGLU(x) = (xW_1) \otimes swish(xW_2)$,其中swish(x)=x*sigmoid(x)。

为什么SwiGLU能得到更好的效果？原因在于加入了门控属性，可以动态地进行特征选择。LLaMA FFN架构为x->Linear->Linear2->SwiGLU->Linear3，实现如下：
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self,dim):
        super().__init__()
        hidden = dim*4
        self.w1 = nn.linear(dim,hidden)
        self.w2 = nn.linear(dim,hidden)
        self.w3 = nn.linear(hidden,dim)

    def forward(self,x):
        gate = F.silu(self.wi(x))
        value = self.w2(x)

        out = gate * value

        return self.w3(out)
```

### 1.4 KV Cache
LLM存在一个推理问题，假设我们现在生成一句话：王某不是人，每生成一个字模型都要重新计算上下文地Attention，复杂度为$O(n^3)$。

所以我们需要引入KV Cache的核心思想：历史token的K和V不会变化，所以可以缓存。第一次计算K1，V1；第二次计算K2 V2，然后K = [K1,K2]，V = [V1,V2]，复杂度就降为了$O(n^2)$。代码示例：
```
K_cache = []
V_cache = []

def forward(x):

    q = Wq(x)
    k = Wk(x)
    v = Wv(x)

    K_cache.append(k)
    V_cache.append(v)

    K = concat(K_cache)
    V = concat(V_cache)

    attn = softmax(q @ K.T)

    out = attn @ V

    return out
```

注意，只有GPT这类采用因果掩膜技术的才能使用KV Cache，因为每个token的Q,K值只与之前的token有关。而像Bert这类上下文计算token的无法使用KV Cache，因为没加入一个新token都需要重新计算Q,K值。

### 1.5 FlashAttention
Attention的计算公式：$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d}})V$$这个计算需要散步：计算$QK^T$---softmax---乘V。而Attention的真实瓶颈在于$QK^T$据很很大，而不是计算量大的问题。Attention需要大量的memory read/write的过程，这会耗费大量的时间，因此有了FlashAttention,他是长context能实现的关键技术。

FlashAttention的核心思想实不要存完整的$QK^T$，而是分块计算。FlashAttention的计算流程如下：
```
for Q_block:
    for K_block:
        compute attention
        update output
```
这个流程全部在SRAM完成，避免GPU global memory IO。
在pytorch中我们可以实现底层自动调用FlashAttention kernel：
```
import torch
import torch.nn.functional as F

out = F.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True
)
```
### 1.6 mini-LLaMA LLaMA
最后的这个小节我们完成一个完整的mini-LLaMA代码：
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMABlock(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()

        self.norm1 = RMSnorm(dim)
        self.norm2 = RMSnorm(dim)
        self.atten = nn.MultiheadAttention(
            dim,
            heads,
            batch_first = True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )

    def forward(self,x):
        n = self.norm1(x)
        h,_ = self.atten(
            n,
            n,
            n
        )

        x = x+h

        x = x +self.ffn(self.norm2(x))

        return x

class MiniLLaMA(nn.Module):
    def __init__(self,vocab,dim,layers):
        super().__init__()
        self.embed = nn.Embedding(vocab,dim)
        self.blocks = nn.ModuleList(
            [LLaMABlock(dim,8) for _ in range(layers)]
        )

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim,vocab)

    def forward(self, x):

        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.lm_head(x)
```
## 2.LLM Tokenization + Pretraining
### 2.1 Tokenizer
神经网络只能处理数字tensor，所以模型输入环节必须做的一件事便是text->tokens->ids。

在tokenizetion这个环节，最简单的方法就是Character Tokenization,也就是一个字符对应一个数字，但是这样会导致一些较差的sequence对应很长的字符token，这会使attention计算陈本爆炸。另外一种方法是Word Tokenization，也就是一个单词对应一个token，但是这会使得Vocabulary爆炸，每个单词都需要一个token，模型矩阵规模极大。现代方法一般采用Subword tokenization,也就是常见单词对应一个token，罕见单词则采取拆分的策略。例如internationalization对应international和ization或者inter、nation、al和ization。而GPT系列采用的是另一种方法BPE(Byte Pair Encoding)。它的核心思想是不断合并最常见的字符对，举个例子，最初的训练语料是：low、lower、newest和widest。则初始tokens就是l、o、w、l、o、w、e、r、n、e、w、e、s、t、w、i、d、e、s、t；然后我们不断统计最常见的字符并进行合并，最终vocabulary就会变为l、o、w、lo、low、er和est。

不同的tokenizetion策略会得到不同的vocabulary，这会导致训练效果大不相同，所以tokenization的策略至关重要。接下来我们介绍LLaMA的Tokenizer算法---SentencePiece tokenizer。