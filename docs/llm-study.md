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
LLM存在一个推理问题，假设我们现在生成一句话：王某不是人，每生成一个字模型都要重新计算上下文的Attention，复杂度为$O(n^3)$。

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
### 1.6 mini-LLaMA
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

在tokenizetion这个环节，最简单的方法就是Character Tokenization,也就是一个字符对应一个数字，但是这样会导致一些较差的sequence对应很长的字符token，这会使attention计算成本爆炸。另外一种方法是Word Tokenization，也就是一个单词对应一个token，但是这会使得Vocabulary爆炸，每个单词都需要一个token，模型矩阵规模极大。现代方法一般采用Subword tokenization,也就是常见单词对应一个token，罕见单词则采取拆分的策略。例如internationalization对应international和ization或者inter、nation、al和ization。而GPT系列采用的是另一种方法BPE(Byte Pair Encoding)。它的核心思想是不断合并最常见的字符对，举个例子，最初的训练语料是：low、lower、newest和widest。则初始tokens就是l、o、w、l、o、w、e、r、n、e、w、e、s、t、w、i、d、e、s、t；然后我们不断统计最常见的字符并进行合并，最终vocabulary就会变为l、o、w、lo、low、er和est。

不同的tokenizetion策略会得到不同的vocabulary，这会导致训练效果大不相同，所以tokenization的策略至关重要。接下来我们介绍LLaMA的Tokenizer算法---SentencePiece tokenizer。

在传统的Tikenization中，通常需要先用空格将句子分词（Pre-tokenization），但这对于中文等不以空格分隔单词的语言非常不友好。SentencePiece的核心思想是将空格也视为一个普通的字符，直接对未经过任意预处理的生文本流进行训练。它有以下两个重要特征：
 - BPE算法 on SenttencePiece：底层的词表合并逻辑依然是BPE，但直接处理包含空格的完整字符串。
 - Byte-fallback机制:当遇到此表中不存在的罕见字符时，传统的Tokenizer会输出<UNK>导致信息丢失。LLaMA会将这些未知字符拆解成底层的UTF-8字节，用256个基础字节Token来表示它们。这样一来，LLaMA的词表就能消除out of vocabulary现象。

### 2.2预训练数据准备:Data Packing
大预言模型的预训练目标非常简单：Next Token Prediction。
但是真是的文档长度不一，未来充分利用GPU的并行计算能力，我们需要把文本拼接起来，塞满模型的上下文窗口，这个过程就叫Packing。

Packing的逻辑就是用 <EOS>（End of Sentence）标记把不同文档连起来，并切分成固定长度的块（Block）。
举个例子：

Doc1: "我爱学习。"Doc2: "LLaMA很强大。"

拼合: [BOS]我爱学习。[EOS][BOS]LLaMA很强大。[EOS]

切块(假设长度为8):

Chunk 1: [BOS], 我, 爱, 学, 习, 。, [EOS], [BOS]

Chunk 2: L, L, a, M, A, 很, 强, 大

LLaMA的此表一开始只有32000代下，但是今天的大模型基本词表都高达60000甚至130000，这是因为LLaMA原生设计主要针对英语，英语依靠26个字母加常见词缀，32000的词表就可以覆盖绝大多数subword。但是LLaMA一遇到中文字符就会触发Byte-fallback，将一个汉字拆解为3个UTF-8字节。这就导致 LLaMA 遇到中文时，一个汉字会消耗 3 个 Token，大大降低了模型的上下文利用率（Context Length），也变相增加了注意力机制的计算压力。因此，未来提高中文处理效率，把大量常见汉字和中文词组加入词表中可以减少消耗的token数，提高计算效率。

在预训练时，我们的输入和标签在形状和内容上的关系类似于向左平移了一位。比如：如果序列是 [1, 2, 3, 4, 5]，作为训练数据时：

输入 x 是 [1, 2, 3, 4]（看到 1 预测 2，看到 1,2 预测 3...）

标签 y 是 [2, 3, 4, 5]

现在让我们用代码来实现Data Packing过程:
```
import torch

def pack_data(tokenized_docs,max_len):
    # 展平文档
    flat_tokens = []
    for doc in tokenized_docs:
        flat_tokens.extend(doc)
    #计算能切分出多少个完整的block
    total_len = len(flat_tokens)
    num_blocks = total_len//max_len

    # 截取正好能被 max_len 整除的长度
    flat_tokens = flat_tokens[:num_blocks * max_len]
    
    # 转换为 Tensor 并 Reshape
    tensor_data = torch.tensor(flat_tokens)
    packed_data = tensor_data.view(num_blocks, max_len)
    
    return packed_data
```
## 3.LLM预训练的核心
预训练阶段，模型唯一的目标就是根据上文猜测下一个字是什么。
### 3.1 自回归域交叉熵损失
在mini-LLaMA中，我们最后经过了一个lm_head（Linear层），输出的维度是 (batch_size, seq_len, vocab_size)。这代表对于序列中的每一个位置，模型都给出了整个词表的概率分布（Logits）。

如何计算模型猜得准不准？我们使用交叉熵损失 (Cross Entropy Loss)。假设当前词的真实下一个词是 $y_i$，模型预测出的对应概率分布为 $\hat{y}_i$，损失函数计算如下：$$Loss = - \frac{1}{N} \sum_{i=1}^{N} \log P(\hat{y}_i = y_i \vert{} x_{<i})$$ 在 PyTorch 中，这可以通过 F.cross_entropy 来实现。但需要注意：我们需要将形状展开，因为标准的交叉熵函数期望输入是二维的 (N, C)，目标是一维的 (N)。

```python
import torch.nn.functional as F

def compute_loss(logits, targets):
    # logits shape: (batch, seq_len, vocab_size)
    # targets shape: (batch, seq_len)
    
    # 展平以便计算
    logits = logits.view(-1, logits.size(-1)) # 变为 (batch * seq_len, vocab_size)
    targets = targets.view(-1)                # 变为 (batch * seq_len)
    
    # ignore_index=-100 通常用于忽略 padding 带来的影响
    loss = F.cross_entropy(logits, targets, ignore_index=-100)
    return loss
```

### 3.2 学习率调度
LLM训练的标配是Cosine Learning Rate Schedule with Warmup（带预热的余弦退火学习率）。

- Warmup（预热）：在训练的前几千步，学习率从 0 线性增长到最大值。这段时间模型权重还是随机的，防止梯度更新过于剧烈。

- Cosine Decay（余弦衰减）：达到最大值后，随着训练进行，学习率按照余弦曲线平滑下降，在训练末期变得极小，帮助模型在损失函数的谷底精细收敛。

### 3.3混合精度训练
在超大参数的模型中，FP32会导致显存瞬间爆炸，并且计算极其缓慢。所以我们引入了半精度计算（16-bit），而 LLM 界目前的绝对主流是 BF16 (Bfloat16)，而不是传统的FP16。
- FP32 (单精度)：1位符号，8位指数，23位小数。

- FP16 (半精度)：1位符号，5位指数，10位小数。（容易发生数值溢出 Overflow，因为最大只能表示 65504）。

- BF16 (Brain Floating Point)：1位符号，8位指数，7位小数。

在深度学习中，数值的范围（指数位）比数值的精度（小数位）重要得多！BF16 保留了和 FP32 完全一样的 8 位指数，这意味着它拥有和 FP32 相同的数值范围，对于极其容易产生极大数值的 Attention 机制和巨大梯度的 LLM 训练来说至关重要。

思考 1：预训练阶段需要使用 KV Cache 吗？为什么？

不需要。核心原因在于训练和推理的计算方式完全不同：
- 推理阶段（需要 KV Cache）：模型是“一个字一个字”往外蹦的（串行自回归）。生成第 100 个字时，必须把前 99 个字的 K 和 V 重新算一遍，所以需要缓存来省时间。
- 预训练阶段（不需要 KV Cache）：我们已经把整段话（比如 2048 个 token）一次性喂给了模型，配合 Causal Mask（因果掩码），GPU 会在一次前向传播中并行算出所有位置的 Attention 和 Loss，根本不存在“逐个生成、重复计算”的过程。而且训练时每走一步，模型权重 $W_k, W_v$ 都在更新，缓存下来的旧 K、V 也会立刻失效。

思考 2：targets 应该怎么从原始数据里切出来匹配 x？

核心原则是错位对齐（Shift by 1）：

如果原始序列是 [1, 2, 3, 4, 5]（长度为 5）：

- 输入 x 取前 4 个：[1, 2, 3, 4]（切片 [:, :-1]）

- 标签 y (targets) 取后 4 个：[2, 3, 4, 5]（切片 [:, 1:]）

这样 logits 和 targets 的形状就完全对齐了（都是长度 4）：

- 位置 0：模型看到 [1] → 预测 2

- 位置 1：模型看到 [1, 2] → 预测 3

- 位置 2：模型看到 [1, 2, 3] → 预测 4

- 位置 3：模型看到 [1, 2, 3, 4] → 预测 5

(注：在实际工程的 Data Packing 中，如果模型上下文长度是 2048，我们通常会把每个 block 切成 2049 的长度，这样切完 x 和 y 后，输入模型的长度正好是完整的 2048。)

```python
import torch

def training_step(model, token_seq, optimizer):
    # 假设 token_seq 形状为 (batch_size, seq_len)，例如 tensor([[1, 2, 3, 4, 5]])
    optimizer.zero_grad()
    
    # 1. 构造输入 x 和标签 y (使用切片操作，保留 batch 维度)
    x = token_seq[:, :-1]  # 形状: (1, 4) -> [[1, 2, 3, 4]]
    y = token_seq[:, 1:]   # 形状: (1, 4) -> [[2, 3, 4, 5]]
    
    # 2. 前向传播
    # logits shape: (1, 4, vocab_size)
    logits = model(x)
    
    # 3. 计算 Loss 
    loss = compute_loss(logits, y)
    
    # 4. 反向传播与更新
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## 4.分布式训练与显存优化
### 4.1显存
训练时的显存占用主要分为两大块：静态显存（模型状态）和动态显存（激活值）。

假设模型参数量为$\Psi$（比如 7B 即 $7 \times 10^9$），在目前最主流的 混合精度（BF16）+ AdamW 优化器 训练配置下，静态显存由以下三部分组成：
- 模型参数 (Parameters)：前向和反向传播使用 BF16，每个参数占 2 字节 → $2\Psi$ 字节。
- 梯度 (Gradients)：反向传播算出的梯度也是 BF16，每个参数占 2 字节 → $2\Psi$ 字节。
- 优化器状态 (Optimizer States)：这是真正的“显存黑洞”。为了防止极小的梯度更新在 BF16 下精度丢失（Underflow），AdamW 内部必须用 FP32（4 字节） 维护三份状态：
    - FP32 主权重副本 (Master Weights)：$4\Psi$ 字节
    - 一阶动量 (Momentum, FP32)：$4\Psi$ 字节
    - 二阶动量 (Variance, FP32)：$4\Psi$ 字节
    - 优化器合计：$12\Psi$ 字节。
把它们加起来，光是把模型静态放进显存准备训练，就需要：$$\text{静态显存} = 2\Psi + 2\Psi + 12\Psi = 16\Psi \text{ 字节}$$算一笔账：训练一个 7B 模型，仅静态显存就需要 $7 \times 16 = \mathbf{112\text{ GB}}$，这还不包括随batch_size和seq_len暴涨的激活值显存。
### 4.2数据并行
目前的训练一般都采取多卡并行的方式，最基础的分布式策略是DDP：
- 做法：每张 GPU 上都完整复制一份模型（参数、梯度、优化器）。把一个大的 Batch 切成 $N$ 份，每张卡各算各的前向和反向传播。
- 同步：算完梯度后，所有 GPU 通过 All-Reduce 通信把梯度求平均，然后每张卡各自用平均后的梯度更新权重。
- 局限性：DDP 极大提升了训练速度，但完全不省显存！每张卡上依然存着一模一样的 $16\Psi$ 字节，存在巨大的显存冗余。

### 4.3DeepSpeed ZeRO（Zero Redundancy Optimizer）
既然 DDP 的痛点是“每张卡都存了重复的数据”，微软提出的 ZeRO（零冗余优化器） 思想非常直白：把这些重复的状态切分（Shard）开，平均摊到 $N$ 张 GPU 上，谁需要时再通过高速网络（如 NVLink）临时通信拿一下。

