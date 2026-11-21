# World Models
## 1.奠基之作World Models（2018）
第一章我们将聚焦于 World Models (2018) 的奠基之作。整个世界模型由三个组件（V、M、C）构成，而第一步，我们需要为智能体装上“眼睛”——也就是 V 模型（Vision Model）。

由于高维像素空间（如游戏画面）对强化学习来说状态空间过于庞大且充满噪声，V 模型的核心任务是降维与特征提取：将复杂的图像压缩成低维的、连续的隐状态（Latent Vector），并在这个过程中过滤掉无关细节。

### 1.1 变分自编码器（VAE）
为了构建V模型，Ha等人采用了变分自编码器 (Variational Autoencoder, VAE)。

为什么用VAE而不是普通的卷积自编码器（CNN Autoencoder）？

如果你使用普通的自编码器，输入一张图像，模型会将其编码为隐空间中的一个确定的点。但这种离散的、点对点的映射会导致隐空间非常不平滑——稍微改变隐向量的值，解码出来的图像可能就变成了一团乱码。这对于后续的连续控制和多任务回归是致命的。

VAE通过概率分布解决了这个问题:

- 编码分布：VAE 的编码器不再输出一个确定的向量，而是输出一个高斯分布的参数，即均值 $\mu$ 和方差 $\sigma^2$。
- 重参数化技巧:为了让梯度能够反向传播，我们从标准正态分布 $\epsilon \sim \mathcal{N}(0, 1)$ 中采样，然后计算 $z = \mu + \sigma \odot \epsilon$。
- 损失函数 (Loss Function)：VAE 的优化目标包含两部分，重构误差和正则化项（KL 散度）。其核心数学对齐公式为：

$$\mathcal{L} = \mathbb{E}_{q_{\phi}(z\vert{}x)}[\log p_{\theta}(x\vert{}z)] - D_{KL}(q_{\phi}(z\vert{}x) \Vert{} p(z))$$

- 第一项是重构损失 (Reconstruction Loss)：通常用 MSE 或 Cross Entropy 计算，保证解码器能从隐变量 $z$ 还原出原图。
- 第二项是KL 散度 (KL Divergence)：迫使编码器学到的分布 $q_{\phi}(z\vert{}x)$ 尽可能贴近标准正态分布 $\mathcal{N}(0, 1)$。这起到了正则化的作用，使得隐空间紧凑且连续。
#### 1.1.1 VAE代码编写
现在，我们用 Python 和 PyTorch 来实现这个 VAE。我们将隐向量的维度设定为 latent_dim = 32（这是原论文中应对 CarRacing-v0 环境的维度）。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()
        
        # ----------------- Encoder (编码器) -----------------
        # 假设输入图像尺寸为 (3, 64, 64)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        # 将卷积特征展平后，映射到均值和对数方差
        # 64x64 经过 4 次 stride=2 的卷积后，空间维度变为 4x4
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # ----------------- Decoder (解码器) -----------------
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1) # 展平操作
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        # 计算标准差 (logvar = 2 * log(sigma) -> sigma = exp(0.5 * logvar))
        std = torch.exp(0.5 * logvar) 
        # 从标准正态分布采样
        eps = torch.randn_like(std)   
        # Z = \mu + \sigma \odot \epsilon
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 256, 4, 4) # 恢复空间维度
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # 最后一层使用 Sigmoid，将像素值限制在 (0, 1) 之间
        x = torch.sigmoid(self.deconv4(x)) 
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# ----------------- 损失函数计算 -----------------
def vae_loss_function(recon_x, x, mu, logvar):
    # 1. 重构损失 (这里使用 MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL 散度 (推导后的闭式解)
    # D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_divergence
```

#### 1.1.2 VAE训练
由于M模型（Memory）的输入并不是原始图像，而是V模块提取出来的隐向量序列$z_t$。如果 V 模块没有经过预训练，提取出来的 $z_t$ 就是毫无意义的随机噪声，M 模型也就无从学习环境的时序动态。

在世界模型的经典设定中，V模型的训练是无监督且独立于奖励信号的。以下是具体的操作逻辑和代码骨架：

#### 1.1.3 使用随机策略收集数据
由于我们的智能体现在没有任何策略（C 模型还没出生），我们只能让它在环境中“瞎跑”（Random Policy），目的是尽可能多地探索环境的不同状态，录下这些画面。

和处理结构化数据时习惯的方式一样，我们将收集到的环境图像打包保存为 .npy 格式的数组文件，方便后续构建 PyTorch 的 DataLoader。

```python
import gym
import numpy as np
import cv2

def collect_random_rollouts(env_name="CarRacing-v2", num_episodes=100, max_steps=1000):
    env = gym.make(env_name, continuous=True)
    all_frames = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        for step in range(max_steps):
            # 随机采样动作 (转向, 油门, 刹车)
            action = env.action_space.sample() 
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 图像预处理: 裁剪、缩放、转置为 (C, H, W)
            # 原图通常是 96x96x3 的 RGB
            frame = cv2.resize(obs[:84, :, :], (64, 64)) 
            frame = frame.transpose((2, 0, 1)) / 255.0  # 归一化到 0-1
            
            all_frames.append(frame)
            
            if terminated or truncated:
                break
                
    env.close()
    
    # 转换为 NumPy 数组并保存
    dataset = np.array(all_frames, dtype=np.float32)
    np.save('carracing_random_rollouts.npy', dataset)
    print(f"数据收集完成，共包含 {len(dataset)} 帧图像，形状为 {dataset.shape}")

# 执行收集
# collect_random_rollouts()
```
#### 1.1.4 离线训练V模块
拿到 carracing_random_rollouts.npy 后，接下来的工作就回归到了标准的计算机视觉模型训练流程：

- 构建 torch.utils.data.DataLoader。
- 将图像送入我们在上一节写的 VAE 网络中。
- 计算重构损失（MSE）和 KL 散度，使用 Adam 优化器反向传播更新权重。
- 训练完成后，冻结 VAE 的权重。

#### 1.1.5 为M模型准备时序数据
当V模块训练完毕后，我们还需要做一次数据转换。我们需要将之前收集到的（或者重新跑一波收集的）图像数据帧 $x_t$，通过训练好的 VAE 编码器，全部映射成隐向量 $z_t$。

此时，我们就把高维的视觉像素流，完全降维成了形状为 (Sequence_Length, 32) 的一维特征时间序列。这不仅大幅降低了对计算资源的消耗，也为处理复杂的时序依赖做好了完美的前置准备。

### 1.2 M模型 (Memory Model)
智能体能在多大程度上“理解”所处的环境，并对未来做出准确预判，全靠M模型。如果将V模型看作是感知视觉神经，那么M模块就是海马体--它不仅要记住过去，还要预测未来。

#### 1.2.1 理论学习 - 混合密度循环网络 (MDN-RNN)
就像在之前的强化学习优化算法中所学的，要让智能体在环境中做出最优决策（比如通过 PPO 寻优），它必须对环境的状态转移有一个清晰的认知。而 M 模块的任务正是建立这样一个关于环境的内部动力学模型 (Dynamics Model)。

在 World Models 中，Ha & Schmidhuber 使用了 MDN-RNN (Mixture Density Network - Recurrent Neural Network)。

基于时间序列的预测，我们首先想到的肯定是 RNN（实际操作中通常用 LSTM 或 GRU）。在每个时间步 $t$，RNN 接收当前的隐状态 $z_t$（由 VAE 输出）和智能体刚刚执行的动作 $a_t$，结合其隐藏层记忆 $h_t$，去预测下一个时间步的隐状态 $z_{t+1}$。

但在复杂的现实或游戏环境中，未来往往是非确定性 (Stochastic) 的。举个例子：你在《超级马里奥》里顶一个带有问号的砖块，下一秒可能蹦出一个金币，也可能蹦出一个蘑菇。如果强迫一个普通的 RNN 预测唯一的确定输出，它大概率会预测出一个“长着蘑菇斑点的金币”（即多个可能结果的平均值），这显然违背物理直觉。

为了解决这个问题，M 模块引入了混合密度网络 (MDN)。它不再让 RNN 直接输出下一个隐状态的具体数值 $z_{t+1}$，而是让 RNN 输出一组高斯分布的参数（多个高斯分布混合在一起，即高斯混合模型 GMM）。在 MDN-RNN 架构中，RNN 在时间步 $t$ 接收输入 $(z_t, a_t)$ 后，会输出三大类参数，用于构建预测分布：

- 混合权重 $\pi$ (Pi)：表示第 $i$ 个高斯分布被选中的概率（比如 70% 概率出金币，30% 概率出蘑菇）。所有 $\pi$ 之和必须为 1。
- 均值 $\mu$ (Mu)：每个高斯分布的中心点。
- 方差 $\sigma$ (Sigma)：每个高斯分布的宽度。
因此，M 模块预测的 $z_{t+1}$ 是从这个混合概率分布 $P(z_{t+1} \vert{} a_t, z_t, h_t)$ 中采样得到的。这就允许模型预测出多种可能分岔的未来，极大增强了模型对复杂环境的表征能力。

由于输出是概率分布，我们在采样时可以引入一个温度参数 (Temperature, $\tau$)：

- 低温度 ($\tau < 1$)：放大混合权重中的最大值，让模型的预测趋于保守和确定，只选择最有可能发生的未来。
- 高温度 ($\tau > 1$)：拉平权重，让模型更敢于“做梦”，采样出那些概率较小但依然合理的边缘状态（比如赛车突然打滑）。这构成了智能体在自己构建的“梦境（幻觉）”中进行想象训练的理论基石。

#### 1.2.2 MDN-RNN代码部署
我们使用单层LSTM作为时序特征提取器，并在其输出端挂载三个线性层，分别预测混合权重 $\pi$、均值 $\mu$ 和标准差 $\sigma$。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_size=256, num_gaussians=5):
        super(MDNRNN, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        
        # LSTM 接收前一时刻的隐状态 z_t 和动作 a_t (维度相加: 32 + 3 = 35)
        self.rnn = nn.LSTM(
            input_size=latent_dim + action_dim, 
            hidden_size=hidden_size, 
            batch_first=True
        )
        
        # MDN 线性输出层
        # 1. pi: 每个高斯分布的权重
        self.fc_pi = nn.Linear(hidden_size, num_gaussians)
        # 2. mu: 均值 (每个高斯分布都需要 latent_dim 维度的均值)
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_dim)
        # 3. sigma: 标准差
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_dim)

    def forward(self, z, action, hidden=None):
        # z shape: (batch_size, seq_len, latent_dim)
        # action shape: (batch_size, seq_len, action_dim)
        
        # 拼接特征作为 LSTM 输入
        seq_input = torch.cat([z, action], dim=-1) 
        rnn_out, hidden = self.rnn(seq_input, hidden)
        
        # 获取 MDN 参数并改变形状以便后续计算
        batch_size, seq_len, _ = rnn_out.size()
        
        # 归一化权重，确保 \sum pi = 1
        pi = self.fc_pi(rnn_out)
        pi = F.softmax(pi, dim=-1) 
        
        # Reshape 为 (batch, seq, num_gaussians, latent_dim)
        mu = self.fc_mu(rnn_out).view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        
        # 标准差必须为正数，这里使用 exp 转换
        sigma = self.fc_sigma(rnn_out)
        sigma = torch.exp(sigma).view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        
        return pi, mu, sigma, hidden
```
由于M模型的训练目标是最大化真实下一状态$z_{t+1}$在预测分布中的概率。等价于最小化负对数似然损失 (Negative Log-Likelihood, NLL)。

数学公式为：

$$\mathcal{L} = -\log \sum_{i=1}^{K} \pi_i \mathcal{N}(z_{t+1} \vert{} \mu_i, \sigma_i)$$

在代码中，我们利用 torch.distributions.Normal 在对数空间完成稳定运算：
```python
def mdn_loss_function(pi, mu, sigma, target_z):
    # target_z 是真实的 z_{t+1}, shape: (batch_size, seq_len, latent_dim)
    # 我们需要将其扩展，以匹配 mu 和 sigma 的 num_gaussians 维度
    # shape 变为: (batch_size, seq_len, 1, latent_dim)
    target_z = target_z.unsqueeze(2)
    
    # 构造正态分布
    dist = torch.distributions.Normal(mu, sigma)
    
    # 计算 log N(z | mu, sigma)
    # 假设隐空间的各个维度是独立的，因此对最后一维 (latent_dim) 求和
    # log_prob shape: (batch_size, seq_len, num_gaussians)
    log_prob = dist.log_prob(target_z).sum(dim=-1)
    
    # 计算 log(pi) 并加入微小常量防止 log(0)
    log_pi = torch.log(pi + 1e-8)
    
    # 计算 log(pi * N) = log(pi) + log N
    log_component_prob = log_pi + log_prob
    
    # 使用 logsumexp 计算 \log(\sum \exp(\cdot))，这是防止下溢出的核心
    # log_mix_prob shape: (batch_size, seq_len)
    log_mix_prob = torch.logsumexp(log_component_prob, dim=-1)
    
    # 负对数似然求均值作为最终 Loss
    loss = -log_mix_prob.mean()
    return loss
```

### 1.3 C模型（Controller 与 CMA-ES）
#### 1.3.1 极简的C模型和CMA-ES
V 模型将当前画面压缩成了隐变量 $z_t$（32 维），M 模型将历史序列总结成了隐状态 $h_t$（256 维）。对于 C 模型来说，它所需要的一切环境信息都已经在这两个向量里了。

因此，Ha & Schmidhuber 故意将 C 模型设计得极其简单：它仅仅是一个单层线性神经网络（无激活函数）。

其数学表达为：

$$
a_t = W_c [z_t, h_t] + b_c
$$

其中，$[z_t, h_t]$ 是拼接后的特征向量（$32 + 256 = 288$ 维）。如果动作空间 $a_t$ 是 3 维（如赛车游戏中的转向、油门、刹车），那么权重矩阵 $W_c$ 的大小仅为 $3 \times 288$，偏置 $b_c$ 为 3 维。

这体现了表征学习的核心哲学：如果特征提取（V）和环境动力学建模（M）做得足够好，最终的决策（C）应该是非常线性且直接的。同时，极小的参数量（不到 1000 个参数）正是为了配合接下来的进化算法。

CMA-ES全程为协方差矩阵自适应进化策略，是一种用于连续域非凸黑盒优化的无导数算法。它不依赖目标函数的梯度，而是通过迭代更新一个多元正态分布的均值和协方差矩阵，来引导整个搜索群体（种群）向全局最优解移动。

CMA-ES 的精髓在于对协方差矩阵 $C$ 的自适应调整，它允许算法自动学习目标函数的底层拓扑结构。算法按以下步骤循环运行：

- 采样 (Sampling)：以当前均值 $m$ 为中心，利用多元正态分布 $\mathcal{N}(m, \sigma^2 C)$ 生成一批候选解（种群）。这里的 $\sigma$ 是全局步长。
- 评估与选择 (Evaluation & Selection)：将生成的候选解输入目标函数（类似于计算 Reward 或 Loss），评估其表现，并挑选出排名前 $\mu$ 个最优秀的个体。
- 均值更新 (Update Mean)：将分布的均值 $m$ 移动到这批优秀个体的加权中心。
- 协方差矩阵更新 (Update Covariance)：利用选出个体的分布趋势来更新矩阵 $C$。

从矩阵理论的角度，我们可以通过特征分解来直观理解这一过程：

$$
C = B D^2 B^T
$$

- $B$（正交阵）：其列向量为特征向量，决定了搜索空间的主成分方向（旋转）。
- $D$（对角阵）：其对角线元素为特征值的平方根，决定了沿各个特征向量方向的分布方差（缩放比例）。
- 通过动态更新 $B$ 和 $D$，CMA-ES 能够自动识别出目标函数中的敏感方向和不敏感方向，让搜索分布的形态（一个高维椭球体）逐渐贴合目标函数的真实地形。

#### 1.3.2 为什么采用进化算法而不是传统RL
在优化策略时，像 PPO 或 TRPO 这样的传统强化学习算法依赖于优势估计 (Advantage Estimation) 和基于梯度的反向传播。但在原始的 World Models 架构中，采用传统 RL 会面临难以逾越的工程障碍，而采用协方差矩阵适应进化策略 (CMA-ES) 却能完美破局。原因有三:

- 避开BPTT梯度爆炸：如果使用 PPO 等 Actor-Critic 方法，为了更新 Actor（这里的 C 模型），梯度需要从最终的奖励信号开始，穿过 C 模型，再沿着时间轴反向传播穿过长达数百步的 M 模型 (LSTM)，甚至穿过 V 模型。在这么长的序列上进行基于时间的截断反向传播 (BPTT)，极易导致梯度消失或爆炸，网络极难收敛。而相比之下，而 CMA-ES 是无梯度 (Gradient-free) 的。它完全把 C 模型的权重当作一个黑盒，只关心“这组权重打完一局游戏最终得分是多少”，彻底绕开了反向传播。
- 无需奖励塑性与价值网络:传统 RL 需要密集且即时的奖励信号来训练 Critic (价值网络) 从而评估优势。而在很多任务中（比如只在到达终点才给分的迷宫），奖励是极其稀疏的。CMA-ES 只需要一个评估指标（Cumulative Reward，即一局游戏的总得分），它不在乎你在第几步拿到了分，只看整体表现。
- 极佳的并行搜索能力：CMA-ES 维护了一个多维高斯分布来表示参数的搜索空间。每一代，它从这个分布中采样出多个不同的权重组合（种群），让它们同时去环境里跑游戏。然后根据得分，更新这个高斯分布的均值和协方差矩阵，让下一代朝着高分区域集中。这种天然的并行性非常适合在 CPU 集群上进行分布式评估。

#### 1.3.3 Controller的代码实现
首先是Controller本身的构建：
```python
import torch
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    def __init__(self, latent_dim=32, hidden_size=256, action_dim=3):
        super(Controller, self).__init__()
        # 极简的单层线性映射
        self.fc = nn.Linear(latent_dim + hidden_size, action_dim)

    def forward(self, z, h):
        # z: 形状 (batch, 32)
        # h: 形状 (batch, 256)
        x = torch.cat([z, h], dim=-1)
        action = self.fc(x)
        # 如果是连续控制（如 -1 到 1 之间），可以使用 tanh 等进行边界截断
        return torch.tanh(action)
```
然后是我们利用CMA-ES对C模型进行训练：
```python

# pip install cma
import cma

def evaluate_controller(weights, env, v_model, m_model, controller):
    """
    评估一组特定的参数在一局游戏中的总得分
    """
    # 1. 将 1 维的进化权重注入到 Controller 网络中
    torch.nn.utils.vector_to_parameters(torch.tensor(weights, dtype=torch.float32), controller.parameters())
    
    obs, info = env.reset()
    total_reward = 0
    hidden = None # RNN 的初始隐藏状态
    
    with torch.no_grad(): # 推理模式，不需要梯度
        for step in range(1000): # 最大步数
            # 图像预处理 -> V 模型降维得到 z
            obs_tensor = preprocess_image(obs) 
            mu, logvar = v_model.encode(obs_tensor)
            z = v_model.reparameterize(mu, logvar)
            
            # 提取 M 模型当前的隐藏状态 h
            # (如果 hidden 为 None，则使用全 0 向量)
            h = hidden[0].squeeze(0) if hidden is not None else torch.zeros(256)
            
            # C 模型输出动作
            action = controller(z, h).numpy()
            
            # 与环境交互
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # M 模型预测下一步 (更新 hidden 状态)
            # 注意：实际代码中需要处理张量维度匹配 (batch, seq, dim)
            _, _, _, hidden = m_model(z.unsqueeze(0).unsqueeze(0), 
                                      torch.tensor(action).unsqueeze(0).unsqueeze(0), 
                                      hidden)
            
            if terminated or truncated:
                break
                
    return total_reward

# ----------------- CMA-ES 训练主循环 -----------------
# 计算 C 模型一共有多少个参数 (288 * 3 + 3 = 867)
num_params = sum(p.numel() for p in Controller().parameters())

# 初始化 CMA 优化器 (初始均值为0，初始标准差为0.1)
es = cma.CMAEvolutionStrategy(num_params * [0], 0.1)

# 迭代训练 (假设跑 100 代)
for generation in range(100):
    # 1. 从当前参数分布中采样出一批新的权重组合 (种群)
    solutions = es.ask()
    
    # 2. 评估每个权重组合的得分 (实际中常使用 multiprocessing 并行加速)
    fitness_list = []
    for weights in solutions:
        # 进化算法默认是求极小值，所以我们把得分取负号
        reward = evaluate_controller(weights, env, v_model, m_model, controller)
        fitness_list.append(-reward) 
        
    # 3. 将得分反馈给 CMA-ES，更新其内部的高斯分布 (均值和协方差)
    es.tell(solutions, fitness_list)
    
    # 打印当前代的最佳表现
    print(f"Generation {generation}: Best Reward = {-np.min(fitness_list)}")

```

### 1.4 虚拟训练
#### 1.4.1 让M模型成为游戏引擎
在真实的强化学习交互中，循环是这样的：

- 真实引擎输出像素 -> V 模型降维成 $z_t$。
- C 模型根据 $z_t$ 和 $h_t$ 决定动作 $a_t$。
- 把 $a_t$ 喂给真实引擎，引擎计算物理碰撞，渲染出下一帧，并告诉你是否死亡（done）。

而在“梦境”交互中，真实引擎和 V 模型都被完全剥离了,梦境训练是C模型的考场：

- 假设一个初始的 $z_0$ 和 $h_0$。
- C 模型根据 $z_0$ 和 $h_0$ 决定动作 $a_0$。
- 把 $a_0$ 喂给 M 模型。M 模型利用 MDN 预测出下一个状态 $z_1$ 的高斯分布，我们从中采样得到具体的 $z_1$。

这里最关键的问题是：在梦里怎么知道游戏结束了？奖励如何计算？

为了支持梦中训练，M 模型需要做一点小小的升级。除了预测 $\pi, \mu, \sigma$，RNN 的输出还需要接入一个额外的全连接层，用来进行二分类预测（预测当前帧智能体是否死亡/游戏结束 done 的概率）。在 VizDoom 存活任务中，只要没死，每坚持一步就给 1 分，所以只要预测出 done，就能算出奖励。

#### 1.4.2 梦境循环
现在，我们要把上一节的evaluate_controller函数重写，这一次，代码里没有任何env.step(),一切都发生在张量运算中。

```python
import torch
import torch.distributions as D

def evaluate_controller_in_dream(weights, m_model, controller, max_steps=1000, temperature=1.0):
    """
    智能体完全在 M 模型的隐空间梦境中进行游戏
    """
    # 注入进化算法生成的权重
    torch.nn.utils.vector_to_parameters(torch.tensor(weights, dtype=torch.float32), controller.parameters())
    
    total_reward = 0
    # 1. 梦境的起点：随机生成一个初始隐状态，或者从真实数据的 VAE 输出中随机挑一个
    z = torch.randn(1, 1, 32)  # (batch=1, seq=1, latent_dim=32)
    hidden = None
    
    with torch.no_grad():
        for step in range(max_steps):
            h_c = hidden[0].squeeze(0) if hidden is not None else torch.zeros(256)
            
            # 2. 智能体在梦中做出决策
            action = controller(z.squeeze(0), h_c).unsqueeze(0) # 保持 shape: (1, 1, 3)
            
            # 3. M 模型推演梦境的下一步 (假设 m_model 已经加上了 done_logits 的输出)
            pi, mu, sigma, done_logits, hidden = m_model(z, action, hidden)
            
            # 4. 从高斯混合分布中采样出下一个画面 z_{t+1}
            # 引入温度系数 (Temperature) 控制梦境的多样性
            pi = pi / temperature
            pi = torch.softmax(pi, dim=-1)
            
            # 挑选激活的高斯分支
            mix_dist = D.Categorical(pi)
            component = mix_dist.sample() # shape: (1, 1)
            
            # 从选中的高斯分布中采样 z_{t+1}
            # 实际上 PyTorch 中可以通过重参数化直接构建混合分布采样，这里做逻辑演示
            z_next_mu = mu[0, 0, component.item(), :]
            z_next_sigma = sigma[0, 0, component.item(), :] * (temperature ** 0.5)
            z = torch.normal(mean=z_next_mu, std=z_next_sigma).view(1, 1, 32)
            
            # 5. 结算奖励与终止条件
            done_prob = torch.sigmoid(done_logits)
            # 根据预测的概率，掷骰子决定智能体在梦里是不是死了
            is_dead = torch.rand(1).item() < done_prob.item()
            
            if is_dead:
                break
            else:
                total_reward += 1  # 存活一步得一分
                
    return total_reward
```

#### 1.4.3梦境训练的致命缺陷与温度系数 ($\tau$)

在梦中训练会遇到一个非常有意思的现象：模型会作弊（对抗性利用）。

如果 M 模型拟合得不够完美，C 模型会在千万次的进化迭代中，精准地找到 M 模型的“物理漏洞”。比如在梦里，C 模型发现某一个极其诡异的动作序列，会让 M 模型推演出的下一个画面永远停留在“安全区域”，或者卡出 bug 导致 done_prob 永远为 0。最终智能体在梦里活了无数个世纪，拿了无限高分，但一放到真实的 VizDoom 引擎里，刚出生就被火球炸死了。

为了对抗这种“做白日梦”的作弊行为，温度系数（Temperature） 成了关键。通过在采样时调高温度（比如 $\tau = 1.15$），我们在梦境中强制注入了更多不确定性。火球的轨迹变得更加飘忽不定，环境变得比现实世界更加恶劣。如果智能体在这样疯狂的梦境中都能活下来，那么当它醒来面对真实环境时，就会显得游刃有余。

## 2.Dreamer系列
初代的World Models证明了梦中训练的可行性，但也暴露了两个致命的工程瓶颈：

- 长程预测崩溃：RNN 的误差会随着时间步累积，梦境时间一长，预测的画面就会变成一团模糊的噪点。
- 进化算法的极限：CMA-ES 虽然避开了梯度爆炸，但采样效率极低，且无法优化成百上千万参数的庞大策略网络。

Dreamer通过引入RSSM(Recurrent State Space Model，循环状态空间模型)和可微解析梯度 (Analytic Gradients) 彻底解决了这两个问题。

### 2.1 RSSM（循环状态空间模型）
RSSM 的核心思想是将世界的状态拆分为“确定性 (Deterministic)”和“随机性 (Stochastic)”两个独立的分支。

你可以把它想象成大脑的两种记忆机制：

- 确定性状态 ($h_t$)：由 RNN（通常是 GRU）维护。它只负责记忆那些符合物理规律、绝对可预测的历史信息（比如我的车速、赛道的走向）。它像一根钢铁主轴，贯穿整个时间序列，绝不被随机噪声污染。
- 随机性状态 ($z_t$)：由多层感知机 (MLP) 生成的高斯分布（或离散分布）。它专门负责吸收环境中的不可预测因素（比如路面的微小颠簸、风向的变化）。

在RSSM中，每一步的状态推演被严密地划分为三个子模型

- 确定性序列模型(Deterministic State Model)：更新确定性状态。基于上一步的确定性状态 $h_{t-1}$、上一步的随机状态 $z_{t-1}$ 和动作 $a_{t-1}$。

$$h_t = \text{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$$

- 先验动力学模型 (Prior / Transition Model - 梦境预测)：这是智能体在梦中预测未来的模型。它不看真实画面，只根据当前的确定性记忆 $h_t$，预测出一个随机状态的分布。

$$\hat{z}_t \sim p(\cdot \vert h_t)$$

- 后验表征模型 (Posterior / Representation Model - 现实校准)：这是智能体在训练时，结合了环境真实画面特征 $x_t$（V模型的输出）后，得出的包含全部信息的随机状态分布。它负责在现实中修正先验模型的猜测。

$$z_t \sim q(\cdot \vert h_t, x_t)$$

关于损失部分，除了要求解码器能从 $(h_t, z_t)$ 重构出原始画面和奖励外，RSSM 强迫先验分布 $p$ 去尽可能逼近后验分布 $q$。数学上就是最小化它们之间的 KL 散度：$D_{KL}(q \Vert p)$。
只要 $p$ 学得足够像 $q$，在脱离了真实画面的梦境中，$p$ 就能完美地代班，生成极其逼真的隐状态。

### 2.2 Actor-Critic 与可微梦境 (Differentiable Dream)
有了 RSSM 提供的稳定且高度准确的梦境，Dreamer 放弃了笨拙的 CMA-ES 进化算法，转而使用了 Actor-Critic (演员-评论家) 架构。

为什么RSSM可以使用RL?因为 RSSM 的推演过程（包括从高斯分布中采样 $z_t$ 的重参数化技巧）是全程可微的。

在 Dreamer 的梦中：

- Actor (策略网络) 根据当前状态输出动作。
- RSSM 推演下一步状态，并预测下一步能拿多少奖励 $r_t$。
- Critic (价值网络) 预估未来的长期总收益 $V(s)$。

更重要的是，由于这些神经网络是可微的，加上梦境采样的特性，Dreamer 的采样效率远高于传统强化学习。

### 2.3RSSM代码实现
在代码中，我们将使用 GRUCell 作为确定性状态的更新器。随机状态则使用普通的 MLP 输出均值和对数方差来构建高斯分布。

```Python
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self,action_dim=3,hidden_dim=256,state_dim=32,embed_dim=256):
        super(RSSM,self).__init__()
        self.hidden_dim = hidden_dim # h_t: 确定性状态维度 (e.g., GRU 隐藏层 256)
        self.state_dim = state_dim   # z_t: 随机状态维度 (e.g., 32 维高斯分布)

        # 1.Deterministic State Model(确定性模型)
        # 输入是 [z_{t-1}, a_{t-1}]，经过一个线性层融合特征，然后送入 GRU
        self.fc_gru_input = nn.Linear(state_dim + action_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # 2. Prior Model (先验模型 - 梦中预测未来)
        # 仅根据当前的 h_t 预测下一个 z_t 的分布 (输出均值和对数方差)
        self.prior_mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.prior_mu = nn.Linear(hidden_dim, state_dim)
        self.prior_logvar = nn.Linear(hidden_dim, state_dim)

        # 3. Posterior Model (后验模型 - 结合真实视觉特征校准)
        # 结合当前的 h_t 和环境真实画面提取出的特征 x_t
        self.post_mlp1 = nn.Linear(hidden_dim + embed_dim, hidden_dim)
        self.post_mu = nn.Linear(hidden_dim, state_dim)
        self.post_logvar = nn.Linear(hidden_dim, state_dim)

    def step_prior(self, prev_state, prev_action, prev_hidden):
        """
        推演一次梦境 (Prior Step)
        在梦境中，我们没有真实画面，纯靠网络推演
        """
        # 计算 GRU 的输入
        gru_in = F.relu(self.fc_gru_input(torch.cat([prev_state, prev_action], dim=-1)))
        
        # 1. 更新确定性状态 h_t
        next_hidden = self.gru_cell(gru_in, prev_hidden)
        
        # 2. 预测先验随机状态的分布参数
        prior_x = F.relu(self.prior_mlp1(next_hidden))
        prior_mu = self.prior_mu(prior_x)
        prior_logvar = self.prior_logvar(prior_x)
        
        # 3. 使用重参数化技巧采样 z_t (保证梯度贯通)
        std = torch.exp(0.5 * prior_logvar)
        eps = torch.randn_like(std)
        next_state = prior_mu + eps * std
        
        return next_state, next_hidden, prior_mu, prior_logvar

    def step_posterior(self, prev_state, prev_action, prev_hidden, obs_embed):
        """
        推演一次现实校准 (Posterior Step)
        在训练时使用，结合真实画面的特征 obs_embed (V 模型的输出)
        """
        # 1. 先用 Prior Model 走一遍 (更新 h_t 并拿到先验预测，为计算 KL loss 准备)
        _, next_hidden, prior_mu, prior_logvar = self.step_prior(prev_state, prev_action, prev_hidden)
        
        # 2. 结合真实画面特征进行后验预测
        post_x = torch.cat([next_hidden, obs_embed], dim=-1)
        post_x = F.relu(self.post_mlp1(post_x))
        post_mu = self.post_mu(post_x)
        post_logvar = self.post_logvar(post_x)
        
        # 3. 采样真正的后验状态 z_t
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std)
        next_state = post_mu + eps * std
        
        return next_state, next_hidden, post_mu, post_logvar, prior_mu, prior_logvar
```
## 3.引入注意力机制与大模型骨干网络
### 3.1 因果注意力机制
在RSSM中，确定性状态 $h_t$ 依赖 GRU 逐步递推：$h_t = \text{GRU}(h_{t-1}, z_{t-1}, a_{t-1})$。这种递归结构存在两个天然缺陷：
- 串行瓶颈与记忆衰减：即使有门控机制，当时间跨度达到数百步时，早期的关键信息（比如 200 帧前看到的一把钥匙或一个弯道标志）在不断的状态覆写中依然会被稀释。
- 训练无法完全并行：必须等 $t-1$ 步算完才能算 $t$ 步。

此时，解决方案利用的是LLM中常见的因果注意力机制 (Causal Self-Attention)。就像 GPT 预测下一个词一样，我们将过去所有的隐状态与动作序列 $[z_1, a_1, z_2, a_2, \dots, z_t, a_t]$ 全部排成一排，直接喂给一个带因果掩码 (Causal Mask) 的 Transformer Decoder。
- 全局直达感受野：注意力机制允许第 $t$ 步的 Query 直接去检索第 1 步的 Key，时间跨度上的信息衰减几乎为零。
- 并行推演：在训练阶段，给定一段长度为 $L$ 的真实轨迹，Transformer 可以通过一次前向传播，同时计算出所有时间步的先验预测分布。

### 3.2从CNN到ViT
传统的CNN编码器通过局部卷积核逐层下采样，容易丢失空间上的长距离几何关系（比如画面左侧的障碍物与右侧的出口之间的相对拓扑结构）。

引入 Vision Transformer (ViT) 后，一帧图像被切分为多个 Patch Tokens（例如 $64 \times 64$ 图像切分为 $8 \times 8$ 的 Patch，共 64 个 Token）。自注意力机制在第一层就能建立全局空间关联。更进一步，在现代世界模型（如 IRIS）中，ViT 配合离散化编码器（VQ-VAE），直接把一帧画面翻译成几十个离散的“视觉词汇”，让世界模型彻底变成了一个多模态语言模型。

这里补充一下LLM的的统一框架，由于LLM的大量训练数据来自于文本，文本语言的一大特征就是离散，有上一个token预测下一个token，单词、汉字都可以用字母表、偏旁的一个码本给编译出来。在早期的 VAE 或 RSSM 中，一帧画面被压缩成了一个连续的、单一的向量（比如 32 维高斯分布）。这种做法有两个局限：一是连续向量难以完美表达复杂多变的物体组合；二是它与 GPT 这种自回归语言模型的架构格格不入。IRIS等现代世界模型把游戏/物理世界变成一门“外语”，用训练 LLM 的方式来训练世界模型。这个过程分为两步：
- 首先，我们用VQ-VAE（向量量化变分自编码器）在架构上维护了一个可学习的离散嵌入表（codebook），里面存有比如$K = 512$ 或 $1024$ 个固定的特征向量，可以把它们理解为512 个基础视觉词汇（类似于汉字的偏旁部首，或者英语的 BPE 子词词表）。
- 当ViT编码器提取出一帧画面的 $4 \times 4 = 16$ 个空间特征向量后，VQ-VAE 不再保留这些连续浮点数。它去字典里做“最近邻查找（Nearest Neighbor Lookup）”：看看这 16 个特征向量分别和字典里的哪个词最像，然后直接用字典里的整数编号（ID）替换掉它们。由此一来，原本复杂的连续像素画面被彻底翻译成了遗传长度为16的离散整数序列。一帧画面被转化为了多个离散的视觉词汇。
- 一旦画面变成了离散的token，动作$a_t$ 也可以被离散化为一个动作 Token（例如 [Action_Left]），奖励 $r_t$ 也可以变成一个奖励 Token（例如 [Reward_+1]）。此时，智能体在世界中的一整段经历，在 Transformer 眼里就变成了一篇标准的多模态文章：$$\underbrace{z_0^{(1)}, z_0^{(2)}, \dots, z_0^{(16)}}_{\text{第0帧的16个视觉词}}, \quad \underbrace{a_0}_{\text{动作词}}, \quad \underbrace{r_0}_{\text{奖励词}}, \quad \underbrace{z_1^{(1)}, z_1^{(2)}, \dots, z_1^{(16)}}_{\text{第1帧的16个视觉词}}, \quad \underbrace{a_1}_{\text{动作词}} \dots$$

这样带来的好处是：
- 完全复用 LLM 生态：我们不需要再去推导复杂的高斯混合损失（MDN Loss）或 KL 散度平衡（KL Balancing）。训练世界模型的 Loss 直接退化成了大语言模型最纯粹的交叉熵损失（Cross-Entropy Loss）——给定前面所有的 Token，预测下一个 Token 是词表中的哪一个 ID。

- 梦境生成 = 自回归文本生成：在梦境中推演未来时，模型就像 ChatGPT 吐字一样，一个 Token 接一个 Token 地往外蹦：先吐出 16 个视觉 Token 拼出下一帧画面，再吐出 1 个奖励 Token 告诉自己这步得了几分，然后输入一个动作 Token，继续往下吐字。

- 天然的多模态扩展性：既然画面、动作、奖励全都是词表里的 Token，如果未来你想给智能体加入自然语言指令（比如“去拿红色的钥匙”），只需要把文本 Token 直接拼接在这串序列的最前面即可。世界模型就这样无缝演变成了一个具身多模态大模型（Embodied MLLM）。

多讲一个概念叫“码本坍缩”，比如字典里有 512 个词，但训练几个 Epoch 后，编码器偷懒，永远只映射其中固定的 10 个词，剩下 502 个词变成了“死词”，导致重构出来的画面丢失大量细节。

除了常规的承诺损失（Commitment Loss），现代架构（如 DreamerV3 或 VQGAN）通常会采用 EMA（指数移动平均）更新字典，或者直接使用 FSQ（Finite Scalar Quantization，有限标量量化）——它甚至不需要维护额外的字典和梯度直通估计（STE），直接把连续特征通过 tanh 限制范围后四舍五入取整，就能完美实现零坍缩的离散化。

### 3.3联合嵌入预测架构（JEPA）

传统VAE、Dreamer有个问题：他们都包含一个庞大的Decoder，强迫模型去重构每一个像素。但现实世界中充满了不可预测且毫无意义的细节。把模型容量和显存浪费在重建像素上，不仅低效，还容易让模型陷入局部噪声。

JEPA的核心思想就是在抽象的隐空间进行掩码预测，彻底砍掉像素解码器。具体而言：
- 输入完整的视频序列，通过目标编码器（Target Encoder，通常由上下文编码器通过指数移动平均 EMA 更新权重）提取出真实的未来特征 $s_y$。
- 对输入进行时空掩码（遮挡掉未来的帧或画面中的某些区域），上下文编码器（Context Encoder）提取可见特征 $s_x$。
- 预测器（Predictor，一个小型 Transformer）根据 $s_x$ 和动作/位置条件，直接在特征空间里预测被遮挡部分的表征 $\hat{s}_y$。
- Loss 计算：直接计算特征空间的距离 $\vert{}\vert{}\hat{s}_y - s_y\vert{}\vert{}^2$。因为预测目标已经是高度抽象的语义表征，树叶的随机晃动在编码阶段就已经被过滤掉了，模型被迫去学习真正的物理规律和物体运动轨迹。

### 3.4Causal Transformer世界模型代码
```python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalTransformerDynamics(nn.Module):
    def __init__(self, state_dim=32, action_dim=3, d_model=256, nhead=4, num_layers=4, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # 1. 特征映射层：将状态 z_t 和动作 a_t 映射到统一的 d_model 维度
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        
        # 2. 可学习的时间步位置编码 (Positional Embedding)
        # 注意：每个时间步包含 (z_t, a_t) 两个 token，因此最大序列长度为 2 * max_seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 * max_seq_len, d_model))
        
        # 3. 标准的 Transformer Encoder 层 (配合因果掩码即成为 Decoder-only 架构)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 结构在时序预测中训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 先验分布预测头 (Prior Head)：预测下一个状态 z_{t+1} 的高斯分布参数
        self.prior_mu = nn.Linear(d_model, state_dim)
        self.prior_logvar = nn.Linear(d_model, state_dim)

    def _generate_causal_mask(self, seq_len, device):
        """生成上三角因果掩码，确保 t 时刻只能看到 <= t 的信息"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def forward(self, states, actions):
        """
        states: (batch_size, T, state_dim) -> [z_0, z_1, ..., z_{T-1}]
        actions: (batch_size, T, action_dim) -> [a_0, a_1, ..., a_{T-1}]
        返回: 针对每个时间步预测的下一帧先验分布参数 prior_mu, prior_logvar，形状均为 (B, T, state_dim)
        """
        B, T, _ = states.shape
        device = states.device
        
        # 1. 独立嵌入
        s_emb = self.state_embed(states)   # (B, T, d_model)
        a_emb = self.action_embed(actions) # (B, T, d_model)
        
        # 2. 交错拼接序列: [z_0, a_0, z_1, a_1, ..., z_{T-1}, a_{T-1}]
        # 堆叠后形状为 (B, T, 2, d_model)，展平为 (B, 2*T, d_model)
        tokens = torch.stack([s_emb, a_emb], dim=2).view(B, 2 * T, self.d_model)
        
        # 3. 加上位置编码
        tokens = tokens + self.pos_embed[:, :2 * T, :]
        
        # 4. 生成因果掩码并前向传播
        causal_mask = self._generate_causal_mask(2 * T, device)
        out_tokens = self.transformer(tokens, mask=causal_mask)
        
        # 5. 提取动作 token 对应的输出位置 (索引 1, 3, 5, ..., 2*T-1)
        # 因为在看过 z_t 和 a_t 之后，模型应当预测 z_{t+1}
        action_out_tokens = out_tokens[:, 1::2, :] # (B, T, d_model)
        
        # 6. 映射为先验分布参数
        prior_mu = self.prior_mu(action_out_tokens)
        prior_logvar = self.prior_logvar(action_out_tokens)
        # 限制 logvar 范围防止数值爆炸
        prior_logvar = torch.clamp(prior_logvar, min=-10.0, max=2.0)
        
        return prior_mu, prior_logvar

# ----------------- 2080 Ti 显存与前向测试 -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerDynamics(state_dim=32, action_dim=3, d_model=256, nhead=4, num_layers=4).to(device)
    
    # 模拟一个 Batch=32, 长度 T=100 的序列
    dummy_states = torch.randn(32, 100, 32, device=device)
    dummy_actions = torch.randn(32, 100, 3, device=device)
    
    # 开启混合精度 (AMP) 以进一步节省 2080 Ti 显存并加速计算
    with torch.cuda.amp.autocast():
        mu, logvar = model(dummy_states, dummy_actions)
        
    print(f"输出先验均值形状: {mu.shape}") # 预期: (32, 100, 32)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

```
## 4.ViT模型接入
### 4.1 空间 Token 数量与时序计算复杂度
将ViT接入世界模型面临这一个核心矛盾就是“空间 Token 数量”与“时序计算复杂度”的博弈：
- Patchify（图像分块嵌入）与 2D 空间位置编码：
    - 对于输入画面 $x_t \in \mathbb{R}^{3 \times H \times W}$（例如 $64 \times 64 \times 3$），使用步长等于卷积核大小的二维卷积 Conv2d(3, D, kernel_size=P, stride=P)（如 $P=8$），在一次矩阵运算中完成图像切块与线性投影，直接得到 $N = (64/8)^2 = 64$ 个 Patch Tokens。
    - 加上可学习的 2D 空间位置编码 $E_{pos} \in \mathbb{R}^{1 \times 64 \times D}$，使模型明确知道每个 Patch 在画面中的绝对坐标。
- 空间特征聚合（从 $N$ 个 Patch 到单帧隐状态 $z_t$），如果直接把每帧的 64 个 Patch 全部塞进时序 Transformer，100 帧就会产生 $6400$ 个 Token，自注意力的 $O(L^2)$ 复杂度会瞬间打满 2080 Ti 显存。工程上有三种主流聚合方案：
    - 方案 A：[CLS] Token 聚合（最轻量）：在 64 个 Patch 前拼接 1 个可学习的 [CLS] Token。经过几层空间自注意力交互后，仅提取 [CLS] 位置的输出向量，通过线性层映射为高斯分布参数 $(\mu_t, \log\sigma_t^2)$。
    - 方案 B：可学习查询交叉注意力（Cross-Attention Pooling，如 Q-Former / Perceiver 思想）：用 $M$ 个（如 $M=4$ 或 $8$ 个）可学习的 Latent Queries 去交叉查询 64 个 Patch Tokens。这样每帧只需要保留 4 个高度浓缩的空间语义 Token（例如分别聚焦于“自车状态”、“前方赛道曲率”、“障碍物位置”、“背景边界”），在精度和显存之间取得最佳平衡。
    - 方案 C：空间保持 + 量化（IRIS / VQ-ViT 范式）：保留 $4 \times 4 = 16$ 个空间网格 Token，每个 Token 独立进行离散量化（VQ 或 FSQ），保留完整的二维空间拓扑结构。
- 预训练权重冻结 vs. 端到端微调
    - 在小规模环境（如 CarRacing、Atari）上，从头训练一个 3~4 层的轻量级 ViT 收敛极快；
    - 若面对真实复杂场景（如自动驾驶真实视频或机器人操作画面），最佳实践是加载预训练轻量 ViT（如 vit_tiny_patch16_224 或 DINOv2-Small）并冻结其前几层权重，仅微调最后一层 Transformer Block 和投影头（Projection Head），既能获得极强的零样本空间语义感知，又能将显存占用压在 3GB 以内。
### 4.2 ViT编码器接入世界模型的代码书写
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialViTEncoder(nn.Module):
    """
    轻量级空间 ViT 编码器：将单帧图像 (B, 3, 64, 64) 编码为连续隐状态 z_t (B, state_dim)
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, d_model=192, nhead=4, num_layers=3, state_dim=32):
        super().__init__()
        assert img_size % patch_size == 0, "图像尺寸必须能被 patch_size 整除"
        self.num_patches = (img_size // patch_size) ** 2  # (64 // 8)^2 = 64
        self.d_model = d_model

        # 1. Patch Embedding: 用 stride=patch_size 的卷积等价实现切块 + 线性投影
        self.patch_embed = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)

        # 2. 可学习的 [CLS] Token 与 2D 空间位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. 空间自注意力层 (Spatial Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 后验分布投影头：输出当前观测的后验分布参数 (mu, logvar)
        self.norm = nn.LayerNorm(d_model)
        self.fc_mu = nn.Linear(d_model, state_dim)
        self.fc_logvar = nn.Linear(d_model, state_dim)

    def forward(self, x):
        """
        x: 图像张量，形状为 (B * T, C, H, W)
        返回: 采样后的隐状态 z (B * T, state_dim)，以及后验分布参数 mu, logvar
        """
        N = x.shape[0]
        # (N, 3, 64, 64) -> (N, d_model, 8, 8) -> (N, 64, d_model)
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)

        # 拼接 [CLS] token -> (N, 65, d_model)
        cls_tokens = self.cls_token.expand(N, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1) + self.pos_embed

        # 空间全局注意力交互
        out_tokens = self.spatial_transformer(tokens)

        # 提取聚合了全局空间信息的 [CLS] token
        cls_out = self.norm(out_tokens[:, 0, :])

        # 计算后验高斯参数并重参数化采样
        mu = self.fc_mu(cls_out)
        logvar = torch.clamp(self.fc_logvar(cls_out), min=-10.0, max=2.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return z, mu, logvar


class ViTWorldModel(nn.Module):
    """
    端到端 ViT + Causal Transformer 世界模型
    """
    def __init__(self, state_dim=32, action_dim=3):
        super().__init__()
        self.state_dim = state_dim
        # 空间视觉感知模块 (ViT)
        self.encoder = SpatialViTEncoder(img_size=64, patch_size=8, d_model=192, nhead=4, num_layers=3, state_dim=state_dim)
        
        # 复用上一节的因果时序动力学模型 (此处用等价结构展示对接)
        self.state_embed = nn.Linear(state_dim, 256)
        self.action_embed = nn.Linear(action_dim, 256)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, 256))
        
        dyn_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, activation='gelu', batch_first=True, norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(dyn_layer, num_layers=4)
        self.prior_mu = nn.Linear(256, state_dim)
        self.prior_logvar = nn.Linear(256, state_dim)

    def forward(self, obs_seq, action_seq):
        """
        obs_seq: 视频帧序列 (B, T, 3, 64, 64)
        action_seq: 动作序列 (B, T, action_dim)
        """
        B, T, C, H, W = obs_seq.shape

        # 1. 将 Batch 和 Time 维度合并，并行通过 ViT 提取所有帧的隐状态
        flat_obs = obs_seq.view(B * T, C, H, W)
        z_flat, post_mu_flat, post_logvar_flat = self.encoder(flat_obs)

        # 恢复时序维度: (B, T, state_dim)
        z_seq = z_flat.view(B, T, self.state_dim)
        post_mu = post_mu_flat.view(B, T, self.state_dim)
        post_logvar = post_logvar_flat.view(B, T, self.state_dim)

        # 2. 构建状态-动作交错序列并输入时序 Transformer
        s_emb = self.state_embed(z_seq)
        a_emb = self.action_embed(action_seq)
        tokens = torch.stack([s_emb, a_emb], dim=2).view(B, 2 * T, 256)
        tokens = tokens + self.pos_embed[:, :2 * T, :]

        causal_mask = torch.triu(torch.ones(2 * T, 2 * T, device=obs_seq.device), diagonal=1).bool()
        dyn_out = self.temporal_transformer(tokens, mask=causal_mask)

        # 3. 提取动作位置输出，预测下一帧的先验分布 (t 时刻预测 t+1 时刻)
        action_out = dyn_out[:, 1::2, :]
        prior_mu = self.prior_mu(action_out)
        prior_logvar = torch.clamp(self.prior_logvar(action_out), min=-10.0, max=2.0)

        return z_seq, (post_mu, post_logvar), (prior_mu, prior_logvar)


# ----------------- 2080 Ti 显存与吞吐量实测 -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTWorldModel(state_dim=32, action_dim=3).to(device)

    # 模拟 Batch=16, 序列长度 T=64 的视频片段
    dummy_obs = torch.randn(16, 64, 3, 64, 64, device=device)
    dummy_act = torch.randn(16, 64, 3, device=device)

    with torch.cuda.amp.autocast():
        z_seq, (post_mu, post_logvar), (prior_mu, prior_logvar) = model(dummy_obs, dummy_act)
        # 计算时间对齐的 KL 损失：用 t 时刻预测的 prior 拟合 t+1 时刻真实观测的 posterior
        # prior[:, :-1] 对应对第 1..T-1 帧的预测；post[:, 1:] 对应第 1..T-1 帧的真实后验
        kl_loss = 0.5 * torch.sum(
            prior_logvar[:, :-1] - post_logvar[:, 1:] +
            (torch.exp(post_logvar[:, 1:]) + (post_mu[:, 1:] - prior_mu[:, :-1]).pow(2)) / torch.exp(prior_logvar[:, :-1]) - 1,
            dim=-1
        ).mean()

    print(f"ViT + 时序 Transformer 总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"单步时间对齐 KL Loss: {kl_loss.item():.4f}")
```


## 5 前沿探索域高阶研究
### 5.1 理论学习：离散化表征、扩散解码与隐空间几何
#### 5.1.1 世界 Token 化：为什么 DreamerV3 抛弃了连续高斯分布？
在之前的章节能看到，我们一直在假设隐状态 $z_t$ 服从连续的高斯分布 $\mathcal{N}(\mu, \sigma^2)$。但在复杂环境中，连续高斯分布存在两个硬伤：
- 多峰表达困难：如果前方路口既可能出现行人，又可能出现车辆，单峰高斯分布只能取两者的平均值，导致特征模糊；
- 梯度缩放敏感：当方差 $\sigma$ 趋近于 0 时，KL 散度 $\frac{1}{\sigma^2}$ 会产生剧烈的梯度尖峰，导致训练不稳定。

那DreamerV3如何解决这个问题的呢？DreamV3 将隐状态 $z_t$ 设计为 32 个独立的离散变量，每个变量有 32 个离散类别（即 $32 \times 32 = 1024$ 维的二进制独热矩阵）。
- 直通梯度估计（Straight-Through Estimator, STE）：在采样出离散的独热向量（One-Hot）后，反向传播时直接把梯度“直通”回采样前的 Softmax 概率层：z = one_hot + probs - probs.detach()。
- 有限标量量化（FSQ, Finite Scalar Quantization）：更现代的简化方案是直接把连续特征通过 tanh 压缩到 $[-1, 1]$，然后按固定步长（例如划分为 $L=5$ 个离散刻度）四舍五入取整，同样配合 STE 传递梯度。它无需维护庞大的 VQ 码本，彻底杜绝了“码本坍缩”。

#### 5.1.2 扩散模型（Diffusion）作为世界模型的“高保真渲染器”
在 IRIS、Genie 或 GameNGen 等生成式世界模型中，架构被明确拆分为“动力学推演（Dynamics）”与“像素渲染（Rendering）”两层：
- 动力学层（Transformer / RSSM）：只负责在极低维的离散或连续隐空间里做高速物理推演，计算下一帧的隐状态 $z_{t+1}$；
- 渲染层（Latent Diffusion Model）：将预测出的 $z_{t+1}$ 作为交叉注意力（Cross-Attention）或 AdaLN（自适应层归一化）的条件输入（Condition），指导一个轻量级 U-Net 或 DiT 从纯高斯噪声中去噪，渲染出清晰、无模糊的物理画面（解决传统 VAE 解码器画面糊成一片的通病）。

