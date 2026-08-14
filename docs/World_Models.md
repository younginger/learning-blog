# <center>World Models</center>
# 1.奠基之作World Models（2018）
第一章我们将聚焦于 World Models (2018) 的奠基之作。整个世界模型由三个组件（V、M、C）构成，而第一步，我们需要为智能体装上“眼睛”——也就是 V 模型（Vision Model）。

由于高维像素空间（如游戏画面）对强化学习来说状态空间过于庞大且充满噪声，V 模型的核心任务是降维与特征提取：将复杂的图像压缩成低维的、连续的隐状态（Latent Vector），并在这个过程中过滤掉无关细节。

## 1.1 变分自编码器（VAE）
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
### 1.1.1 VAE代码编写
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

### 1.1.2 VAE训练
由于M模型（Memory）的输入并不是原始图像，而是V模块提取出来的隐向量序列$z_t$。如果 V 模块没有经过预训练，提取出来的 $z_t$ 就是毫无意义的随机噪声，M 模型也就无从学习环境的时序动态。

在世界模型的经典设定中，V模型的训练是无监督且独立于奖励信号的。以下是具体的操作逻辑和代码骨架：

### 1.1.3 使用随机策略收集数据
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
### 1.1.4 离线训练V模块
拿到 carracing_random_rollouts.npy 后，接下来的工作就回归到了标准的计算机视觉模型训练流程：

- 构建 torch.utils.data.DataLoader。
- 将图像送入我们在上一节写的 VAE 网络中。
- 计算重构损失（MSE）和 KL 散度，使用 Adam 优化器反向传播更新权重。
- 训练完成后，冻结 VAE 的权重。

### 1.1.5 为M模型准备时序数据
当V模块训练完毕后，我们还需要做一次数据转换。我们需要将之前收集到的（或者重新跑一波收集的）图像数据帧 $x_t$，通过训练好的 VAE 编码器，全部映射成隐向量 $z_t$。

此时，我们就把高维的视觉像素流，完全降维成了形状为 (Sequence_Length, 32) 的一维特征时间序列。这不仅大幅降低了对计算资源的消耗，也为处理复杂的时序依赖做好了完美的前置准备。

## 1.2 M模型 (Memory Model)
智能体能在多大程度上“理解”所处的环境，并对未来做出准确预判，全靠M模型。如果将V模型看作是感知视觉神经，那么M模块就是海马体--它不仅要记住过去，还要预测未来。

### 1.2.1 理论学习 - 混合密度循环网络 (MDN-RNN)
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

### 1.2.2 MDN-RNN代码部署
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

## 1.3 C模型（Controller 与 CMA-ES）
### 1.3.1 极简的C模型
V 模型将当前画面压缩成了隐变量 $z_t$（32 维），M 模型将历史序列总结成了隐状态 $h_t$（256 维）。对于 C 模型来说，它所需要的一切环境信息都已经在这两个向量里了。

因此，Ha & Schmidhuber 故意将 C 模型设计得极其简单：它仅仅是一个单层线性神经网络（无激活函数）。

其数学表达为：$$a_t = W_c [z_t, h_t] + b_c$$

其中，$[z_t, h_t]$ 是拼接后的特征向量（$32 + 256 = 288$ 维）。如果动作空间 $a_t$ 是 3 维（如赛车游戏中的转向、油门、刹车），那么权重矩阵 $W_c$ 的大小仅为 $3 \times 288$，偏置 $b_c$ 为 3 维。

这体现了表征学习的核心哲学：如果特征提取（V）和环境动力学建模（M）做得足够好，最终的决策（C）应该是非常线性且直接的。同时，极小的参数量（不到 1000 个参数）正是为了配合接下来的进化算法。

### 1.3.2 为什么采用进化算法而不是传统RL
在优化策略时，像 PPO 或 TRPO 这样的传统强化学习算法依赖于优势估计 (Advantage Estimation) 和基于梯度的反向传播。但在原始的 World Models 架构中，采用传统 RL 会面临难以逾越的工程障碍，而采用协方差矩阵适应进化策略 (CMA-ES) 却能完美破局。原因有三:
- 避开BPTT梯度爆炸：如果使用 PPO 等 Actor-Critic 方法，为了更新 Actor（这里的 C 模型），梯度需要从最终的奖励信号开始，穿过 C 模型，再沿着时间轴反向传播穿过长达数百步的 M 模型 (LSTM)，甚至穿过 V 模型。在这么长的序列上进行基于时间的截断反向传播 (BPTT)，极易导致梯度消失或爆炸，网络极难收敛。而相比之下，而 CMA-ES 是无梯度 (Gradient-free) 的。它完全把 C 模型的权重当作一个黑盒，只关心“这组权重打完一局游戏最终得分是多少”，彻底绕开了反向传播。
- 无需奖励塑性与价值网络:传统 RL 需要密集且即时的奖励信号来训练 Critic (价值网络) 从而评估优势。而在很多任务中（比如只在到达终点才给分的迷宫），奖励是极其稀疏的。CMA-ES 只需要一个评估指标（Cumulative Reward，即一局游戏的总得分），它不在乎你在第几步拿到了分，只看整体表现。
- 极佳的并行搜索能力：CMA-ES 维护了一个多维高斯分布来表示参数的搜索空间。每一代，它从这个分布中采样出多个不同的权重组合（种群），让它们同时去环境里跑游戏。然后根据得分，更新这个高斯分布的均值和协方差矩阵，让下一代朝着高分区域集中。这种天然的并行性非常适合在 CPU 集群上进行分布式评估。

### 1.3.3 Controller的代码实现
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

## 1.4 虚拟训练
### 1.4.1 让M模型成为游戏引擎
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

### 1.4.2 梦境循环
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

梦境训练的致命缺陷与温度系数 ($\tau$)在梦中训练会遇到一个非常有意思的现象：模型会作弊（对抗性利用）。

如果 M 模型拟合得不够完美，C 模型会在千万次的进化迭代中，精准地找到 M 模型的“物理漏洞”。比如在梦里，C 模型发现某一个极其诡异的动作序列，会让 M 模型推演出的下一个画面永远停留在“安全区域”，或者卡出 bug 导致 done_prob 永远为 0。最终智能体在梦里活了无数个世纪，拿了无限高分，但一放到真实的 VizDoom 引擎里，刚出生就被火球炸死了。

为了对抗这种“做白日梦”的作弊行为，温度系数（Temperature） 成了关键。通过在采样时调高温度（比如 $\tau = 1.15$），我们在梦境中强制注入了更多不确定性。火球的轨迹变得更加飘忽不定，环境变得比现实世界更加恶劣。如果智能体在这样疯狂的梦境中都能活下来，那么当它醒来面对真实环境时，就会显得游刃有余。

# 2.Dreamer系列
初代的World Models证明了梦中训练的可行性，但也暴露了两个致命的工程瓶颈：
- 长程预测崩溃：RNN 的误差会随着时间步累积，梦境时间一长，预测的画面就会变成一团模糊的噪点。
- 进化算法的极限：CMA-ES 虽然避开了梯度爆炸，但采样效率极低，且无法优化成百上千万参数的庞大策略网络。
Dreamer通过引入RSSM(Recurrent State Space Model，循环状态空间模型)和可微解析梯度 (Analytic Gradients) 彻底解决了这两个问题。

## 2.1 RSSM（循环状态空间模型）
RSSM 的核心思想是将世界的状态拆分为“确定性 (Deterministic)”和“随机性 (Stochastic)”两个独立的分支。

你可以把它想象成大脑的两种记忆机制：
- 确定性状态 ($h_t$)：由 RNN（通常是 GRU）维护。它只负责记忆那些符合物理规律、绝对可预测的历史信息（比如我的车速、赛道的走向）。它像一根钢铁主轴，贯穿整个时间序列，绝不被随机噪声污染。
- 随机性状态 ($z_t$)：由多层感知机 (MLP) 生成的高斯分布（或离散分布）。它专门负责吸收环境中的不可预测因素（比如路面的微小颠簸、风向的变化）。

在RSSM中，每一步的状态推演被严密地划分为三个子模型