# How to Tame Your Deep RL

<div style="text-align: center; margin-bottom: 2em;">
    <img src="/img/tame_rl/rl.jpg" alt="" style="width: 75%;">
</div>

Running reinforcement learning (RL) experiments—especially when implementing models yourself—is notoriously more challenging than typical supervised learning (SL) experiments (as attested by none other than Andrej Karpathy [here](https://news.ycombinator.com/item?id=13519044)). SL models are generally well-behaved: they converge predictably, respond well to cross-validation, and scale efficiently on GPUs. Deep RL models, however, are entirely different beasts...

Note that this is NOT a post about how to learn deep RL. The assumption is that the reader is familiar with the theory of deep RL, and they want to start getting their hands dirty, and experiment with training deep RL models on their own.

## Why Deep RL is So Challenging

### Counterintuitive Behavior

Many intuitions from supervised learning simply don't transfer to RL. In SL, we expect loss to decrease steadily—if it rises, something's wrong. But in RL, your actor or critic loss might increase while your model's overall performance improves. I've witnessed this phenomenon countless times, though I still don't fully understand why it happens.

### Technical Complexity

Deep RL models present several unique challenges:

- **GPU utilization is tricky**: Unlike SL models, RL algorithms don't easily benefit from GPU acceleration
- **Complex infrastructure**: You're juggling environment data collection, replay buffers, and multiple models training simultaneously
- **No cross-validation**: Without this standard validation approach, it's much harder to assess whether your model is truly working
- **Scaling paradox**: Larger models often perform worse, not better—making GPU usage even less beneficial
- **Standard techniques fail**: Dropout and batch normalization, staples of deep learning, may actually hurt performance

### Different Optimization Landscape

In supervised learning, most hyperparameters relate to model architecture. In deep RL, architecture is often the least of your concerns—the critical hyperparameters lie in the algorithm itself, environment interaction, and training procedures.

### Reproducibility Challenges

Even with published results, success isn't guaranteed. I know researchers who've made humanoid model in OpenAi Gym stand using PPO, but I've never replicated that success. Similarly, SAC has worked exceptionally well in my experience, while PPO has consistently underperformed—despite following the same implementation principles.

## The Learning Curve

Beyond experimental challenges, mastering DRL requires a fundamentally different mindset and skill set compared to traditional machine learning.

### Advanced Mathematical Foundation
The mathematical machinery of RL goes significantly deeper than standard ML. While supervised learning primarily deals with optimization and statistics, RL draws from multiple advanced mathematical areas including dynamic programming, stochastic processes, optimal control theory, and even game theory for multi-agent scenarios. The mathematics of policy gradients, for instance, involves taking derivatives of expectations—a concept that doesn't exist in standard ML.

### Extensive Study Requirements
RL's breadth is staggering compared to other ML subfields. You need to understand not just individual algorithms, but entire families—model-free vs. model-based, on-policy vs. off-policy, value-based vs. policy-based. Each family has different assumptions, trade-offs, and suitable application domains. The field also builds on decades of research in control theory, operations research, and cognitive science, requiring familiarity with classical methods to understand modern deep RL approaches.

### Conceptual Difficulty
RL involves fundamentally different challenges than supervised learning. The temporal nature of decision-making, the exploration-exploitation trade-off, and the non-stationary learning environment (where your improving policy changes the data distribution) create conceptual hurdles that even experienced ML practitioners find challenging. These aren't just technical details—they represent different ways of thinking about learning itself.

## Getting Started: A Strategic Approach

### Choose Your Foundation Wisely

Start with a verified implementation like [**cleanrl**](https://github.com/vwxyzjn/cleanrl). While frameworks like [**torchrl**](https://docs.pytorch.org/rl/stable/index.html) exist, they're too high-level for learning fundamentals. **Never** begin by implementing algorithms from scratch—this [blog post](https://andyljones.com/posts/rl-debugging.html) explains why convincingly.

I recommend a middle path: the [CS285 course](https://rail.eecs.berkeley.edu/deeprlcourse/) homeworks from Berkeley (Sergey Levine). The course is excellent (available on YouTube and GitHub), and the assignments are perfectly structured—you implement the algorithm's core logic with helpful guidance, without worrying about infrastructure. This approach gave me deep understanding of the nuts and bolts.

### Build Your Expertise Gradually

Once you're comfortable with multiple algorithms and have a flexible codebase, implement additional algorithms yourself. Read the papers, but critically, check online implementations—papers often omit crucial implementation details that can make or break your results.

## Practical Tips for Success

### Model Training Best Practices

**Visualize your progress**: Record videos of your agent at regular intervals (not too frequently—it slows training). This visual feedback is invaluable for understanding what's actually happening.

**The golden rule**: Never change two things simultaneously. This seems obvious but happens constantly. You might think you're testing feature A while also tweaking "minor" feature B. When you see dramatic changes, you'll naturally credit feature A—only to discover later that feature B was the real driver. Version control and extensive logging can save you from this trap.

**Use version control religiously**: I learned this the hard way. Now I commit every feature change, no matter how small.

**Start simple**: Use environments that train quickly and work reliably. Begin with inverted pendulum, not humanoid! Start with OpenAI Gym's simple environments—avoid Atari games or image-based tasks initially, as they take forever to train.

**Progress strategically**: After validating your algorithm on simple environments, graduate to medium-complexity tasks (Hopper, HalfCheetah) before tackling hard problems (Humanoid). This progression helps you distinguish algorithm issues from environment complexity.

**Know your target**: Research benchmarks and watch success videos online. You need to know what good performance looks like. [This repo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md) is a great source, as it reports on typical rewards and number of steps for various environments.

**Scale up gradually**: Eventually you'll need cloud services or more powerful machines for sophisticated models and hyperparameter tuning. But you can accomplish a lot with simpler environments first—I've run everything on my MacBook Air! It's not fast, but it works for overnight and weekend runs.

### Logging Strategy

**Keep a research journal**: Document what worked and what didn't. Take screenshots of your experiments. Your future self will thank you.

**Track comprehensive metrics**:

- Average return
- Action entropy
- Actor and critic losses
- Average episode length
- Learning rate changes

and many more...

**Use config files**: Never hard-code hyperparameters. Create config files and log them meticulously.

**Use proper tools**: Tensorboard works, but Weights & Biases is superior:

- Monitor experiments remotely (even from your phone)
- Automatic code uploads
- Config logging with tabular comparison across runs (once helped me catch a subtle bug)
- Seamless Tensorboard integration
- No local memory usage for server hosting

### Hyperparameter Optimization

Hyperparameter selection is arguably the most challenging aspect of RL experimentation. Each run is time-intensive, you're dealing with dozens of parameters, and natural variability makes it hard to distinguish real improvements from noise.

**Control randomness**: Set seeds for numpy, torch, and all relevant libraries. Run experiments with identical seeds to measure natural variability, then change only the seed to understand your model's consistency.

**Learning rate matters**: Use appropriate learning rates and consider LR scheduling. High rates cause divergence; low rates slow learning dramatically.

**Batch size is critical**: Some RL methods require very large batch sizes to train. PPO, for example, shows highly variable minibatch size requirements across different tasks, ranging from 64 to 4096 samples or more. This happens because policy gradient methods need sufficient data to get stable gradient estimates. The data in PPO is also correlated (all coming from the same policy rollout), which requires larger batches to reduce variance.

**Understand your timeline**: Know how many steps are needed for success through benchmarks or literature. If success requires 1M steps but you're only running 100K, your conclusions will be meaningless.

**Don't set and forget**: Understand each hyperparameter's impact. I used to blindly set gamma (discount factor) to 0.99 until a serendipitous mistake—caught through extensive logging and Weights & Biases—revealed its dramatic impact on critic loss and result variability in Policy Gradient methods.

## Compute requirement

I started training models on my Macbook Air M1 (8GB of RAM), and it did well for environments in OpenAI Gym. As an example, I ran SAC algorithm on Humanoid env for 750k steps took 2h15m. GPU was not needed for this task, so it ran on CPU only. However, training Atari games would take more than 24 hours, so it was time to upgrade.

I built my own budget AI desktop with an 8-core AMD CPU, 32 GB of RAM, and a Nvidia RTX3060 GPU (for under $1000), and I could train things much faster. As a comparison, SAC ran 7 and 30 times faster on the PC on CPU only and CPU+GPU, respectively. PPO ran 3 and 6 times faster on the PC on CPU only and CPU+GPU, respectively. SAC gains a lot more because of larger batch sizes, I imagine.

Of course, you can use cloud computing, and it is probably the way to go for most people, but I personally wanted to build my own machine.

## Final Thoughts

Deep RL experimentation is challenging but rewarding. The key is to approach it systematically: start simple, log everything, change one thing at a time, and build your intuition gradually. The field is complex enough without adding unnecessary complications through poor experimental practices.

For additional perspectives from more experienced practitioners, check out these excellent resources [1](https://andyljones.com/posts/rl-debugging.html), [2](https://www.alexirpan.com/2018/02/14/rl-hard.html), [3](https://amid.fish/reproducing-deep-rl), [4](https://spinningup.openai.com/en/latest/spinningup/spinningup.html). Their retrospectives on RL experimentation provide valuable insights that complement the practical tips shared here.