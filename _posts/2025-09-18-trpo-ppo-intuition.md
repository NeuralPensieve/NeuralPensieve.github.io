# The Art of Safe Policy Updates: From REINFORCE to TRPO and PPO

<div style="text-align: center; margin-bottom: 2em;">
    <img src="/img/trpo-ppo/cartoon.png" alt="" style="width: 100%;"><figurecaption>Policy gradient methods are like climbing a treacherous terrain to find the highest peak. Better use a harness like TRPO or PPO!</figurecaption>
</div>

**Proximal Policy Optimization (PPO)** has become the de facto algorithm for reinforcement learning across an astonishing range of applications. It powers the fine-tuning of large language models like ChatGPT, enables AI systems to master complex games like Dota 2 and StarCraft II, and controls robots learning to walk, manipulate objects, and navigate environments. Despite its ubiquity, many practitioners treat PPO as a black box, missing the elegant theoretical insights that make it so effective.

This article traces PPO's intellectual lineage through three pivotal algorithms: **REINFORCE** established the mathematical foundation for policy gradients but suffered from dangerous instability. **TRPO** solved the stability problem through principled constrained optimization but was complex to implement. **PPO** distilled TRPO's core insights into a simple, practical algorithm that has become the modern standard.

The story reveals a fundamental challenge in reinforcement learning: how do you ensure that each learning update improves performance without destroying what the agent has already learned? The evolution from REINFORCE to PPO shows how the field transformed an elegant mathematical insight into a reliable, practical tool.

***

## REINFORCE: The Foundation and Its Fatal Flaws

REINFORCE established the mathematical foundation that all modern policy gradient methods build upon. At its core, it seeks to improve an agent's **policy** ($\pi_\theta$)—the decision-making system that maps what the agent observes (**states** $s$) to what it should do (**actions** $a$).

The policy is controlled by tunable **parameters** ($\theta$)—think of these as the low-level settings that define the agent's behavior: how to weight different pieces of information, which actions to favor in specific situations, how aggressively to explore new strategies. The **objective** ($J(\theta)$) measures success through accumulated rewards. The goal is to adjust these parameters to maximize this score.

The fundamental challenge is that we can't calculate the perfect parameter settings directly. Each parameter only influences the *probability* of taking certain actions, and the agent's overall success depends on a complex sequence of these probabilistic decisions. This makes the connection between any single parameter change and the final outcome nearly impossible to compute analytically.

### The Policy Gradient Theorem

REINFORCE's breakthrough was the **policy gradient theorem**, which uses the clever **log-derivative trick** to transform this intractable problem into something we can estimate through sampling.

#### Deriving the Core Insight

Let's define our key terms:
- **$\tau$**: A complete **trajectory** from start to finish, like $(s_0, a_0, s_1, a_1, \ldots, s_{T-1}, a_{T-1})$
- **$R(\tau)$**: The **total reward** for trajectory $\tau$ 
- **$P(\tau; \theta)$**: The **probability** of trajectory $\tau$ under policy $\pi_\theta$

The derivation proceeds in four key steps:

**Step 1:** Start with our objective—the gradient of expected return:

$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \nabla_\theta \int P(\tau; \theta) R(\tau) d\tau$$

**Step 2:** Apply the log-derivative trick using the identity $\nabla_x f(x) = f(x) \nabla_x \log f(x)$:

$$\nabla_\theta J(\theta) = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau$$

**Step 3:** Recognize this as an expectation we can sample:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\nabla_\theta \log P(\tau; \theta) R(\tau)]$$

**Step 4:** Simplify using the fact that environment dynamics don't depend on policy parameters:

$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

This gives us the final **policy gradient theorem**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) R(\tau) \right]$$

This elegant result tells us we can estimate gradients by collecting trajectories and increasing the probability of actions that led to high rewards while decreasing the probability of those that led to poor outcomes.

### The Twin Problems: Inefficiency and Instability

While mathematically beautiful, REINFORCE suffers from two critical problems that make it nearly unusable in practice.

#### Problem 1: Sample Inefficiency

REINFORCE is **on-policy**, meaning it must collect fresh trajectories from the current policy for every update. This creates a vicious cycle of waste:

1. Collect expensive trajectories using current policy
2. Use them once to compute a gradient 
3. Update the policy, making all that data obsolete
4. Throw away the data and start over

For complex tasks, this means thousands of episodes just to learn each small improvement—clearly impractical for real-world applications.

#### Problem 2: The Death Spiral

Even worse than inefficiency is REINFORCE's tendency toward catastrophic failure. The algorithm measures progress in **parameter space** (how much $\theta$ changes) but what actually matters is **behavior space** (how much the agent's actions change). This mismatch creates dangerous scenarios:

- **Cliff edges**: Tiny parameter changes can cause dramatic behavioral shifts, making the agent forget successful strategies entirely
- **Flat plateaus**: Large parameter changes might produce almost no behavioral improvement
- **Death spirals**: Bad updates lead to poor data collection, which leads to worse updates, creating a downward spiral toward complete failure

REINFORCE has no mechanism to distinguish between these scenarios. It blindly applies whatever gradient it estimates, potentially taking an agent from competent performance to complete collapse in a single update.

This stability problem became the driving force behind the next generation of algorithms. The field needed a way to ensure that each update was not just an improvement in expectation, but a *safe* improvement that wouldn't destroy existing capabilities.

***

## TRPO: Constrained Optimization for Guaranteed Improvement

**Trust Region Policy Optimization (TRPO)** emerged as the principled solution to REINFORCE's stability crisis. Its key insight was revolutionary: instead of measuring update size in parameter space, measure it in **behavior space** and constrain changes to remain within a safe "trust region."

TRPO introduced two complementary innovations that work together to solve both the efficiency and stability problems:

### Innovation 1: Importance Sampling for Better Data Utilization

While TRPO remains fundamentally on-policy, it dramatically improves how each batch of data is used through **importance sampling**. This technique allows us to evaluate how a proposed new policy would perform using data collected from the old policy.

The mathematics behind importance sampling are straightforward. If our old policy $\pi_{\theta_{old}}$ took action $a$ in state $s$ with probability 0.3, and our proposed new policy $\pi_\theta$ would take the same action with probability 0.6, we can reweight that data point by the **importance sampling ratio**: $\frac{0.6}{0.3} = 2.0$. This tells us the new policy values that action twice as much as the old policy did.

This reweighting allows TRPO to make more confident, larger updates from each batch of data compared to REINFORCE's conservative approach. However, importance sampling only works when the old and new policies aren't too different—if they diverge significantly, the importance weights become unreliable and can lead to poor estimates.

### Innovation 2: Trust Region Constraints for Stability  

This limitation of importance sampling leads directly to TRPO's second innovation: constraining policy updates to remain within a "trust region" where importance sampling remains reliable.

TRPO formalizes this as a **constrained optimization problem**:

$$\text{maximize } \mathbb{E}_{s,a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}_{\theta_{old}}(s,a) \right]$$

$$\text{subject to } \mathbb{E}_{s \sim \pi_{\theta_{old}}} \left[ D_{KL}(\pi_{\theta_{old}}(\cdot|s) || \pi_\theta(\cdot|s)) \right] \leq \delta$$

Let's break this down:

- **The objective** uses per-step importance sampling ratios (not trajectory-level ratios, which would have explosive variance) multiplied by the **advantage function** $\hat{A}(s,a)$, which measures how much better action $a$ is compared to the average action in state *s*.

- **The constraint** defines our trust region using **KL divergence**—a measure of how different two probability distributions are. By keeping $D_{KL} \leq \delta$, we ensure the new and old policies remain similar enough for importance sampling to work reliably.

This elegant formulation guarantees that every update improves performance while maintaining stability—solving both of REINFORCE's core problems.

### The Mathematical Solution: Natural Gradients

Solving this constrained optimization problem directly would be computationally prohibitive. TRPO uses two key approximations to make it tractable:

1. **Linear objective approximation**: The surrogate objective is approximated using a first-order Taylor expansion: $g^T \Delta\theta$, where $g$ is the standard policy gradient.

2. **Quadratic constraint approximation**: The KL constraint is approximated using a more accurate second-order Taylor expansion involving the **Fisher Information Matrix (FIM)**, denoted $F$.

The Fisher Information Matrix captures the curvature of the policy's probability distributions—essentially encoding how sensitive the policy's behavior is to parameter changes in different directions. Areas where small parameter changes cause large behavioral shifts (cliff edges) have high curvature, while flat plateaus have low curvature.

The approximations above transform our problem into a simpler quadratic program:

$$\text{maximize } g^T \Delta\theta \quad$$

$$\text{subject to } \frac{1}{2}\Delta\theta^T F \Delta\theta \leq \delta$$

#### Deriving the Natural Gradient

Using **Lagrange multipliers** to solve this constrained optimization:

**Step 1:** Form the Lagrangian:
$$\mathcal{L}(\Delta\theta, \lambda) = g^T\Delta\theta - \lambda\left(\frac{1}{2}\Delta\theta^T F \Delta\theta - \delta\right)$$

**Step 2:** Take the gradient with respect to $\Delta\theta$:
$$\nabla_{\Delta\theta} \mathcal{L} = g - \lambda F \Delta\theta$$

**Step 3:** Set equal to zero for optimality:
$$g = \lambda F \Delta\theta$$

This is the heart of TRPO: it reveals that the optimal update direction is the **natural gradient**, which accounts for the geometry of the policy space through the Fisher Information Matrix. Instead of taking steps of uniform size in all parameter directions (like REINFORCE), TRPO takes smaller steps in sensitive directions and larger steps in stable directions.

### From Theory to Practice: The Complete TRPO Algorithm

The theoretical derivation gives us the update direction, but implementing TRPO requires three practical steps:

#### Step 1: Computing the Natural Gradient Direction

First, solve $F x = g$ for the natural gradient direction $x$. Since the Fisher Information Matrix can be enormous (millions by millions for modern neural networks), TRPO avoids computing $F$ explicitly. Instead, it uses the **Conjugate Gradient method**, which only requires computing matrix-vector products $Fx$ for various vectors $x$—something that can be done efficiently using automatic differentiation.

#### Step 2: Determining the Step Size

Once we have the direction $x$, we need to determine how far to step along it. Using the quadratic approximation of the constraint:

$$\alpha_{max} = \sqrt{\frac{2\delta}{x^T F x}}$$

This gives us the maximum step size that satisfies our trust region constraint under the quadratic approximation.

#### Step 3: Backtracking Line Search for Safety

The quadratic approximation is just that—an approximation. The final step is a **backtracking line search** that ensures the actual policy update is safe:

1. Start with the full calculated step size
2. Check if the true objective improves and the true KL constraint is satisfied  
3. If not, shrink the step size by a constant factor and try again
4. Repeat until both conditions are met

This process guarantees that the final update provides real improvement while staying within the trust region, preventing the catastrophic failures that plague REINFORCE.

***

## PPO: Simplifying the Complex

TRPO was a theoretical triumph that solved policy gradient learning's fundamental problems. But implementing it was a nightmare. The Fisher Information Matrix computations, Conjugate Gradient iterations, and backtracking line search created a complex system with numerous hyperparameters and potential numerical instabilities. Practitioners found themselves debugging TRPO's intricate machinery rather than solving their actual problems.

**Proximal Policy Optimization (PPO)** emerged as the practical solution that captures TRPO's essential insights through a remarkably simple approach. Instead of complex second-order methods and explicit constraints, PPO achieves similar stability through a clever **clipped objective function**.

### The Clipped Objective: TRPO's Essence Simplified

PPO replaces TRPO's constrained optimization with an unconstrained objective that implicitly limits policy updates:

$$L^{CLIP}(\theta) = \mathbb{E}_t[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t) ]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ is the importance sampling ratio.

This objective implements a brilliant heuristic: reward the policy for good actions and penalize it for bad ones, but only up to a reasonable limit defined by the clipping range $[1-\epsilon, 1+\epsilon]$.

#### Understanding the Clipping Mechanism

The *min* function creates different behaviors depending on whether an action was beneficial:

**For good actions** ($\hat{A}_t > 0$): We want to increase their probability, meaning we want $r_t > 1$.
- If the increase is modest ($r_t \leq 1+\epsilon$): The algorithm uses the normal objective $r_t \hat{A}_t$
- If the increase is excessive ($r_t > 1+\epsilon$): The objective plateaus at $(1+\epsilon)\hat{A}_t$, providing no additional incentive for larger changes

**For bad actions** ($\hat{A}_t < 0$): We want to decrease their probability, meaning we want $r_t < 1$.  
- If the decrease is modest ($r_t \geq 1-\epsilon$): The algorithm uses the normal objective $r_t \hat{A}_t$
- If the decrease is excessive ($r_t < 1-\epsilon$): The objective plateaus at $(1-\epsilon)\hat{A}_t$, preventing overly aggressive probability reductions

This clipping removes incentives for the importance sampling ratio to venture outside the safe range $[1-\epsilon, 1+\epsilon]$, achieving a similar effect to TRPO's KL constraint through elementary arithmetic operations.

### Why PPO Works: The Practical Magic

PPO's success stems from its ability to capture the essential insight behind TRPO—controlling behavioral change for stability—while being trivially simple to implement. The clipped objective:

- **Prevents catastrophic updates** by capping how much policies can change
- **Allows confident improvements** within the safe zone
- **Requires no second-order computations** or complex constraint handling
- **Is robust to hyperparameter choices** (typical $\epsilon$ values of 0.1-0.3 work well across diverse problems)

The result is an algorithm that practitioners can implement in a few dozen lines of code, tune with minimal effort, and deploy reliably across a wide range of applications.

## Algorithm Comparison: From Foundation to Practice

| Algorithm | Key Innovation | Strengths | Weaknesses |
|-----------|----------------|-----------|------------|
| **REINFORCE** | Policy gradient theorem | Mathematical foundation, conceptual clarity | Sample inefficient, catastrophically unstable |
| **TRPO** | Trust region constraints + importance sampling | Guaranteed monotonic improvement, principled approach | Complex implementation, computationally expensive |
| **PPO** | Clipped surrogate objective | TRPO's stability with simple implementation | No theoretical guarantees, heuristic approach |

***

## Conclusion: The Evolution of Stable Learning

The journey from REINFORCE to PPO illustrates a fundamental principle in machine learning: the most impactful advances often come from identifying core problems, developing principled solutions, and then finding practical ways to implement those insights.

REINFORCE established that we could learn policies through gradient ascent on expected rewards, but its instability made it unreliable for real applications. TRPO provided the mathematical framework showing that constraining behavioral change—not just parameter change—was the key to stability. PPO distilled this insight into a simple, practical form that anyone could implement and use successfully.

The evolution mirrors the broader trajectory of machine learning: from elegant mathematical insights that are difficult to implement in practice, to robust algorithms that capture the essential ideas while being simple enough to deploy widely. This is why PPO has become the workhorse for everything from training language models to mastering complex games—it reliably delivers results without the complexity overhead that made its predecessors difficult to use.

Today, when researchers and practitioners need to train agents for complex tasks, they reach for PPO not because it has the most elegant theory, but because it consistently works. Understanding this evolution doesn't just help you use these algorithms more effectively; it reveals the deeper principles that will guide the next generation of reinforcement learning methods. The core insight—that stability comes from controlling behavioral change rather than parameter change—remains as relevant today as it was when TRPO first introduced it.