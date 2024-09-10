# Adaptive-automation-for-DF

## Thoughts on designing DF events 

So far, we are trying to learn occupant comfort/discomfort depending on time of day and where you are in a home, what ativity you are doing or anticipated to do, and other larger(not "point in time") contextual considerations. 

But the aim of the study is to provide newer, innovative ways of providing occupant-centric grid interaction servicing from efficient buildings. Here, rather than focusing on determining when occupant decides to override the automations designed for the `proverbial` grid interaction service why not learn to modulate/adapt the grid interaction service based on the anticipated occupant preferences? 


### Question becomes how to predict the ocucpant preferences? 
Can we learn a local/personalized preference model based on a short-term mini-batch (eg. 24 hr ahead) context window and online machine learning approaches such as SGD/B (stoachastic gradient Descent/Boosting) that incrementally learns User preferences by minimizing the loss function 

- SGD - $h(x)$ has to be a logistic regeression model or a Support Vector Machine (SVM). Deifned by model parameter $\theta = \{w, b\}$, where $w$ denotes weight of the model and $b$ deontes the bias/intercept of the model. 

    starting at $0^{th}$ day as the model parameter $h_0(x) = 0$, at each subsequent day/update period... model updates based on the observed data during the previous day (previous batch of data) 
        - batch of data: pairs of input, output: $\{(x^i, y^i); i = n,...,m\}$ - used as training input for 
        updating the model error that updates model parameters

```math
 w_{d} \leftarrow w_{d-1} - \eta \nabla \left(\frac{1}{(m-n+1)} \sum_{i=n}^{m}L(h(x^i),y^i) \right)
```
```math
b_{d} \leftarrow b_{d-1} - \eta \nabla \left(\frac{1}{(m-n+1)} \sum_{i=n}^{m}L(h(x^i),y^i) \right)
```
; $\eta$: learning rate $\in [0,1], $(m-n+1)$ - mini-batch size, $L$ - loss function; in equations above log loss function entials logistic regression and SVM  $

$\textcolor{red}{\textsf{My questions to research more about:}}$
- In SGD, 
    - why log-loss function uses logical regression,
    - hinge-loss uses SVM classifier
- In SGB, 
    - has K coupled classification tree expansions, 
        - K - # of classification trees,
        - each tree has model partameters of 
        ```math
            \theta = \frac {\partial L(y^i, h^k_{d-1}(x^i))}{\partial h^k_{d-1}(x^i)}, 
        ```
        where the loss function is $h(x)$  - multinomial deviance
        constructed trees are used to update the $h(x)$ 

        ```math
        h^k_d (x) = h^k_{d -1} (x) + \eta T^k_d
        ```
# Iterative Global learning - Q-learning based RL Algo 

- each RL algo has all possible actions, all posisble states of the system, given the environment the system operates in. The Q-learning based RL algorithm has an associated q-value derived from a reward function value for each combination of state and action. This paper uses a custom fuction to calculate Q-value for each pair of state and action;
```math
Q(s_i, a_i) = r_{action}(s_i, a_j) + \gamma Max[Q(s_j, all\ possible\ actions)]; where 
```
$ \gamma \in [0, 1] $ - interpretation: value closer to 1 means algorithm will consider higher weights for rewards from future actions compared to value of 0. 

- note on rewards of a control policy; for a state $s_i$;  $r_{policy} (s_i) = w_{US}.r_{US}+ w_{AB}.r_{AB}; where$ US is user satisfaction; AB is achieved benefit from the automation 


formula for calculating the reward from the user satisfaction; three weighing factors/points to converts number of successful/unsuccessful events over a given duration 
