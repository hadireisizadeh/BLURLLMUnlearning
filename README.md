# BLUR-A-Bi-Level-Optimization-Approach-for-LLM-Unlearning


## Abstract

Enabling large language models (LLMs) to unlearn knowledge and capabilities acquired during training has proven vital for ensuring compliance with data regulations and promoting ethical practices in generative AI. Although there are growing interests in developing various unlearning algorithms, it remains unclear how to best formulate the unlearning problem. The most popular formulation uses a weighted sum of forget and retain loss, but it often leads to performance degradation due to the inherent trade-off between forget and retain losses. In this work, we argue that it is important to model the hierarchical structure of the unlearning problem, where the forget problem (which *unlearns* certain knowledge and/or capabilities) takes priority over the retain problem (which preserves model utility). This hierarchical structure naturally leads to a bi-level optimization formulation where the lower-level objective focuses on minimizing the forget loss, while the upper-level objective aims to maintain the model's utility. Based on this new formulation, we propose a novel algorithm, termed Bi-Level UnleaRning ($\texttt{BLUR}$), which not only possesses strong theoretical guarantees but more importantly, delivers superior performance. In particular, our extensive experiments demonstrate that $\texttt{BLUR}$ consistently outperforms all the state-of-the-art algorithms across various unlearning tasks, models, and metrics.

## Installation
### MUSE
You can install the required dependencies using the following command
```
conda env create -f environment.yml
conda activate muse_env
```

### WMDP
```
conda create -n wmdp_env python=3.9.21 -y
conda activate wmdp_env
conda install pytorch=2.1.1 torchvision=0.16.1 torchaudio=2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install datasets==3.2.0 wandb==0.19.2 transformers==4.37.2 sentencepiece==0.1.99 sentence-transformers==2.5.1
pip install terminaltables==3.1.10 sacrebleu==2.4.0 rouge-score==0.1.2 matplotlib==3.8.3 seaborn==0.13.2 scikit-learn==1.4.0
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

