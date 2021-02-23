# PPO Implementation for SEVN

This is a PyTorch implementation of
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:
```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

```

## Run the code
```
python3 main.py --env-name "SEVN-Test-AllObs-Shaped-v1" \
               --custom-gym SEVN_gym \
               --algo ppo \
               --use-gae \
               --lr 3e-4 \
               --clip-param 0.2 \
               --value-loss-coef 0.5 \
               --num-processes 10 \
               --num-steps 256 \
               --num-mini-batch 32 \
               --log-interval 1 \
               --use-linear-lr-decay \
               --save-after 200 \
               --save-multiple \
               --entropy-coef 0.00 \
               --seed 0 \
               --num-env-steps 3000000
```
## Reference
Please use this bibtex if you want to cite this repository in your publications:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
    }

