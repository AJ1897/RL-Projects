[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/3288)

The work contains RL implementation on the SEVN environment.

## Techniques Used
- Dueling Double Deep-Q-Network(D3QN)
- Proximal Policy Optimisation(PPO)

# SEVN: A Sidewalk Environment for Visual Navigation

SEVN contains around 5,000 full panoramic images and labels for house numbers, doors, and street name signs, which can be used for several different navigation tasks.
Agents trained with SEVN have access to variable-resolution images, visible text, and simulated GPS data to navigate the environment. 
The SEVN Simulator is OpenAI Gym-compatible to allow the use of state-of-the-art deep reinforcement learning algorithms. An instance of the simulator using low-resolution imagery can be run at 400-800 frames per second on a machine with 2 CPU cores and 2 GB of RAM.

Low Resolution Views (84x84px)             |  High Resolution Views (1280x1280px)
:-------------------------:|:-------------------------:
![game.png](imgs/low-res-viewer.png)  |  ![game.png](imgs/high-res-viewer.png)

![spatial_graph.png](imgs/spatial_graph.png)

4,988 panoramic images across 6.3 kilometers with 3,259 labels on house numbers, doors, and street name signs.

A longer introduction can be found here: [Creating a Navigation Assistant for the Visually Impaired](https://github.com/mweiss17/SEVN/blob/master/docs/01-article-env-introduction.md)


## Installation

In order to setup the environment, do something like the following. If using a fresh Ubuntu install, ensure that build-essential is installed (i.e., `sudo apt-get build-essential`). We'll need GCC for this, and that installs it.

```bash
# Install the code
git clone https://github.com/mweiss17/SEVN.git
cd SEVN

# Create a new conda environment for the depenencies
conda create -n sevn python=3.7

# Install the dependencies
conda activate sevn
pip install -e .

# Download the low resolution image data, labels, and spatial graph
python scripts/download.py

# Test that your environment is correctly setup
python scripts/01-play.py

# WARNING! Running this step downloads 28GB of image data and is not required to run the model or play with the environment.
# python scripts/download.py --high-res
# python scripts/01-play.py --high-res

```

## Dataset
You can manually download the dataset here (in case you don't want to follow the installation instructions above).
- [Metadata](https://zenodo.org/record/3521988#.Xbi0nnUzaV4)
- [Low resolution](https://zenodo.org/record/3521905#.XbhKu3UzaV4)
- [High resolution](https://zenodo.org/record/3526490/files/high-res-panos.zip) (Warning! 48 GB of images in a zip file)

### Dataset pre-processing
For more information about the data-preprocessing and the data format consult the `README` in the [SEVN-data](https://github.com/mweiss17/SEVN-data) github repository.


## Reference

```
@misc{weiss2019navigation,
    title={Navigation Agents for the Visually Impaired: A Sidewalk Simulator and Experiments},
    author={Martin Weiss and Simon Chamorro and Roger Girgis and Margaux Luck and
            Samira E. Kahou and Joseph P. Cohen and Derek Nowrouzezahrai and
            Doina Precup and Florian Golemo and Chris Pal},
    year={2019},
    eprint={1910.13249},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Built With
* [OpenAI Gym](https://github.com/openai/gym) - A toolkit for developing and comparing reinforcement learning algorithms


## License

This project is licensed under the MIT licence - please see the `LICENCE` file in the repository.
 
