# irlc-vqa
Code for **[Interpretable Counting for Visual Question Answering](https://arxiv.org/pdf/1712.08697.pdf)** for ICLR 2018 reproducibility challenge.


## Results (without caption grounding)

| Model | Test Accuracy | Test RMSE | Training Time
| --- | --- | -- | -- |
| Reported Model | **56.1** | 2.45 | Unknown |
| Implemented Model | 54.4* | **2.40** | ~6 hours (Nvidia-1080 Ti) |

*= Still improving. Work in Progress. 

The accuracy was calculated using the [VQA evaluation metric](http://www.visualqa.org/evaluation.html).

RMSE = root mean sqared error from the ground truth.


## Key differences from the paper
- GRU was used instead of LSTM for generating question embeddings. Experiments with LSTM led to slower learning and more over-fitting. More hyper-parameter search is required to fix this.

- The hidden size of this implementation's scoring function is 1024 compared to their 2048. Using size larger than 1024 led to an over-fitting behaviour. More hyper-parameter search is required to find the right amount of regularization. This can lead to significant improvements in accuracy.

- Gated Tanh Unit is not used. Instead a 2-layer Leaky ReLu based network inspired by https://github.com/hengyuan-hu/bottom-up-attention-vqa with slight modifications is used.


## Filling in missing details in the paper

#### VQA Ground Truth
I couldn't find any annotations for a "single ground truth" which is requred to calculate the REINFORCE reward in IRLC. Also, I could not find any details in the paper relating to this issue. So I took as ground truth the label that was reported as the answer most number of times. In case there are more than one such label, the one having the least numerical value was picked (this might explain a lower RMSE).

#### Number of epochs
The authors mentioned that they use early stopping based on the development set accuracy but I couldn't find an exact method to determine when to stop. So I run the training for 100 epochs.

#### Number of candidate objects
I could not find the value of N = number of candidate objects that are taken from Faster-R-CNN so following https://github.com/hengyuan-hu/bottom-up-attention-vqa I took N=36. 

## Minor descrepancies

#### Number of images due to Visual Genome
From Table 1 in the paper it would seem that adding the extra data from Visual Genome doesn't change the number of training images (31932). However while writing the dataloaders for Visual Genome I noticed a around 45k images after including the visual genome dataset. This is not really a big issue, but I still thought I'd write it so that other people can avoid wasting their time investigating it.

## Implementation Details

- This implementation borrows most of its pre-processing and data loading code from https://github.com/hengyuan-hu/bottom-up-attention-vqa

- No real hyper-parameter search was performed. We use the same learning rate, learning rate schedule, loss weighing coefficients. The authors didn't mention the amount of dropout they used. After trying a few values the value 0.3 was chosen.


## Usage
#### Prerequisites
Make sure you are on a machine with a NVIDIA GPU and Python 3 with about 100 GB disk space. Python 2 might be required for running some scripts in ./tools (will try to fix this soon)

#### Installation
- Install PyTorch v0.4 with CUDA and Python 3.5.
- Install h5py.

#### Data Setup
All data should be downloaded to a 'data/' directory in the root directory of this repository.

The easiest way to download the data is to run the provided scripts `tools/download.sh` and then `tools/download_hmqa.sh` from the repository root. If the script does not work, it should be easy to examine the script and modify the steps outlined in it according to your needs. Then run `tools/process.sh` and `tools/process_hmqa.sh` from the repository root to process the data to the correct format. Some scripts in `tools/process.sh` might require Python2 (I am working on fixing this).

#### Training
Simply execute the cells in the IPython notebook `Training IRLC.ipynb` to start training. The development and testing scores will be printed every epoch. The model is saved every 10 epochs under the `saved_models` directory.


## Acknowledgements
The repository https://github.com/hengyuan-hu/bottom-up-attention-vqa was a huge help. It would have taken me a week at the least to write all code for pre-processing the data myself. A big thanks to the authors of this repository!
