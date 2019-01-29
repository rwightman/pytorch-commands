# Kaggle Speech Recognition Challenge (PyTorch)

## About
Getting frustrated with the lack of progress using my Tensorflow based codebase (https://github.com/rwightman/tensorflow-speech_commands), I threw together a PyTorch solution in the last week of the competition. Based on some previous image competitions, my DPN ported models, and adapted ResNet models from Torchvision, I built a dataset for the competiton data from the ground up with Rosa based augmentation I had been working on when I abandoned TF.

The DPN 92, 98 and Resnet 50, 101 models were scoring 0.85-0.87 single model on the competition LB and over 0.89 when ensembled across folds. Some basic boostinf of the silence class was done to achieve that.

In the final days I went back to learning to see if I could get metric learning via Triplet Loss to work and help solve issues with the silence and unknown class. The Triplet Loss did not converge (or converged to nothing of use). I used https://github.com/VisualComputingInstitute/triplet-reid and https://github.com/Cysu/open-reid/tree/master/reid as a reference.

2019-01-28: I updated the code to PyTorch 1.0 compatibility and revisited the triplet loss (needing to use it in another project). I discovered that sampling the positive and negative examples instead of always picking hardest positive and negative example avoids the collapse and results in reasonable convergence. A bug was also fixed in the PK batch sampler that builds batches by randomly sampling K samples from each of P classes  (batch size = P*K).

## Usage

Usage is straightforward, run `train.py` script and pass folder to your data training directory as the first arg. The defaults will train a resnet18 model with mel-spectrogram inputs. Other arguments can be displayed with `-h` or by inspecting code.

`python train.py /data/commands/train/`

Triplet training was split (rather redundantly) into its own training script for experimentation.

`python train_triplet.py /path/to/train/folder`