# Epic Kitchens using TimesSformer 

 
## Description   
This project fine-tunes [TimesSformer](https://github.com/facebookresearch/TimeSformer) for a small subset of [Epic Kitchens](http://epic-kitchens.github.io) dataset.

## How to run   
First, install dependencies
```bash
# Make sure git lfs is installed and enabled
git lfs install

# clone project   
git clone https://github.com/fran6co/epic_kitchens

# install project   
cd epic_kitchens
conda env create -f environment.yml
conda activate epickitchens

 ```   
 Next, you can train the network
 ```bash
# download a [pretrained TimeSformer](https://www.dropbox.com/s/9v8hcm88b9tc6ff/TimeSformer_divST_8x32_224_HowTo100M.pyth?dl=0), it works without it as well but it will produce better results 
python project/video_action_classifier.py --data_path=data --batch_size=1 --max_epochs=50 --log_every_n_steps=13 --num_classes=2 --checkpoint=TimeSformer_divST_8x32_224_HowTo100M.pyth    
```
 Next, you can run the network on the validation videos
 ```bash
python tests/test_video_action_classifier.py --video_path=data/val/P01/P01_11 --onnx=lightning_logs/version_0/checkpoints/epoch=17-step=233.onnx --class_names_path=data/class_names.json
```

## Implementation details

The amount of training data is very small, which makes it really hard to not over-fit on the training data.

There are some techniques that can be used when working with a reduced dataset (there are more):
 - Transfer learning by fine-tuning a pretrained network
 - Freeze layers
 - Fine-tune the learning rate
 - Add/Increase augmentation
 - Generate synthetic data
 - Change loss function that penalizes local minima

In this work only the first 3 options have been explored with some degree of success.

To counter the over-fitting, the accuracy on the validation was used as a way to select the best model that would generalize the best with this dataset.
Ideally we would want the model to stabilize around a solution that is not an over-fit, but given the dataset and time constrains this technique was chosen.

The [pytorch lightning framework](https://www.pytorchlightning.ai/) was chosen as it offers a very concise way to define the training pipeline.

[TimeSformer](https://github.com/facebookresearch/TimeSformer) is a visual transformer applied to video action recognition, it produces good metrics on known datasets and wes easy enough to use.
A simple baseline of a 3D convolution network was considered, but it offers poor metrics and the main worry was that it would have a harder time to adapt to the reduced dataset.

[ONNXRuntime](https://onnxruntime.ai/) was chosen as the inference framework as it offers great flexibility and performance when running networks in production environments.

Conda was used to set the working environment as it makes it easy to create reproducible environments.

It's possible to generate a single prediction for the whole video by:
- taking the mean of the predictions over all the frames (the test code uses this)
- [Voting algorithm](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm)


### Experiments

The best results were achieved using the pretrained TimeSformer on HowTo100M dataset and setting the learning rate `0.0001` while not freezing any of the layers.

Freezing all layers but the head didn't produce good results, I think that freezing layers might get better if combined with a better way to find a good learning rate.

Here is the validation accuracy, the model spikes at some point bringing over the 60% accuracy. Same goes for the non-pretrained case, it just takes more epochs to get there.

#### 0.0001 lr/Pretrained/Non Frozen

![Good](assets/good.png?raw=true "Good")

#### 0.0001 lr/Non Frozen

![Good](assets/non-pretrained.png?raw=true "Good")

#### 0.0001 lr/Pretrained/Frozen

![Good](assets/frozen.png?raw=true "Good")

The results for the validation set are (using the mean and showing the rolling mean):

P01_11: open bag (0.65)

![P01_11](assets/P01_11.gif?raw=true "P01_11")

P01_13: take bag (0.53)

![P01_13](assets/P01_13.gif?raw=true "P01_11")

P03_32: take bag (0.54)

![P03_22](assets/P03_22.gif?raw=true "P01_11")

P11_24: open bag (0.60) (it starts as take_bag as it should be but then it does open_bag)

![P11_24](assets/P11_24.gif?raw=true "P01_11")