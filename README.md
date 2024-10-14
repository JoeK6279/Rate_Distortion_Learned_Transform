# Rate-Distortion-Learned-Transform
The source code for "Learning Optimal Linear Block Transform by Rate Distortion Minimization".
The learned matrices can be found in the folder "TrainedMatrices".

## Install
```bash
git clone https://github.com/JoeK6279/Rate_Distortion_Learned_Transform
pip install torch compressai mat73
```

## Example
Set up the path to datasets and output models in the `train.py`.
```bash
python train.py 
```

### Arguments
`-b`: Block size (8,16,32)

`-lambda_max`, `-lambda_min`: maximum and minumum lambda values for variable-rate trianing

`-lr`: learning rate

`-exp_name`: name of the experiment (for file saving)


## Ackonwledgement
We modified the implementaion of Gaussian entropy model from [CompressAI](https://github.com/InterDigitalInc/CompressAI); we thank the authors for releasing the code.