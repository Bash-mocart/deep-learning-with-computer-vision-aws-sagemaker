# deep-learning-with-computer-vision-aws-sagemaker
The goal of this project was to use AWS Sagemaker to finetune a pretrained model that can perform Image Classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML Engineering Practices.

This project takes a pretrained CNN model (Resnet18) and finetunes it for use in classifying dog breeds into 133 classifications based on a Dog Breeds dataset.

The steps that the notebook goes through are:

1. Retrieving a dataset
1. Uncompressing that dataset.
1. Uploading that dataset to an S3 bucket.
1. Setting up hyperparameter tuning using learning rate, weight decay, eps and batch size using the AdamW optimizer on ResNet18.
1. Starting a hyperparameter tuning job using 20 training jobs (2 at a time) with autoshutdown on the AWS SageMaker Hyperparameter Tuning system.
1. Record the best hyperparameters as discovered from the above.
1. Train a fine-tuned model using ResNet18 and the best hyperparameters over a larger number of epochs, recording profiling and debug data.
1. Examine the output from the profiling and debug of the above.
1. Re-run the training with profiling and debug turned off (due to issues deploying model trained with the above).
1. Deploy that model as an endpoint on AWS.
1. Test that endpoint with a new data.

### Project Set Up and Installation
 Instructions:
 - Enter AWS through the gateway in the course and open SageMaker Studio.
 - Download the starter files.
 - Download/Make the dataset available.

I used Pytorch deep learning framework in this project to build a CNN to classify images of dogs according to their breed. First, I installed packages, like torch and smdebug, and imported libraries including sagemaker, numpy, boto3, os, etc.

### Dataset
I used the dog breed classification dataset provided by Udacity in the project. The dataset contains 8351 color images of dogs of 133 differnt breeds grouped into train, test and validation subsets in proportion 80:10:10 %. The images have different shapes which i standardized by using transforms library. Several transforms including normalization are stacked in transforms.Compose.

#### Access
The data were uploaded to an S3 bucket through the AWS Gateway, so that SageMaker can have access to the data. 

## Hyperparameter Tuning
In order to obtain optimal accuracy values SageMaker provides utility for tuning different hyperparameters of the Pretrained ResNet-50 model. I finetuned the model using the following hyperparameter ranges:
```
hyperparameter_ranges = {"lr": sagemaker.tuner.ContinuousParameter(1e-4, 1e-1),
    "weight-decay": sagemaker.tuner.ContinuousParameter(1e-3, 1e-1),
    "eps": sagemaker.tuner.ContinuousParameter(1e-9, 1e-7),
    "batch-size": sagemaker.tuner.CategoricalParameter([32, 64])}
```
![training_hpo1](https://github.com/Bash-mocart/deep-learning-with-computer-vision-aws-sagemaker/blob/main/hyperparameterranges.PNG)

![training_hpo2](https://github.com/Bash-mocart/deep-learning-with-computer-vision-aws-sagemaker/blob/main/variables.PNG)


As shown, the training objective was to minimize the average loss with Cost Entropy Loss as the cost function.


The following screenshots show completed training jobs, and the hyperparameter values obtained from the best tuning job.

![training_hpo3](https://github.com/Bash-mocart/deep-learning-with-computer-vision-aws-sagemaker/blob/main/besttrainingjob.PNG)


![training_hpo4](https://github.com/Bash-mocart/deep-learning-with-computer-vision-aws-sagemaker/blob/main/trainingjobs.PNG)


![training_hpo5](https://github.com/Bash-mocart/deep-learning-with-computer-vision-aws-sagemaker/blob/main/besttrainingjobhp.PNG)



## Debugging and Profiling
After obtaining the best hyperparameters I created a new model where I used SageMaker debugging and profiling utilities in order to track and detect potential issues with the model. In train_model.py script  I added SMDebug hooks for PyTorch with the TRAIN and EVAL modes to the train and test functions correspondingly. In the main function I created SMDebug hook and register it to the model. I also passed the hook as an argument to the train and test functions.
I also configured Debugger rules and hook parameters in the train_and_deploy notebook. With the help of SageMaker profiler instance resources like CPU and GPU memory utilization can be tracked. I then added Profiler and debugger configuration to estimator to train the model.

### Results
I found some issues were with PoorWeightInitialization, and LowGPUUtilization during training the model. Some CPU bottlenecks encountered. Rules summary was provided in the profiling report. 

![training_hpo5](https://github.com/Bash-mocart/deep-learning-with-computer-vision-aws-sagemaker/blob/main/output.PNG)



### Model Deployment
Originally the model was deployed as intended (by calling deploy on the Estimator), however querying the created endpoint resulted in the error presented on the following screenshot. I could not obtain any predictions on the inference because of the error on the screenshot below. 

![error](https://user-images.githubusercontent.com/54789219/146488759-a9bd498b-9a97-40e4-9518-e6380af1be1f.JPG)


The new inference handler script was adapted from [here]( https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html ), where the model_fn, input_fn and predict_fn added functionality for loading the model, deserializing the input data so that it could be passed to the model, and for getting predictions from the function. The code is implemented in test.py script.
The prediction was generated for one of the custom images using SageMaker runtime API with the invoke_endpoint method.

The screenshot below shows the active endpoint for the model.

![endpoints](https://github.com/Bash-mocart/deep-learning-with-computer-vision-aws-sagemaker/blob/main/endpoint.PNG)

