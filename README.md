# Blind Image Denoising using Residual Dense Transformer Block Network (RDBNet)

The proposed method explores U-Net like structure that incorporates Residual Dense Blocks, Channel Attention Blocks, and Skip connections to handle blind image denoising. The performance of our proposed approach outperforms many state-of-the-art methods. 

# Visual Results for the proposed Method

![RDBNet removing noise from extremely large unseen noise level of 175 to 200](https://github.com/sander-ali/Variable_Image_Denoising_with_RDBNet/assets/26203136/911fd891-b56f-4e0f-8ff3-0728671f290e)

For usage, download the repository.

Install all the necessary packages mentioned in the Requirement.txt.

The link to download the pretrained model is mentioned in a text file available in the folder model_zoo. Place the download trained model to model_zoo folder. 

Copy the images that you want to denoise in the testsets --> Mytestset folder. 

Adjust the noise level in Line 42 and 45.

At the anaconda prompt, write the following:

python RDBNet_IE.py
