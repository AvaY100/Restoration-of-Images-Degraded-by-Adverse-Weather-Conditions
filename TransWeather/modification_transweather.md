Original code repo from paper Transweather: https://github.com/jeya-maria-jose/TransWeather
Original dataset: https://drive.google.com/file/d/1tfeBnjZX1wIhIFPl6HOzzOKOyo0GdGHl/view

##Baseline
For baseline, we use the code from the original transweather repo. The code from the repo needs a few changes to be able to run. We made the following changes to the code:
1. We added a utils.py that was missing from the original repo. 
2. The dataset has to be downloaded from the link above. The dataset contains a total of 17069 input gt picture pairs. The dataset has to manually be split to training and validation set, formulated into a structure of data/train/input, data/train/gt, data/test/input, data/test/gt, and put a txt file containing the file path of all pictures contained in the directory for train and test folder respectively. We used 9:1 for training:validation split, which is 16263 input-gt picture pairs for training, and 1806 input-gt picture pairs for validation.
3. The command to run is python train.py -train_batch_size 32 -exp_name try1. On V100 machine on AWS, the time and cost for training 200 epoch on the full dataset is 26 hours and roughly $80.

##Implicit Depth Experiment:
For the first experiment, we added the depth information to the input image in an implicit way - by directly adding the depth information of input picture as a fourth channel. 
The depth information is obtained by using a depth estimation model Marigold(https://github.com/prs-eth/Marigold). We first run the Marigold model on all training input pictures Matigold is a diffusion based model and is very slow, takes about 18 hours to process the 16263 pictures on V100 machine on AWS. 
After getting the depth picture of the input images, we preprocess all the input pictures by adding a channel as the fourth channel. The depth information is added to the input image in the dataloader. The model architecture is modified to take in 4 channels instead of 3. The model is trained with the same hyperparameters as the baseline. The command to run is python train.py -train_batch_size 32 -exp_name try2.
Modification1: 
In model TransWeather/transweather_model.py, change model class definition init in_chan parameter from 3 to 4.
""""""
class EncoderTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=4, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
""""""
Modification2: 
In TransWeather/val_data_functions.py and TransWeather/train_data_functions.py, add one dimention to the normalization.
class ValData(data.Dataset)/class TrainData(data.Dataset):
  ...
  def get_images(self, index):
    ...
    # --- Transform to tensor --- #
    # transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
    transform_gt = Compose([ToTensor()])
    input_im = transform_input(input_img)
    gt = transform_gt(gt_img)

    return input_im, gt, input_name

##Explicit Depth Experiment:
For the second experiment, we added the depth information to the input image in an explicit way - by adding a depth loss to explicitly guide the model to consider depth information. The high level idea is for each interation where back propagation happens, we conduct depth estimate on the predicted image and corresponding ground truth image, and calculate the depth loss. The depth loss is then added to the total loss. 
For this experiemnt, we faced some significant engineering challenges at first. Our first try is still using the Marigold model. But the problem is that transweather uses python 3.6 and torch 1.7.1, whereas Marigold uses python 3.10 and torch 2.0.1. The environment required by the two models are not compatible. On top of that, there's the issue of making sure the instance we launch supports the corresponding cuda version. After failing to create a compatible environment for both, we first tried to resolve the issue with setting up marigold as an api endpoint and call the endpoint during each iteration. After doing so, we found out that the original slow nature of the marigold as a diffusion model plus the communication time of calling endpoint made the 200 epochs training time-wise impossible. So for the second try, we tried with MiDas, another depth estimation model that is CNN based and relatively faster. But still, the environment of midas is also not compatible with the transweather environment. So we tried a third depth estimation model that is utilized in the paper that proposed the depth loss, ADDS-DepthNet, https://github.com/LINA-lln/ADDS-DepthNet. Fortunately, ADDS-DepthNet is compatible with the transweather environment and can be smoothly integrated with transweather. We directly used the model weights shared by the ADDS-DepthNet repo, added model loading and inference logic and implementation code of ADDS-DepthNet into the transweather codebase, and successfully integrated the depth loss into the transweather model. The model is trained with the same hyperparameters as the baseline, with the exception that batchsize is set to 16 because due to the additional model that is loaded to GPU, 16GPU ram can no longer support 32 batchsize training, and the newly  added depth loss factor is 0.01, compared to perceptual loss factor 0.04. The command to run is python train.py -train_batch_size 32 -exp_name try3. The training time for 200 epochs is roughly 150 hours on a T4 machine on AWS.
Modification in Transweather/train.py:
Model Loading:
# --- Load depth --- #
model_path = "/home/ec2-user/ADDS-DepthNet/Pretrained_model"
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()
Depth prediction in each iteration and depth loss calculation:
for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
#-------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)
        
        # Depth prediction for input_image
        features_input = encoder(input_image, 'day', 'val')
        outputs_input = depth_decoder(features_input)
        disp_input = outputs_input[("disp", 0)]
        disp_input_resized = torch.nn.functional.interpolate(
            disp_input, input_image.shape[2:], mode="bilinear", align_corners=False)

        # Depth prediction for ground truth
        features_gt = encoder(gt, 'day', 'val')
        outputs_gt = depth_decoder(features_gt)
        disp_gt = outputs_gt[("disp", 0)]
        disp_gt_resized = torch.nn.functional.interpolate(
            disp_gt, gt.shape[2:], mode="bilinear", align_corners=False)

        # Calculate depth loss (here using simple L1 loss as an example)
        depth_loss = F.l1_loss(disp_input_resized, disp_gt_resized)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)

        # loss = smooth_loss + lambda_loss*perceptual_loss 
        loss = smooth_loss + lambda_loss*perceptual_loss + 0.01*depth_loss

##Ablations for Explicit Depth Experiment:
Potentially there are several meaningful ablations to do:
1. Different depth loss factor
2. The influence of adding depth information at different dataset scale
3. Different learning rate, batchsize, etc.
Due to the long training hours and constraint of time and budge, we conducted the followings:
1. Different depth loss factor on one tenth of training set.
2. Training for longer epochs of the implciti one.
