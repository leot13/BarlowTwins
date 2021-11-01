import torch

IMG_HEIGHT = 32 #In the paper, it is 224
IMG_WIDTH = 32 
DEVICE =  "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_FILENAME = "Barlow_Twins_check.pth.tar"
CHECKPOINT_FT_FILENAME = "Barlow_Twins_FT_check.pth.tar"
PROJECT_NAME = "BarlowTwins_project"
LOAD_CHECKPOINT = False
SAVE_CHECKPOINT = True
LR = 3e-4 
NUM_EPOCHS = 200 #In the paper, they run for >300 epochs
FT_NUM_EPOCHS = 5
BATCH_SIZE = 256
NUM_CAT = 10
IN_FEATURES =2048
Z_DIM = 2048 #Dimension used through the projector's layers. 8192 in the paper
LAMBDA = 5e-3