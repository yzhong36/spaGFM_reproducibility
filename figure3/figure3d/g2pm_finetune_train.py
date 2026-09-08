import os.path as Path
import os
import logging
from stRoamer.training.run_train_ft import run_ft
import torch
import argparse



# Create the ArgumentParser object with a description
parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")
parser.add_argument("train_file_fold", help="The absolute folder of the input training pt files.")
parser.add_argument("--frozen_backbone", action="store_true", help="The absolute folder of the input training pt files.")
parser.add_argument("--batch_size", type= int, default=80, help="The absolute folder of the input training pt files.")
parser.add_argument("--lr", type= float, default=0.005, help="The absolute folder of the input training pt files.")


# # Add a positional argument (required)
# parser.add_argument("train_file_path", help="The absolute path of the input training pt file.")
#
# # Add a positional argument (required)
# parser.add_argument("val_file_path", help="The absolute path of the input validation pt file.")


# Parse the arguments from the command line
args = parser.parse_args()



gpu_id = 0  # Change to your desired GPU index
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
train_file_fold = args.train_file_fold

train_file_path = Path.join(train_file_fold, 'train.pt')
val_file_path = Path.join(train_file_fold, 'val.pt')

graph_infernece_l1_ref = torch.load(train_file_path, map_location=device, weights_only=False)
train_graph = graph_infernece_l1_ref
# train_graph.y = graph_infernece_l1_ref.niche
# train_graph.y = graph_infernece_l1_ref.binary_tls
train_graph.y = graph_infernece_l1_ref.sub_tls

val_graph = torch.load(val_file_path, map_location=device, weights_only=False)
val_graph.y = val_graph.sub_tls


ft_params = {}
ft_params['mode'] = 'train'
# ft_params['walk_length'] = 8 # try smaller values
ft_params['task_type'] = 'classification'

ft_params['subgraph_cls_token'] = False
ft_params['subgraph_pooling'] = 'attn'
ft_params['output_dim'] = len(train_graph.y.unique())
ft_params['linear_decoder'] = True
ft_params['frozen_backbone'] = args.frozen_backbone
ft_params['ft_batch_size'] = args.batch_size
ft_params['epochs'] = 50  #1  ## better to set >=20
ft_params['save_epoch_wise_size'] = 1
ft_params['lr'] = args.lr
ft_params['device'] = device


# model_zoo = {'36M': '/fs/ess/PAS1475/yzhong/sf_project/jobs/novae_human/scale_model_size/02_03_26/run_scgpt_f_pretrain_stRoamer_beta_36M_var_loss_8_2026-02-05_04-03-10/step_100000.pt',
#              "317M":'/fs/ess/PAS1475/yzhong/sf_project/jobs/novae_human/scale_model_size/02_06_26/run_scgpt_f_pretrain_stRoamer_beta_317M_var_loss_8_2026-02-07_14-28-23/step_100000.pt'}
# model_zoo = {'36M': '/fs/ess/PAS1475/yzhong/sf_project/jobs/novae_human/scale_model_size/02_03_26/run_scgpt_f_pretrain_stRoamer_beta_36M_var_loss_8_2026-02-05_04-03-10/step_100000.pt'}

model_zoo = {"3M":'/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26/run_scgpt_f_human_full_3M_8_2026-05-22_19-39-46_balanced_quartz/step_150000.pt',
             "15M":'/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26/run_scgpt_f_human_full_15M_8_2026-05-22_11-31-27_balanced_quartz/step_150000.pt',
            "36M":'/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26/run_scgpt_f_human_full_36M_8_2026-05-21_16-28-12_balanced_quartz/step_150000.pt',
            # "317M":'/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26/run_scgpt_f_human_full_317M_8_2026-05-23_10-24-41_balanced_quartz/step_150000.pt'
             }

# model_zoo = {"317M":'/fs/ess/PAS1475/yzhong/sf_project/jobs/spatial_corpus/scale_model_size/05_19_26/run_scgpt_f_human_full_317M_8_2026-05-23_10-24-41_balanced_quartz/step_150000.pt'}

if args.frozen_backbone:
    backbone = 'frozen'
else:
    backbone = 'unfrozen'


for model_name, model_file in model_zoo.items():

    ft_params[
        'save_folder'] = f'/fs/ess/PAS1475/Fei/data/spaGFM/TLS_results/5fold_CV/{Path.basename(train_file_fold)}_{backbone}_b{args.batch_size}_lr{args.lr}_checkpoint/'
    os.makedirs(ft_params['save_folder'], exist_ok=True)


    for walk_length in range(1, 9):

        ft_params['walk_length'] = walk_length  # try smaller values
        ft_params['save_path'] = Path.join(ft_params['save_folder'], model_name,  str(walk_length))
        # Configure logging to save to 'app.log'
        logger = logging.getLogger(__name__)

        # 2. Create a handler for saving to a file
        os.makedirs(Path.join(ft_params['save_folder'], model_name), exist_ok=True)
        file_handler = logging.FileHandler(Path.join(ft_params['save_folder'], model_name, f'{backbone}_{model_name}_b{args.batch_size}_lr{args.lr}_{Path.basename(train_file_fold)}.log'))
        file_handler.setLevel(logging.INFO)  # Only save warnings or worse to this file

        # 3. Create a formatter and add it to the handler
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 4. Add the handler to your logger
        logger.addHandler(file_handler)

        run_ft(model_file=model_file,
               train_graph=train_graph,
               val_graph = val_graph,
               ft_params=ft_params,
               logger=logger)