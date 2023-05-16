import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch_pruning as tp
import argparse
import math
import warnings
from torch.utils.data import DataLoader
from datasets import *
from model_builder import *
from functools import partial
import sys


warnings.filterwarnings("ignore")
'''
    Train on voxceleb2 and test on voxceleb1
'''
parser = argparse.ArgumentParser()

# For pruning
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--speed-up", type=float, default=2)
parser.add_argument("--max-sparsity", type=float, default=1.0)
parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
parser.add_argument("--reg", type=float, default=5e-4)
parser.add_argument("--weight-decay", type=float, default=5e-4)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=True)
parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
parser.add_argument("--sl-restore", type=str, default=None)
parser.add_argument("--iterative-steps", default=5, type=int)
parser.add_argument("--mode", type=str, choices=['train', 'prune', 'eval'])
parser.add_argument("--init_model", type=str)

args = parser.parse_args()


def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, base_nparams = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, nparams = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        sys.stderr.write(
            "  Params: %.2f M => %.2f M\n"
            % (base_nparams, nparams)
        )
        sys.stderr.write(
            "  MACs: %.2f G => %.2f G\n"
            % (base_ops / 1e9, pruned_ops / 1e9)
        )
        sys.stderr.write("  Speed up: %s\n"%current_speed_up)
        sys.stderr.write("="*16)
        sys.stderr.flush()
        
    return current_speed_up


def get_pruner(model, example_inputs):
    args.sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 192:
            ignored_layers.append(m)
    
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters
    )
    return pruner

def train_model(ecapa_model,
                train_loader, 
                start_epoch, 
                max_epoch, 
                save_path,
                eval_list,
                eval_path, 
                test_step=1):
    epoch = start_epoch
    EERs = []
    score_file = open(save_path, "a+")
    best_eer = np.inf

    while(1):
        ## Training for one epoch
        loss, lr, acc = ecapa_model.train_network(epoch = epoch, loader=train_loader)

        ## Evaluation every [test_step] epochs
        if epoch % test_step == 0:
            EER, minDCF,thresholds = ecapa_model.eval_eer(
                        eval_list=eval_list, 
                        eval_path=eval_path,
                        tuning=True,
                        thresholds=None)
            EERs.append(EER)
            sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n" %(epoch, acc, EERs[-1], min(EERs)))
            score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
            score_file.flush()

            if EER < best_eer:
                ecapa_model.save_parameters(os.path.join(save_path,"/model_best_eer.model"))

        if epoch >= max_epoch:
            quit()

        epoch += 1


if __name__ == '__main__':
    configs = utils.load_config_file('configs/configs_vox.yaml')
    audio_cfgs = configs['AudioProcessing']
    dataset_cfgs = configs['Dataset']
    param_cfgs = configs['Parameters']
    folder_cfgs = configs['RunningFolder']

    train_transform = build_transform(audio_config=audio_cfgs,
                                        mode='train',
                                        noise_path=dataset_cfgs,
                                        stage=2)
    train_list = '../voxceleb_trainer/dataset_tmp/vox2_dev/train_list.txt'
    train_path = '../voxceleb_trainer/dataset_tmp/vox2_dev/aac'
    train_dataset = VoxCelebDataset(train_list=train_list,
                                    train_path=train_path,
                                    sample_rate=16000,
                                    transform=train_transform)
    # print(len(set(train_dataset.data_label)))

    train_loader = DataLoader(train_dataset,
                            batch_size=param_cfgs['batch_size'],
                            num_workers=param_cfgs['num_workers'],
                            shuffle=True,
                            drop_last=True)

    ecapa_model = ECAPAModel(configs=configs,
                        n_class=5994)

    if args.init_model:
        ecapa_model.load_parameters(args.init_model)
    
    eval_list = '../voxceleb_trainer/dataset/VoxCeleb1/veri_test2.txt'
    eval_path = '../voxceleb_trainer/dataset/VoxCeleb1/test/wav'
    # print(len(train_dataset.data_list))

    if args.mode == 'train':
        train_model(train_loader, 
                      start_epoch=1,
                      max_epoch=param_cfgs['max_epoch'],
                      save_path=os.path.join(folder_cfgs['run_path'], folder_cfgs['score_file']),
                      test_step=param_cfgs['test_step'],
                      eval_list=eval_list,
                      eval_path=eval_path)
        # for num, batch in enumerate(tqdm.tqdm(train_loader)):
        #     labels = torch.LongTensor(batch['target']).to('cuda:1')
        #     data = batch['input'].to('cuda:1')
        
    elif args.mode == 'prune':
        model = ecapa_model.speaker_encoder.to('cpu')
        # model.eval()
        # example_inputs = train_dataset['input'][:2].to('cpu')
        # print(example_inputs.size())
        example_inputs = torch.rand(1,32240)
        sys.stderr.write("Pruning phase\n")
        pruner = get_pruner(model, example_inputs=example_inputs)
        
        progressive_pruning(pruner,model, speed_up=args.speed_up, example_inputs=example_inputs)

        model = model.to(param_cfgs['device'])
        ecapa_model.speaker_encoder = model
        sys.stderr.write("Retraining phase\n")
        train_model(ecapa_model,
                    train_loader, 
                    start_epoch=1,
                    max_epoch=param_cfgs['max_epoch'],
                    save_path=os.path.join(folder_cfgs['run_path'], folder_cfgs['score_file']),
                    test_step=param_cfgs['test_step'],
                    eval_list=eval_list,
                    eval_path=eval_path)
        

    # path_model = 'pretrain.model'
    # # print('The number of classes', 30)
    # # print("Model %s loaded from previous state!" % 'pretrain.model')
    # ecapa_model.load_parameters(path_model)
    # model = ecapa_model.speaker_encoder
    # model.eval()
    # example_inputs = torch.randn(1,16000,device='cpu')
    # imp = tp.importance.MagnitudeImportance(p=1)
    # pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=True)
    # iterative_steps = 5
    # pruner = pruner_entry(
    #         model,
    #         example_inputs,
    #         importance=imp,
    #         iterative_steps=50,
    #         ch_sparsity=1,
    #     )
    # 
    