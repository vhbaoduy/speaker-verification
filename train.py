import argparse
import math
import warnings
from torch.utils.data import DataLoader
from datasets import *
from model_builder import *

warnings.filterwarnings("ignore")

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('-root_dir', type=str,
                        default='./data', help='path to root dir dataset')
    parser.add_argument('-dataset_name', type=str, required=True,
                        choices=['arabic', 'google_speech_v0.01', 'google_speech_v0.02','audio_mnist'],
                        help='name of dataset')

    parser.add_argument('-df_train', type=str, default='./output/train.csv',
                        help='path to df train in prepare_data.py')
    parser.add_argument('-df_valid', type=str, default='./output/val.csv',
                        help='path to df valid in prepare_data.py')
    parser.add_argument('-info_data', type=str,
                        default='./output/info.json', help='path to info of data')
    parser.add_argument('-config_file', type=str,
                        default='configs.yaml', help='name of config file')
    parser.add_argument('-init_model', type=str,
                        help='path to pretrained model')
    parser.add_argument('-eval_list', type=str, default="/data08/VoxCeleb1/veri_test2.txt",
                        help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
    parser.add_argument('-eval_path', type=str, default="/data08/VoxCeleb1/test/wav",
                        help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
    parser.add_argument('-seed', type=int, default=2022, help='seed of train')
    parser.add_argument(
        '-stage', type=int, choices=[1, 2], default=1, help='use {1:accuracy,2:eer} to evaluate')
    parser.add_argument('-eval', type=bool, default=False)
    parser.add_argument('-path_to_result', type=str)
    parser.add_argument('-gender', type=str, default='mix', choices=['mix', 'female', 'male'])
    # Parse args
    args = parser.parse_args()

    configs = load_config_file(os.path.join('./configs', args.config_file))

    audio_cfgs = configs['AudioProcessing']
    dataset_cfgs = configs['Dataset']
    param_cfgs = configs['Parameters']
    folder_cfgs = configs['RunningFolder']

    with open(args.info_data, 'r') as file_in:
        info = json.load(file_in)

    # torch.manual_seed(args.seed)
    if args.stage == 1:
        if args.gender == 'mix':
            classes = info['speakers']
        elif args.gender == 'female':
            classes = info['female_speakers']
        else:
            classes = info['male_speakers']
    else:
        classes = info['speaker_train']
    n_class = len(classes)

    train_transform = build_transform(audio_config=audio_cfgs,
                                      mode='train',
                                      noise_path=dataset_cfgs,
                                      stage=args.stage)
    train_dataset = GeneralDataset(root_dir=dataset_cfgs['root_dir'],
                                   path_to_df=args.df_train,
                                   classes=classes,
                                   sample_rate=audio_cfgs['sample_rate'],
                                   dataset_name=args.dataset_name,
                                   stage=args.stage,
                                   transform=train_transform,
                                   gender=args.gender)

    train_loader = DataLoader(train_dataset,
                              batch_size=param_cfgs['batch_size'],
                              num_workers=param_cfgs['num_workers'],
                              shuffle=True,
                              drop_last=True)
    if args.stage == 1:
        valid_transform = build_transform(audio_config=audio_cfgs,
                                          mode='eval',
                                          stage=1)
        valid_dataset = GeneralDataset(root_dir=dataset_cfgs['root_dir'],
                                       path_to_df=args.df_valid,
                                       classes=classes,
                                       sample_rate=audio_cfgs['sample_rate'],
                                       dataset_name=args.dataset_name,
                                       stage=1,
                                       transform=valid_transform,
                                       gender=args.gender)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=param_cfgs['batch_size'],
                                  num_workers=param_cfgs['num_workers'],
                                  shuffle=False)

    model = ECAPAModel(configs=configs,
                       n_class=n_class)

    if args.eval:
        print("Model %s loaded from previous state!" % args.init_model)
        model.load_parameters(args.init_model)
        if args.stage == 2:
            EER, minDCF = model.eval_network(
                eval_list=args.eval_list, eval_path=args.eval_path)
            print("EER %2.2f%%, minDCF %.4f%%" % (EER, minDCF))
        else:
            res = model.eval_stage_1(
                valid_loader, classes=info['speakers'], path_to_result=args.path_to_result)
        quit()
    # print(args.init_model)
    if args.init_model is not None:
        print("Model %s loaded from previous state!" % args.init_model)
        model.load_parameters(args.init_model)
        epoch = 36
    else:
        epoch = 1

    EERs = []
    if args.stage == 1:
        best_score = -math.inf
    else:
        best_score = math.inf
        best_DCF = math.inf
    if not os.path.exists(folder_cfgs['run_path']):
        os.mkdir(folder_cfgs['run_path'])

    score_file = open(folder_cfgs['run_path'] +
                      '/' + folder_cfgs['score_file'], "a+")
    score_file.write("Seed: %d\n" % args.seed)
    # epoch = 1
    while (1):
        # Training for one epoch
        loss, lr, acc = model.train_network(epoch=epoch, loader=train_loader)

        # Evaluation every [test_step] epochs
        if epoch % param_cfgs['test_step'] == 0:
            if args.stage == 2:
                eer, min_dcf = model.eval_eer(
                    eval_list=args.eval_list, eval_path=args.eval_path)
                EERs.append(eer)
                if EERs[-1] < best_score:
                    best_score = EERs[-1]
                    model.save_parameters(
                        folder_cfgs['run_path'] + "/model_best.model")
                if min_dcf < best_DCF:
                    best_DCF = min_dcf
                print(time.strftime("%Y-%m-%d %H:%M:%S"),
                      "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, minDCF %f, best minDCF %f" % (epoch, acc, EERs[-1], min(EERs), min_dcf, best_DCF))
                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%, minDCF %f, best minDCF %f\n" % (
                    epoch, lr, loss, acc, EERs[-1], min(EERs), min_dcf, best_DCF))
                score_file.flush()
            else:
                _, val_acc = model.eval_acc(epoch=epoch, loader=valid_loader)
                if val_acc > best_score:
                    best_score = val_acc
                    model.save_parameters(
                        folder_cfgs['run_path'] + "/model_best.model")
                print(time.strftime("%Y-%m-%d %H:%M:%S"),
                      "%d epoch, ACC %2.2f%%,BestACC %2.2f%%" % (epoch, val_acc, best_score))
                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, BestACC %2.2f%%\n" % (
                    epoch, lr, loss, acc, best_score))
                score_file.flush()

        if epoch >= param_cfgs['max_epoch']:
            quit()

        epoch += 1
