import argparse
import math
import warnings
from torch.utils.data import DataLoader
from datasets import *
from model_builder import *

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('-root_dir', type=str,
                        default='./data', help='path to root dir dataset')
    parser.add_argument('-dataset_name', type=str, required=True,
                        choices=['arabic', 'google_speech_v0.01', 'google_speech_v0.02','audio_mnist','voxceleb1','voxceleb2'],
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
                        help='path to init model')
    parser.add_argument('-pretrained_model', type=str,
                        help='path to pretrained model')
    parser.add_argument('-seed', type=int, default=2022, help='seed of train')
    parser.add_argument(
        '-stage', type=int, choices=[1, 2], default=1, help='use {1:accuracy,2:eer} to evaluate')
    parser.add_argument('-eval', type=bool, default=False)
    parser.add_argument('-tune_threshold', type=bool, default=False)
    parser.add_argument('-path_to_result', type=str)
    parser.add_argument('-gender', type=str, default='mix', choices=['mix', 'female', 'male'])
    parser.add_argument('-set', type=str,default='dev',choices=['dev', 'eval'] )
    # Parse args
    args = parser.parse_args()

    configs = load_config_file(os.path.join('./configs', args.config_file))

    audio_cfgs = configs['AudioProcessing']
    dataset_cfgs = configs['Dataset']
    param_cfgs = configs['Parameters']
    folder_cfgs = configs['RunningFolder']

    eval_info = {
        'female': configs['Pairs']['Female'],
        'male': configs['Pairs']['Male'],
        'all': configs['Pairs']['Overall']

    }
    name_set = args.set


    with open(args.info_data, 'r') as file_in:
        info = json.load(file_in)

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    if args.stage == 1:
        if args.gender == 'mix':
            classes = info['speakers']
        elif args.gender == 'female':
            classes = info['female_speakers']
        else:
            classes = info['male_speakers']
    else:
        classes = info['development']['female'] + info['development']['male']

    if args.dataset_name not in ['voxceleb1','voxceleb2']:
        n_class = len(classes)
    else:
        n_class = 5994

    if args.dataset_name not in ['voxceleb1','voxceleb2']:
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
    else:
        train_transform = build_transform(audio_config=audio_cfgs,
                                        mode='train',
                                        noise_path=dataset_cfgs,
                                        stage=args.stage)
        train_list = '../voxceleb_trainer/dataset/VoxCeleb2/train_list.txt'
        train_path = '../voxceleb_trainer/dataset/VoxCeleb2'
        train_dataset = VoxCelebDataset(train_list='')

        train_loader = DataLoader(train_dataset,
                                batch_size=param_cfgs['batch_size'],
                                num_workers=param_cfgs['num_workers'],
                                shuffle=True,
                                drop_last=True)

    if args.stage == 1:
        valid_transform = build_transform(audio_config=audio_cfgs,
                                          mode='eval',
                                          noise_path=dataset_cfgs,
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
    print('The number of classes', n_class)
    if args.eval:
        
        if args.dataset_name not in ['voxceleb1','voxceleb2']:
            print("Model %s loaded from previous state!" % args.init_model)
            model.load_parameters(args.init_model)
        else:
            if args.pretrained_model:
                model.load_parameters(args.pretrained_model)
            else:
                encoder = torch.load(args.init_model)
                model.speaker_encoder = encoder
                print('Load Param:', sum(param.numel() for param in encoder.parameters()))
        if args.stage == 2:
            # score_file = open(folder_cfgs['run_path'] +
            #           '/' + folder_cfgs['threshold_file'], "a+")
            # print('Tune threshold', args.tune_threshold)
            
            
            results = {}
            if args.dataset_name not in ['voxceleb1','voxceleb2']:
                tuned_threshold = {'male':[], 'female':[], 'all':[]}
                # sum_eer = 0
                # sum_minDCF = 0
                if args.tune_threshold:
                    threshold_store = {}
                else:
                    tuned_threshold = json.load(open('/'.join(args.init_model.split('/')[:-1]) + configs['Pairs']['threshold_path']))
                for gender in eval_info:
                    if not args.tune_threshold:
                        print('Threshold', tuned_threshold[gender])
                    
                    
                        eval_list = eval_info[gender]['%s_list'%(name_set)]
                        eval_path = dataset_cfgs['root_dir']
                    
                    EER, minDCF,thresholds = model.eval_eer(
                        eval_list=eval_list, eval_path=eval_path,
                        tuning=args.tune_threshold,
                        thresholds=tuned_threshold[gender])
                    
                    results[gender] = {'eer': EER, 'minDCF': minDCF}
                    
                    # sum_eer += EER
                    # sum_minDCF += minDCF
                    sys.stderr.write("Gender %s, EER %2.2f%%, minDCF %.4f, threshold %s\n" % (gender, EER, minDCF,thresholds))
                    sys.stderr.flush()
                    json.dump(results, open(folder_cfgs['run_path'] + '/all_eval_results_tuning_2.json','w'))
    
                    if args.tune_threshold:
                        json.dump(threshold_store, open(folder_cfgs['run_path'] + '/eval_thresholds.json','w'))
            else:
                eval_list = '../voxceleb_trainer/dataset/VoxCeleb1/veri_test2.txt'
                eval_path = '../voxceleb_trainer/dataset/VoxCeleb1/test/wav'
                EER, minDCF,thresholds = model.eval_eer(
                        eval_list=eval_list, eval_path=eval_path,
                        tuning=True,
                        thresholds=None)
                sys.stderr.write("EER %2.2f%%, minDCF %.4f, threshold %s\n" % (EER, minDCF,thresholds))
                # score_file.write("Gender %s, EER %2.2f%%, minDCF %.4f, threshold %s\n" % (gender, EER, minDCF,thresholds))
                # score_file.flush()
                results['eer'] = EER
                results['minDCF'] = minDCF
                results['threshold'] = thresholds
                json.dump(results, open('pruning_results/base.json','w'))
                # if args.tune_threshold:
                #     threshold_store[gender] = thresholds
            # results['overall'] = {'eer': sum_eer/2, 'minDCF': sum_minDCF/2}
            
        else:
            res = model.eval_stage_1(
                valid_loader, classes=info['speakers'], path_to_result=args.path_to_result)
        quit()
    # print(args.init_model)
    if args.pretrained_model is not None:
        print("Model %s loaded from previous state!" % args.pretrained_model)
        model.load_parameters(args.pretrained_model)
    else:
        epoch = 1

    EERs = []
    if args.stage == 1:
        best_score = -math.inf
    else:
        # best_loss = math.inf
        best_acc =  -math.inf
        best_eer = math.inf
        best_DCF = math.inf
    if not os.path.exists(folder_cfgs['run_path']):
        os.mkdir(folder_cfgs['run_path'])

    score_file = open(folder_cfgs['run_path'] +
                      '/' + folder_cfgs['score_file'], "a+")
    score_file.write("Seed: %d\n" % args.seed)
    with open(folder_cfgs['run_path'] + '/config.yaml', 'w') as outfile:
        yaml.dump(configs, outfile, default_flow_style=False)
    epoch = 1
    while (1):
        # Training for one epoch
        loss, lr, acc = model.train_network(epoch=epoch, loader=train_loader)

        # Evaluation every [test_step] epochs
        if epoch % param_cfgs['test_step'] == 0:
            if args.stage == 2:
                if acc >= best_acc:
                    best_acc = acc
                    model.save_parameters(
                        folder_cfgs['run_path'] + "/model_best_acc.model")

                sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") +
                      "%d epoch, ACC %2.2f%%,BestACC %2.2f%%\n" % (epoch, acc, best_acc))
                sys.stderr.flush()

                score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, BestACC %2.2f%%\n" % (
                    epoch, lr, loss, acc, best_acc))
                score_file.flush()

                # sum_eer = 0
                # tuned_threshold = {'male':{}, 'female':{}}

                # print(time.strftime("%Y-%m-%d %H:%M:%S"),
                #       " %d epoch, ACC %2.2f%%, LOSS %f" % (epoch, acc, loss))
                # score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%\n" % (
                #     epoch, lr, loss, acc))
                # score_file.flush()


                # for gender in eval_info:
                #     eval_list = eval_info[gender]['%s_list'%(name_set)]
                #     EER, minDCF, thresholds = model.eval_eer(eval_list=eval_list,
                #                                             eval_path=dataset_cfgs['root_dir'],
                #                                             tuning=True)
                #     sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + 
                #                             " Gender %s, EER %2.2f%%, minDCF %.4f\n" % (gender, EER, minDCF))
                #     sys.stderr.flush()

                #     tuned_threshold[gender] = thresholds
                #     sum_eer += EER
                #     score_file.write("\tGender %s, EER %2.2f%%, minDCF %.4f, threshold %s\n" % (gender, EER, minDCF,thresholds))
                #     score_file.flush()


                # avg_eer = sum_eer / 2
                # if avg_eer < best_eer:
                #     best_eer = avg_eer
                #     model.save_parameters(
                #         folder_cfgs['run_path'] + "/model_best_eer.model")
                #     json.dump(tuned_threshold, open(folder_cfgs['run_path'] + '/thresholds.json','w'))
                
                # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + 
                #       " %d epoch, EER %2.2f%%, BEST_EER %2.2f%%" % (epoch, avg_eer, best_eer))
                # sys.stderr.flush()
                # score_file.write("%d epoch, EER %2.2f%%, BEST_EER %2.2f%%\n" % (epoch, avg_eer, best_eer))
                # score_file.flush()
                # if min_dcf < best_DCF:
                #     best_DCF = min_dcf
                #     model.save_parameters(
                #         folder_cfgs['run_path'] + "/model_best_dcf.model")

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
