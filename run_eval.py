import utils
import os
import yaml
import argparse
import math
import warnings
from torch.utils.data import DataLoader
from datasets import *
from model_builder import *
import json

CONFIGS = utils.load_config_file('configs/configs.yaml')
LOOP = 2
EXP = 'exp1'
DEVICE = 'cuda:2'
start_digit = 0
end_digit = 9
FOLDER_CHECKPOINTS = 'pruning_checkpoints'
# WORDS = [(3, 1), (1, 0), (1, 6), (1, 3), (1, 9), (3, 0), (3, 9), (3, 6), (9, 1), (0, 1), (6, 1), (9, 0), (9, 3), (9, 6), (0, 3), (0, 9), (0, 6), (6, 0), (6, 3), (6, 9)]
# ['0_1', '9_0', '0_3', '0_9', '3_0', '0_6', '6_0', '1_0']


def eval_data(configs,args):
    audio_cfgs = configs['AudioProcessing']
    dataset_cfgs = configs['Dataset']
    param_cfgs = configs['Parameters']
    folder_cfgs = configs['RunningFolder']

    eval_info = {
        # 'female': configs['Pairs']['Female'],
        'male': configs['Pairs']['Male'],
        # 'all': configs['Pairs']['Overall'],
    }
    name_set = args['set']


    with open(args['info_data'], 'r') as file_in:
        info = json.load(file_in)

    # torch.manual_seed(args.seed)
    classes = info['development']['female'] + info['development']['male']
    n_class = len(classes)

    model = ECAPAModel(configs=configs,
                        n_class=n_class)
    print('The number of classes', n_class)
    print("Model %s loaded from previous state!" % args['init_model'])
    model.load_parameters(args['init_model'])
    sum_eer = 0
    sum_minDCF = 0
    print('Tune threshold', args['tune_threshold'])
    tuned_threshold = {'male':[], 'female':[],'all':[]}
    sum_eer = 0
    sum_minDCF = 0
    if args['tune_threshold']:
        threshold_store = {}
    else:
        if configs['Pairs']['threshold_path'] != '':
            tuned_threshold = json.load(open(configs['Pairs']['threshold_path']))
    
    results = {}
    for gender in eval_info:
        if not args['tune_threshold']:
            print('Threshold', tuned_threshold[gender])
        eval_list = eval_info[gender]['%s_list'%(name_set)]
        EER, minDCF,thresholds = model.eval_eer(
            eval_list=eval_list, eval_path=dataset_cfgs['root_dir'],
            tuning=args['tune_threshold'],
            thresholds=tuned_threshold[gender])
        
        results[gender] = {'eer': EER, 'minDCF': minDCF}
        
        sum_eer += EER
        sum_minDCF += minDCF
        sys.stderr.write("Gender %s, EER %2.2f%%, minDCF %.4f, threshold %s\n" % (gender, EER, minDCF,thresholds))
        sys.stderr.flush()
        # score_file.write("Gender %s, EER %2.2f%%, minDCF %.4f, threshold %s\n" % (gender, EER, minDCF,thresholds))
        # score_file.flush()
        
        if args['tune_threshold']:
            threshold_store[gender] = thresholds
    # results['overall'] = {'eer': sum_eer/2, 'minDCF': sum_minDCF/2}
    json.dump(results, open(folder_cfgs['run_path'] + '/all_%s_result_real_data.json'%(name_set),'w'))
    if args['tune_threshold']:
                json.dump(threshold_store, open(folder_cfgs['run_path'] + '/%s_thresholds_real_data.json'%(name_set),'w'))

    return

if __name__ == '__main__':
    start_digit = 0
    end_digit = 9
    for exp in ['exp1','exp2','exp3']:
        EXP = exp
        for loop in [1,2,3,4,5]:
            if loop != 1:
                path_threshold = 'eval_thresholds.json'
                FOLDER_CHECKPOINTS = 'checkpoints/pretrained/channel_128'
            else:
                path_threshold = 'eval_thresholds.json'
                FOLDER_CHECKPOINTS = 'checkpoints/pretrained/channel_128'
            
            LOOP = loop
            sys.stderr.write("%s - %s - %s\n" % (FOLDER_CHECKPOINTS, EXP, LOOP))
            for i in range(start_digit,end_digit+1):
                text = str(i) + ('_' + str(i))*(LOOP-1)
                CONFIGS['Parameters']['device'] = DEVICE
                CONFIGS['Parameters']['C'] = 128
                CONFIGS['Dataset']['path'] = '/loop%s/%s' % (LOOP, text)
                CONFIGS['AudioProcessing']['duration'] = LOOP
                CONFIGS['Dataset']['root_dir'] = 'my_audio_mnist_0_9'
                CONFIGS['RunningFolder']['run_path'] = os.path.join(FOLDER_CHECKPOINTS,EXP,'loop%s/%s' % (LOOP, text))
                CONFIGS['Pairs']['threshold_path'] = os.path.join(FOLDER_CHECKPOINTS,EXP,'loop%s/%s/%s' % (LOOP, text,path_threshold))
                # CONFIGS['Pairs']['threshold_path'] = ''
                CONFIGS['Pairs']['Male']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'male_pairs_tw.txt')
                # CONFIGS['Pairs']['Male']['eval_list'] =  os.path.join('my_audio_mnist_0_9','loop%s/%s' % (LOOP, text),'male_eval_no_tw.txt')
                CONFIGS['Pairs']['Female']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'female_eval_2.txt')
                CONFIGS['Pairs']['Overall']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'all_eval_2.txt')
                CONFIGS['Pairs']['Male']['dev_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'male_dev.txt')
                CONFIGS['Pairs']['Female']['dev_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'female_dev.txt')
                CONFIGS['Pairs']['Overall']['dev_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'all_dev.txt')

                args = {'info_data':'meta_data/stage2/split_0.5_stage2.json',
                    'stage':2,
                    'eval':True,
                    'set':'eval',
                    'tune_threshold':True,
                    'df_train':'data/loop1/0/background.csv',
                    'dataset_name':'audio_mnist',
                    'init_model':'%s/%s/loop%s/%s/model_best_acc.model' % (FOLDER_CHECKPOINTS,EXP, LOOP, text)}
                    # 'init_model':'pretrain.model'}
                eval_data(configs= CONFIGS, args=args)

    # WORDS = [(0, 1),(0, 3), (0, 9), (0, 6),(1,0), (1,3), (1,6), (1,9),(3, 0), (3,1)]
    # WORDS = []
    # WORDS = [ (3,6), (3,9),(6, 0),(6, 1), (6, 3), (6, 9),(9, 0), (9, 1), (9, 3), (9, 6)]
    # # WORDS = [(6, 0),(6, 1), (6, 3), (6, 9)]
    # # WORDS = [(9, 0), (9, 1), (9, 3), (9, 6)]

    # FOLDER_CHECKPOINTS = 'checkpoints/non_pretrained/channel_1024_old/exp1/loop2_experiments'
    # path_threshold = 'eval_thresholds_no_tw.json'
    # for exp in ['']:
    #     EXP = exp
    #     for w in WORDS:
    #         str_list = [str(digit) for digit in w]
    #         text ='_'.join(str_list)
    #         sys.stderr.write(text + '\n')
    #         sys.stderr.flush()
    #         CONFIGS['Parameters']['device'] = DEVICE
    #         CONFIGS['Dataset']['path'] = '/%s' % (text)
    #         CONFIGS['AudioProcessing']['duration'] = LOOP
    #         CONFIGS['Dataset']['root_dir'] = ''
    #         CONFIGS['RunningFolder']['run_path'] = os.path.join(FOLDER_CHECKPOINTS,EXP,'%s' % (text))
    #         CONFIGS['Pairs']['threshold_path'] = os.path.join(FOLDER_CHECKPOINTS,EXP,'%s/%s' % (text,path_threshold))
    #         # CONFIGS['Pairs']['threshold_path'] = os.path.join(FOLDER_CHECKPOINTS,EXP,'loop%s/%s/thresholds.json' % (LOOP, text))
    #         # CONFIGS['Pairs']['threshold_path'] = ''
    #         CONFIGS['Pairs']['Male']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'data/%s/%s' % ('experiments', text),'male_eval_version_2.txt')
    #         CONFIGS['Pairs']['Female']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'data/%s/%s' % ('experiments', text),'female_eval_version_2.txt')
    #         CONFIGS['Pairs']['Overall']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'data/%s/%s' % ('experiments', text),'all_eval_version_2.txt')
    #         CONFIGS['Pairs']['Male']['dev_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'%s/%s' % ('experiments', text),'male_dev.txt')
    #         CONFIGS['Pairs']['Female']['dev_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'%s/%s' % ('experiments', text),'female_dev.txt')
    #         CONFIGS['Pairs']['Overall']['dev_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'%s/%s' % ('experiments', text),'all_dev.txt')
    #         sys.stderr.write("%s - %s - %s\n" % (FOLDER_CHECKPOINTS, EXP, LOOP))
    #         args = {'info_data':'meta_data/stage2/split_0.5_stage2.json',
    #             'stage':2,
    #             'eval':True,
    #             'set':'eval',
    #             'tune_threshold':True,
    #             'df_train':'data/loop1/0/background.csv',
    #             'dataset_name':'audio_mnist',
    #             'init_model':'%s/%s/%s/model_best_acc.model' % (FOLDER_CHECKPOINTS,EXP, text)}
    #             # 'init_model':'pretrain.model'}
    #         eval_data(configs= CONFIGS, args=args)


        