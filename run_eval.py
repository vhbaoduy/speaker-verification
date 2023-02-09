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
LOOP = 4
EXP = 'exp1'
DEVICE = 'cuda:2'
start_digit = 0
end_digit = 9



def eval_data(configs,args):
    audio_cfgs = configs['AudioProcessing']
    dataset_cfgs = configs['Dataset']
    param_cfgs = configs['Parameters']
    folder_cfgs = configs['RunningFolder']

    eval_info = {
        'female': configs['Pairs']['Female'],
        'male': configs['Pairs']['Male'],
        'all': configs['Pairs']['Overall'],
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
    
    
    results = {}
    for gender in eval_info:
        # if not args.tune_threshold:
        #     print('Threshold', tuned_threshold[gender])
        eval_list = eval_info[gender]['%s_list'%(name_set)]
        EER, minDCF,thresholds = model.eval_eer(
            eval_list=eval_list, eval_path=dataset_cfgs['root_dir'],
            tuning=True,
            thresholds=[1,1])
        
        results[gender] = {'eer': EER, 'minDCF': minDCF}
        
        sum_eer += EER
        sum_minDCF += minDCF
        sys.stderr.write("Gender %s, EER %2.2f%%, minDCF %.4f, threshold %s\n" % (gender, EER, minDCF,thresholds))
        sys.stderr.flush()
        # score_file.write("Gender %s, EER %2.2f%%, minDCF %.4f, threshold %s\n" % (gender, EER, minDCF,thresholds))
        # score_file.flush()
        
        # if args.tune_threshold:
        #     threshold_store[gender] = thresholds
    # results['overall'] = {'eer': sum_eer/2, 'minDCF': sum_minDCF/2}
    json.dump(results, open(folder_cfgs['run_path'] + '/all_results.json','w'))

    return

if __name__ == '__main__':
    for i in range(start_digit,end_digit+1):
        text = str(i) + ('_' + str(i))*(LOOP-1)
        CONFIGS['Parameters']['device'] = DEVICE
        CONFIGS['Dataset']['path'] = '/loop%s/%s' % (LOOP, text)
        CONFIGS['AudioProcessing']['duration'] = LOOP
        CONFIGS['RunningFolder']['run_path'] = os.path.join('pretrained_checkpoints',EXP,'loop%s/%s' % (LOOP, text))
        CONFIGS['Pairs']['Male']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'male_eval.txt')
        CONFIGS['Pairs']['Female']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'female_eval.txt')
        CONFIGS['Pairs']['Overall']['eval_list'] =  os.path.join(CONFIGS['Dataset']['root_dir'],'loop%s/%s' % (LOOP, text),'all_eval.txt')
        args = {'info_data':'meta_data/stage2/split_0.5_stage2.json',
            'stage':2,
            'eval':True,
            'set':'eval',
            'df_train':'data/loop1/0/background.csv',
            'dataset_name':'audio_mnist',
            'init_model':'non_pretrained_checkpoints/%s/loop%s/%s/model_best_acc.model' % (EXP, LOOP, text)}
        eval_data(configs= CONFIGS, args=args)
    