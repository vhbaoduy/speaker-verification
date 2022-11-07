'''
This part is used to train the speaker model and evaluate the performances
'''

import torch
import sys
import os
import tqdm
import numpy
import soundfile
import time
import pickle
import torch.nn as nn
from utils import *
from losses import AAMsoftmax
from models import ECAPA_TDNN
import pandas as pd
from metrics import *
from transforms import build_transform
import copy


class ECAPAModel(nn.Module):
    def __init__(self, configs, n_class):
        super(ECAPAModel, self).__init__()
        self.configs = configs
        self.audio_cfgs = configs['AudioProcessing']
        param_cfgs = configs['Parameters']

        device = param_cfgs['device']
        # ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C=param_cfgs['C']).to(device)
        # Classifier
        self.speaker_loss = AAMsoftmax(
            n_class=n_class, m=param_cfgs['m'], s=param_cfgs['s']).to(device)

        self.optim = torch.optim.Adam(
            self.parameters(), lr=param_cfgs['lr'], weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                         step_size=param_cfgs['test_step'],
                                                         gamma=param_cfgs['lr_decay'])

        self.device = device
        self.param_cfgs = param_cfgs
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f" % (
            sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        metric = AccumulatedAccuracyMetric()
        # Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']

        for num, batch in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(batch['target']).to(self.device)
            data = batch['input'].to(self.device)
            speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            nloss, prec, preds = self.speaker_loss.forward(
                speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()

            metric(preds, labels, nloss)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                             " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) +
                             " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), metric.value()))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, metric.value()

    def eval_acc(self, epoch, loader):
        with torch.no_grad():
            self.eval()
            metric = AccumulatedAccuracyMetric()
            index, top1, loss = 0, 0, 0
            for num, batch in enumerate(loader, start=1):
                labels = torch.LongTensor(batch['target']).to(self.device)
                data = batch['input'].to(self.device)
                speaker_embedding = self.speaker_encoder.forward(
                    data, aug=False)
                nloss, prec, preds = self.speaker_loss.forward(
                    speaker_embedding, labels)
                index += len(labels)
                top1 += prec
                loss += nloss.detach().cpu().numpy()

                metric(preds, labels, nloss)
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                                 " [%2d] Validating: %.2f%%, " % (epoch, 100 * (num / loader.__len__())) +
                                 " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), metric.value()))
                sys.stderr.flush()
            sys.stdout.write("\n")
            return loss / num, metric.value()

    def eval_stage_1(self, loader, classes, path_to_result):
        with torch.no_grad():
            self.eval()
            stat = {}
            metric = AccumulatedAccuracyMetric()
            index, top1, loss = 0, 0, 0
            for num, batch in enumerate(loader, start=1):
                labels = torch.LongTensor(batch['target']).to(self.device)
                data = batch['input'].to(self.device)
                speaker_embedding = self.speaker_encoder.forward(
                    data, aug=False)
                nloss, prec, preds = self.speaker_loss.forward(
                    speaker_embedding, labels)
                index += len(labels)
                top1 += prec
                loss += nloss.detach().cpu().numpy()

                metric(preds, labels, nloss)

                preds = preds.data.max(1, keepdim=True)[
                    1].cpu().numpy().ravel()
                labels = labels.cpu().numpy().ravel()
                for i in range(len(batch['input'])):
                    word = batch['word'][i]
                    if word not in stat:
                        stat[word] = {
                            'true': 0,
                            'false': 0
                        }
                    pred = index2label(classes, preds[i])
                    truth = index2label(classes, labels[i])
                    if pred == truth:
                        stat[word]['true'] += 1
                    else:
                        stat[word]['false'] += 1

                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                                 " Validating: %.2f%%, " % (100 * (num / loader.__len__())) +
                                 " Loss: %.5f, ACC: %2.2f%% \r" % (loss / (num), metric.value()))
                sys.stderr.flush()
            sys.stdout.write("\n")
            res = {
                'word': [],
                'true': [],
                'false': []
            }
            for w in stat:
                res['word'].append(w)
                res['true'].append(stat[w]['true'])
                res['false'].append(stat[w]['false'])

            res = pd.DataFrame(res)
            res.to_csv(path_to_result, index=False)
            return res

    def eval_eer(self, eval_list, eval_path):
        with torch.no_grad():
            self.eval()
            files = []
            embeddings = {}
            lines = open(eval_list).read().splitlines()
            for line in lines:
                files.append(line.split()[1])
                files.append(line.split()[2])
            setfiles = list(set(files))
            setfiles.sort()
            trans_1 = build_transform(audio_config=self.audio_cfgs,
                                      mode='eval',
                                      num_stack=1)
            trans_2 = build_transform(audio_config=self.audio_cfgs,
                                      mode='eval',
                                      num_stack=5)
            for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
                audio, sr = utils.load_audio(os.path.join(eval_path, file),self.audio_cfgs['sample_rate'])
                data_1 = {
                    'samples': audio,
                    'sample_rate': sr
                }
                data_2 = copy.deepcopy(data_1)

                # audio, _ = soundfile.read(os.path.join(eval_path, file))
                # Full utterance
                # data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()
                data_1 = trans_1(data_1)
                # print(data_1)
                # print("#"*10, data)
                # Splited utterance matrix
                data_2 = trans_2(data_2)
                data_1 = data_1['input'].to(self.device)
                data_2 = data_2['input'].to(self.device)
                # data_2 = data_2.unsqueeze(1)
                # Speaker embeddings
                with torch.no_grad():
                    embedding_1 = self.speaker_encoder.forward(
                        data_1, aug=False)
                    embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                    embedding_2 = self.speaker_encoder.forward(
                        data_2, aug=False)
                    embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                # embedding_1 = embedding_1.detach().cpu().numpy()
                # embedding_2 = embedding_2.detach().cpu().numpy()
                embeddings[file] = [embedding_1, embedding_2]
            scores, labels = [], []

            for line in lines:
                embedding_11, embedding_12 = embeddings[line.split()[1]]
                embedding_21, embedding_22 = embeddings[line.split()[2]]
                # Compute the scores
                score_1 = torch.mean(torch.matmul(
                    embedding_11, embedding_21.T))  # higher is positive
                score_2 = torch.mean(torch.matmul(
                    embedding_12, embedding_22.T))
                score = (score_1 + score_2) / 2
                score = score.detach().cpu().numpy()
                scores.append(score)
                labels.append(int(line.split()[0]))

            # Coumpute EER and minDCF
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
            return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=self.device)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                    origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)


if __name__ == '__main__':
    configs = utils.load_config_file('configs/configs.yaml')
    model = ECAPA_TDNN(C=1024)
    trans = build_transform(audio_config=configs['AudioProcessing'],
                    mode='eval',
                    num_stack=3)
    audio, sr = utils.load_audio(
        './combine_data\\0ff728b5\\five_0_five_0_.wav', 16000)
    data = {
        'samples': audio,
        'sample_rate': sr
    }
    data = trans(data)
    from IPython import embed
    embed()
