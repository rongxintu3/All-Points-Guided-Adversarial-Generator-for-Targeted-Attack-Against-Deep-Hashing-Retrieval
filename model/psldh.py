import os
import torch
import torch.optim as optim

from model.backbone import *
from utils.hamming_matching import *
import pickle


def data_loss(label_code, label, train_b, train_label, gamma=0.2, sigma=0.2, epsilon = 1e-5):

    code_length = label_code.shape[1]
    logit = train_b.mm(label_code.t())

    max_item = logit.max(1)[0]
    logit = logit - max_item.view(-1, 1)
    sim_label = (train_label.mm(label.t()) > 0).float()
    our_logit = torch.exp((logit - sigma * code_length) * gamma) * sim_label
    mu_logit = (torch.exp(logit * gamma) * (1 - sim_label)).sum(1).view(-1, 1) + our_logit
    loss = - (((torch.log(our_logit / (mu_logit + epsilon) + epsilon) * sim_label).sum(1) / (
                sim_label.sum(1) + epsilon))).mean()
    return loss

class PSLDH(object):
    def __init__(self, args):
        super(PSLDH, self).__init__()
        self.bit = args.bit
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.backbone = args.backbone
        self.model_name = '{}_PSLDH_{}_{}_{}'.format(args.dataset, self.backbone, args.bit, args.round)
        if args.dataset == 'FLICKR-25K':
            with open('/home/trc/PSLDH/label_codes/mir_label_code_' + str(args.bit) + '.pkl', 'rb') as f:
                self.label_code = pickle.load(f)
        elif args.dataset == 'NUS-WIDE':
            with open('/home/trc/PSLDH/label_codes/nus_label_code_' + str(args.bit) + '.pkl', 'rb') as f:
                self.label_code = pickle.load(f)
        else:
            with open('/home/trc/PSLDH/label_codes/coco_label_code_' + str(args.bit) + '.pkl', 'rb') as f:
                self.label_code = pickle.load(f)
        print(self.model_name)
        self.args = args

        self._build_graph()

    def _build_graph(self):
        if self.backbone == 'AlexNet':
            self.model = AlexNet(self.args.bit)
        elif 'VGG' in self.backbone:
            self.model = VGG(self.backbone, self.args.bit)
        else:
            self.model = ResNet(self.backbone, self.args.bit)
        self.model = self.model.cuda()

    def load_model(self):
        self.model = torch.load(
            os.path.join(self.args.save, self.model_name + '.pth'))
        self.model = self.model.cuda()

    # def EncodingOnehot(self, target, nclasses):
    #     target_onehot = torch.FloatTensor(target.size(0), nclasses)
    #     target_onehot.zero_()
    #     target_onehot.scatter_(1, target.view(-1, 1), 1)
    #     return target_onehot

    def CalcSim(self, batch_label, train_label):
        S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
        return S

    def log_trick(self, x):
        lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(
            x, Vtorch.FloatTensor([0.]).cuda())
        return lt

    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1**(epoch // (self.args.n_epochs // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def generate_code(self, data_loader, num_data):
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        for iter, data in enumerate(data_loader, 0):
            data_input, _, data_ind = data
            data_input = data_input.cuda()
            output = self.model(data_input)
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        return B

    def train(self, train_loader, train_labels, num_train, database_loader, test_loader, database_labels, test_labels,
             num_database, num_test):
        if "VGG" in self.backbone:
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        else:
            optimizer = optim.SGD([{'params': self.model.feature_layers.parameters(), 'lr': self.args.lr * 0.01},
                                   {'params': self.model.hash_layer.parameters(), 'lr': self.args.lr}],
                                  weight_decay=self.args.wd)
        nclass = train_labels.shape[1]
        best_map = 0.
        labels = torch.zeros((nclass, nclass)).type(torch.FloatTensor).cuda()
        for i in range(nclass):
            labels[i, i] = 1
        for epoch in range(self.args.n_epochs):
            epoch_loss = 0.0
            epoch_loss_r = 0.0
            epoch_loss_e = 0.0
            self.model.train()
            ## training epoch
            for iter, traindata in enumerate(train_loader, 0):
                train_img, train_label, batch_ind = traindata
                train_label = torch.squeeze(train_label)
                train_img = train_img.cuda()
                train_label = train_label.type(torch.FloatTensor).cuda()
                the_batch = len(batch_ind)
                self.model.zero_grad()
                hash_out = self.model(train_img)
                loss = data_loss(self.label_code, labels, hash_out, train_label, gamma=0.2,
                                 sigma=0.2)
                Bbatch = torch.sign(hash_out)
                regterm = (Bbatch - hash_out).pow(2).sum()
                loss_all = loss + regterm * 0.01 / the_batch

                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()
                epoch_loss += loss_all.item()
                epoch_loss_e += loss.item() / the_batch
                epoch_loss_r += regterm.item() / the_batch
            print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f, Loss_e: %3.5f, Loss_r: %3.5f]' %
              (epoch + 1, self.args.n_epochs, epoch_loss / len(train_loader), epoch_loss_e / len(train_loader),
               epoch_loss_r / len(train_loader)))
            optimizer = self.adjust_learning_rate(optimizer, epoch)

            if (epoch + 1) % 10 == 0:
                map_c = self.test(database_loader, test_loader, database_labels, test_labels, num_database, num_test)
                if map_c > best_map:
                    best_map = map_c
                    torch.save(self.model, os.path.join(self.args.save, self.model_name + '.pth'))


    def test(self, database_loader, test_loader, database_labels, test_labels,
             num_database, num_test):
        self.model.eval()
        qB = self.generate_code(test_loader, num_test)
        dB = self.generate_code(database_loader, num_database)
        # map_ = CalcMap(qB, dB, test_labels.numpy(), database_labels.numpy())
        # print('Test_MAP(retrieval database): %3.5f' % (map_))
        map_ = CalcTopMap(qB, dB, test_labels, database_labels.numpy(), 5000)
        print('Test_MAP(retrieval database): %3.5f' % (map_))
        return map_
        # database_code_path = '/home/trc/CgAT-our/hash_codes/{}defense_dpsh_{}_{}_{}_code.pkl'.format(
        #     'no',
        #     self.args.dataset,
        #     self.args.backbone,
        #     self.args.bit)
        # with open(database_code_path, 'wb') as f:
        #     pickle.dump(
        #         {'B': dB, 'L': database_labels.numpy(), 'Bq': qB,
        #          'Lq': test_labels},
        #         f)
