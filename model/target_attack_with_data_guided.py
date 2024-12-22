import os
import time
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable
import pickle
import random
from model.module import *
from model.utils import *
from utils.hamming_matching import *
import torch.nn.functional as F
from model.backbone2 import *

def data_loss_pos(train_b, train_label, B, label, gamma=0.2, sigma=0.0, epsilon = 1e-5):

    code_length = train_b.shape[1]
    same_code = B

    sim = (train_label.mm(label.t()) > 0).float()
    train_b = F.normalize(train_b)
    same_code = F.normalize(same_code)
    

    logit = train_b.mm(same_code.t())
    negative = 1. - sim

    max_item = logit.max(1)[0]
    logit = logit - max_item.view(-1, 1)
    logit = torch.exp((logit - sigma * code_length) * gamma) * sim
    logit_all = (torch.exp(logit * gamma) * negative).sum(1).view(-1, 1) + logit

    loss = - ((torch.log(logit / (logit_all + epsilon) + epsilon) * sim).sum(1) / (sim.sum(1) + epsilon)).mean()
    return loss


class TargetAttackGAN(nn.Module):
    def __init__(self, args):
        super(TargetAttackGAN, self).__init__()
        self.bit = args.bit
        classes_dic = {'FLICKR-25K': 24, 'NUS-WIDE':21, 'MS-COCO': 80}
        self.num_classes = classes_dic[args.dataset]
        self.rec_w = args.rec_w
        self.dis_w = 1
        self.batch_size = args.batch_size
        self.model_name = '{}_{}_{}_{}'.format(args.dataset, args.hash_method, args.backbone, args.bit)
        self.lr = args.lr
        self.args = args

        self._build_model()

    def _build_model(self):
        self.generator = nn.DataParallel(Generator(self.num_classes)).cuda()
        
        if 'CSQ' in self.model_name:
            hashing_model = torch.load(
            os.path.join(self.args.save, self.model_name + '.pth'))
            if 'AlexNet' in self.model_name:
                model = AlexNet(self.args.bit)
            elif 'VGG' in self.model_name:
                model = VGG(self.args.backbone, self.args.bit)
            elif 'ResNet' in self.model_name:
                model = ResNet(self.args.backbone, self.args.bit)
            model.load_state_dict(hashing_model)
            # model.load_state_dict(torch.load(path))
            # model = torch.load(path)
            self.hashing_model = model.cuda()
        else:
            hashing_model = torch.load(
            os.path.join(self.args.save, self.model_name + '.pth'))
            self.hashing_model = hashing_model.cuda()
        self.hashing_model.eval()

        self.criterionGAN = GANLoss('lsgan').cuda()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_hash_code(self, data_loader, num_data, bs):
        B = torch.zeros(num_data, self.bit)
        self.train_labels = torch.zeros(num_data, self.num_classes)
        for it, data in enumerate(data_loader, 0):
            data_input = data[0]
            data_input = Variable(data_input.cuda())
            output = self.hashing_model(data_input)

            batch_size_ = output.size(0)
            u_ind = np.linspace(it * bs,
                                np.min((num_data,
                                        (it + 1) * bs)) - 1,
                                batch_size_,
                                dtype=int)
            B[u_ind, :] = torch.sign(output.cpu().data)
            self.train_labels[u_ind, :] = data[1]
        return B

    def generate_train_code(self, data_loader, num_data, percent_training=1.):
        B = torch.zeros(num_data, self.bit)
        self.train_labels_fixed = torch.zeros(num_data, self.num_classes)
        for it, data in enumerate(data_loader, 0):
            data_input = data[0]
            data_input = Variable(data_input.cuda())
            output = self.hashing_model(data_input)

            batch_size_ = output.size(0)
            u_ind = np.linspace(it * self.batch_size,
                                np.min((num_data,
                                        (it + 1) * self.batch_size)) - 1,
                                batch_size_,
                                dtype=int)
            B[u_ind, :] = torch.sign(output.cpu().data)
            self.train_labels_fixed[u_ind, :] = data[1]
        num_train = B.shape[0]
        selected_index = random.sample(list(range(num_train)), int(percent_training*num_train))
        selected_index = torch.LongTensor(selected_index)
        self.train_codes_fixed = B[selected_index].cuda()
        self.train_labels_fixed = self.train_labels_fixed[selected_index].cuda()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()

        self.lr = self.optimizers[0].param_groups[0]['lr']

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def test_prototype(self, target_labels, database_loader, database_labels, num_database, num_test):
        targeted_labels = np.zeros([num_test, self.num_classes])
        qB = np.zeros([num_test, self.bit])
        for i in range(num_test):
            select_index = np.random.choice(range(target_labels.size(0)), size=1)
            batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
            targeted_labels[i, :] = batch_target_label.numpy()[0]

            _, target_hash_l, __ = self.prototype_net(batch_target_label.cuda().float())
            qB[i, :] = torch.sign(target_hash_l.cpu().data).numpy()[0]

        database_code_path = os.path.join('log', 'database_code_{}.txt'.format(self.model_name))
        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=float)
        else:
            dB = self.generate_hash_code(database_loader, num_database, database_loader.batch_size)
            dB = dB.numpy()
        t_map = CalcMap(qB, dB, targeted_labels, database_labels.numpy())
        print('t_MAP(retrieval database): %3.5f' % (t_map))

    def train(self, train_loader, target_labels, train_labels, database_loader, database_labels, num_database,
              num_train, num_test, test_loader, test_labels):
        # L2 loss function
        criterion_l2 = torch.nn.MSELoss()

        self.generate_train_code(train_loader, num_train, percent_training=self.args.percent)

        # Optimizers
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        self.optimizers = [optimizer_g]
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]

        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        # total_epochs = 150
        for epoch in range(self.args.epoch_count, total_epochs):
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.lr))
            for i, data in enumerate(train_loader):
                real_input, batch_label, batch_ind = data
                real_input = set_input_images(real_input)
                batch_label = batch_label.cuda()

                select_index = np.random.choice(range(target_labels.size(0)), size=batch_label.size(0))
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index)).cuda()

                fake_g, _ = self.generator(real_input, batch_target_label.detach())

                # update G
                optimizer_g.zero_grad()

                reconstruction_loss = criterion_l2(fake_g, real_input)

                target_hashing_g = self.hashing_model((fake_g + 1) / 2)

                loss_neighbor = data_loss_pos(target_hashing_g, batch_target_label, self.train_codes_fixed,
                                              self.train_labels_fixed, gamma=self.args.gamma_softmax)
                                              
                loss = self.rec_w * reconstruction_loss + loss_neighbor
                loss.backward()
                optimizer_g.step()

                if i % self.args.print_freq == 0:
                    print('step: {:3d} hash_loss: {:.3f} r_loss: {:.7f}'
                        .format(i, loss_neighbor, reconstruction_loss))

            self.update_learning_rate()
        self.save_generator()

    def save_generator(self):
        torch.save(self.generator.module.state_dict(),
            os.path.join(self.args.save, 'generator_{}_{}_{}_round{}.pth'.format(self.model_name, self.rec_w, self.dis_w, str(self.args.round))))

    def load_generator(self):
        self.generator.module.load_state_dict(
            torch.load(os.path.join(self.args.save, 'generator_{}_{}_{}_round{}.pth'.format(self.model_name, self.rec_w, self.dis_w, str(self.args.round)))))

    def load_model(self):
        self.load_generator()


    def cross_network_test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
        self.hashing_model.eval()
        # self.prototype_net.eval()
        self.generator.eval()
        qB = np.zeros([num_test, self.t_bit])
        if os.path.exists(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit))):
            targeted_labels = np.loadtxt(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit)))
        else:
            targeted_labels = np.zeros([num_test, self.num_classes])

        perceptibility = 0
        start = time.time()
        for it, data in enumerate(test_loader):
            data_input, data_label, data_ind = data

            if not os.path.exists(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit))):
                select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
                targeted_labels[data_ind.numpy(), :] = batch_target_label.numpy()
            else:
                batch_target_label = torch.tensor(targeted_labels[data_ind.numpy(), :]).float()

            

            data_input = set_input_images(data_input)
            # feature = self.prototype_net(batch_target_label.cuda())[0]
            target_fake, mix_image = self.generator(data_input, batch_target_label.cuda())
            target_fake = (target_fake + 1) / 2
            data_input = (data_input + 1) / 2 

            perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)

            target_hashing = self.hashing_model(target_fake)
            qB[data_ind.numpy(), :] = torch.sign(target_hashing.cpu().data).numpy()

        end = time.time()
        print('Running time: %s Seconds'%(end-start))
        self.bit = self.t_bit
        dB = self.generate_hash_code(database_loader, num_database, database_loader.batch_size)
        dB = dB.numpy()
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        t_map = CalcMap(qB, dB, targeted_labels, database_labels.numpy())
        print('t_MAP(retrieval database): %3.5f' % (t_map))
    
    def test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test):
        # self.prototype_net.eval()
        self.generator.eval()
        qB = np.zeros([num_test, self.bit])
        targeted_labels = np.zeros([num_test, self.num_classes])


        if os.path.exists(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit))):
            targeted_labels = np.loadtxt(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit)))
        else:
            targeted_labels = np.zeros([num_test, self.num_classes])
        perceptibility = 0
        
        for it, data in enumerate(test_loader):
            data_input, _, data_ind = data
            if not os.path.exists(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit))):
                select_index = np.random.choice(range(target_labels.size(0)), size=data_ind.size(0))
                batch_target_label = target_labels.index_select(0, torch.from_numpy(select_index))
                targeted_labels[data_ind.numpy(), :] = batch_target_label.numpy()
            else:
                batch_target_label = torch.tensor(targeted_labels[data_ind.numpy(), :]).float()

            data_input = set_input_images(data_input)
            target_fake, mix_image = self.generator(data_input, batch_target_label.cuda())
            target_fake = (target_fake + 1) / 2
            data_input = (data_input + 1) / 2 

            target_hashing = self.hashing_model(target_fake)
            qB[data_ind.numpy(), :] = torch.sign(target_hashing.cpu().data).numpy()

            perceptibility += F.mse_loss(data_input, target_fake).data * data_ind.size(0)


        if not os.path.exists(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit))):
            np.savetxt(os.path.join('log', 'target_label_{}_gan_{}.txt'.format(self.args.dataset, self.bit)), targeted_labels, fmt="%d")
        database_code_path = os.path.join('log', 'xdatabase_code_{}.txt'.format(self.model_name))
        if os.path.exists(database_code_path):
            dB = np.loadtxt(database_code_path, dtype=float)
        else:
            dB = self.generate_hash_code(database_loader, num_database, database_loader.batch_size)
            dB = dB.numpy()
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))
        t_map = CalcMap(qB, dB, targeted_labels, database_labels.numpy())
        print('t_MAP(retrieval database): %3.5f' % (t_map))
        map_ = CalcTopMap(qB, dB, targeted_labels, database_labels.numpy(), 5000)
        print('Test_MAP(retrieval database): %3.5f' % (map_))

    def transfer_test(self, target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test, args):
        self.target_model_path = '{}_{}_{}_{}'.format(args.dataset, args.t_hash_method, args.t_backbone, args.t_bit)
        if 'CSQ' in self.target_model_path:
            hashing_model = torch.load(
            os.path.join(self.args.save, self.target_model_path + '.pth'))
            if 'AlexNet' in self.target_model_path:
                model = AlexNet(self.args.t_bit)
            elif 'VGG' in self.target_model_path:
                model = VGG(self.args.t_backbone, self.args.t_bit)
            elif 'ResNet' in self.target_model_path:
                model = ResNet(self.args.t_backbone, self.args.t_bit)
            model.load_state_dict(hashing_model)
            self.hashing_model = model.cuda()
        else:
            hashing_model = torch.load(
            os.path.join(self.args.save, self.target_model_path + '.pth'))
            self.hashing_model = hashing_model.cuda()
        self.t_bit = self.args.t_bit
        self.cross_network_test(target_labels, database_loader, test_loader, database_labels, test_labels, num_database, num_test)
