import torch
from users.user_base import User
import torch.nn.functional as F
from networks import G, FTNet
import os
from ray import tune
from utils.loss_utils import HLoss, MMD_loss, NonSaturatingLoss
import math

class FeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.F = input[0]
        if len(self.F.shape)>2:
            self.F = self.F.squeeze(-1).squeeze(-1)

    def close(self):
        self.hook.remove()

class UserFRAug(User):
    def __init__(self, config, id, model):
        super().__init__(config, id, model)

        self.distance_metric_type_G = config['distance_metric_G']
        self.distance_metric_type_FT = config['distance_metric_FT']
        self.cls_criterion_type_FT = config['cls_criterion_type_FT']
        self.init_loss_fn_FA()

        self.class_num = config['class_num']
        self.global_epochs = config['global_epochs']

        ############################### init for Local Model ######################
        for name, module in self.model.named_modules():
            if ('fc'==name and 'resnet' in self.model_type):
                self.classifier = self.model.fc
                self.ModelFeature = FeatureHook(module)
            elif ('classifier'==name and not 'resnet' in self.model_type):
                self.classifier = self.model.classifier
                self.ModelFeature = FeatureHook(module)

        self.optimizer=torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate, momentum=config['local_momentum'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)

        ############################### init for Generator ######################
        if 'latent_dim' in config.keys():
            self.latent_dim = config['latent_dim']
        else:
            self.latent_dim = int(self.feature_dim/2)
        self.G = G(self.feature_dim, self.latent_dim, self.class_num).to(self.device)
        self.optimizer_G = torch.optim.SGD(params=self.G.parameters(), lr=config['lr_G'], momentum=config['local_momentum'])
        self.L_G_dist_lambda = config['L_G_dist_lambda']
        self.steps_using_G = config['steps_using_G']

        ######################### init for Feature Transformation Network #################
        self.FTNet = FTNet(self.feature_dim, self.latent_dim, self.class_num, config['FTNet_with_BN'], config['FTNet_with_cls']).to(self.device)
        self.optimizer_FTNet = torch.optim.SGD(params=self.FTNet.parameters(), lr=config['lr_FTNet'], momentum=config['local_momentum'])
        self.L_FT_dist_lambda = config['L_FT_dist_lambda']

        self.cls_mean_feature = torch.zeros((self.class_num, self.feature_dim), requires_grad=False).to(self.device)
        self.cls_mean_momentum = config['cls_mean_momentum']
        self.start_add_G = config['start_add_G']

        self.use_G = config['use_G']
        self.use_FTNet = config['use_FTNet']
        self.use_homo_label_aug = config['use_homo_label_aug']
        self.use_feature_inter = config['use_feature_inter']

        self.inter_gamma = config['inter_gamma']
        self.inter_form = config['inter_form']

    def init_loss_fn_FA(self):

        if self.cls_criterion_type_FT=='entropy':
            self.cls_criterion_FT = HLoss()
        elif self.cls_criterion_type_FT=='nonsatCE':
            self.cls_criterion_FT = NonSaturatingLoss(num_classes=self.class_num, epsilon=0.1)

        if self.distance_metric_type_FT == 'L1':
            self.distance_metric_FT = torch.nn.L1Loss()
        elif self.distance_metric_type_FT == 'L2':
            self.distance_metric_FT = torch.nn.MSELoss()
        elif self.distance_metric_type_FT.upper() == 'MMD':
            self.distance_metric_FT = MMD_loss()
        elif self.distance_metric_type_FT.upper() == 'MMD_KERNEL':
            self.distance_metric_FT = MMD_loss(kernel_type='gaussian')

        if self.distance_metric_type_G == 'L1':
            self.distance_metric_G = torch.nn.L1Loss()
        elif self.distance_metric_type_G == 'L2':
            self.distance_metric_G = torch.nn.MSELoss()
        elif self.distance_metric_type_G.upper() == 'MMD':
            self.distance_metric_G = MMD_loss()
        elif self.distance_metric_type_G.upper() == 'MMD_KERNEL':
            self.distance_metric_G = MMD_loss(kernel_type='gaussian')

    def update_cls_mean_feature(self, f, y_real):
        y_one_hot = F.one_hot(y_real.view((y_real.shape[0],)), num_classes=self.class_num).to(self.device).float()
        y_count = y_one_hot.sum(0).unsqueeze(1)
        f_cls = (y_one_hot.T @ f) / (y_count + torch.eq(y_count, 0.))

        momentum = torch.ne(y_count, 0.) * self.cls_mean_momentum
        self.cls_mean_feature = self.cls_mean_feature * (1-momentum) + f_cls * momentum

    def train(self, com_round, personalized=False, lr_decay=False):
        local_metrics = {}
        cls_list = list(range(self.class_num))

        if self.inter_form=='avg':
            ratio = 0.5
        elif self.inter_form=='exp':
            ratio = math.exp(self.inter_gamma*(com_round-self.global_epochs))
        elif self.inter_form=='1overx':
            ratio = 1 - 1 / (self.inter_gamma*com_round + 1)

        f_ratio = ratio
        # prevent overfit to the synthetic ones
        L_ratio = min(0.5, ratio)

        for it in range(self.local_epochs):
            try:
            # Samples a new batch for personalizing
                (X, y_real) = next(self.iter_trainloader)
            except:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.train_loader)
                (X, y_real) = next(self.iter_trainloader)
            if X.shape[0]==1: continue

            batch_size = X.shape[0]
            local_metrics = {}
            if com_round>self.start_add_G:
                self.use_G = True

            ################# Update Local Model  ######################
            self.G.eval()
            self.FTNet.eval()
            self.model.train()
            self.optimizer.zero_grad()

            o_real = self.model(X)
            f_real = self.ModelFeature.F
            self.update_cls_mean_feature(f_real.clone().detach(), y_real)
            L_S_ce_real = self.loss(o_real, y_real)
            local_metrics[f'loss_S_{self.id}_CE_real'] = float(L_S_ce_real)

            if self.use_G:
                L_S_ce_G = 0
                for st in range(self.steps_using_G):
                    if self.use_homo_label_aug==1 or self.use_homo_label_aug==2:
                        with torch.no_grad():
                            split = ((com_round*self.local_epochs + it*self.steps_using_G + st) * batch_size) % self.class_num
                            cls = cls_list[split:] + cls_list[:split]
                            y_fake = torch.LongTensor(cls * (batch_size//self.class_num) + cls[:batch_size%self.class_num]).to(self.device)
                            f_cls_mean = self.cls_mean_feature[y_fake]
                            z = torch.rand((batch_size, self.latent_dim), dtype=torch.float32, device=self.device)
                            f_fake = self.G(z, y_fake)

                            if self.use_FTNet:
                                f_fake = self.FTNet(f_fake, f_cls_mean, y_fake)

                            if self.use_feature_inter:
                                f_fake = f_fake*f_ratio + f_cls_mean*(1-f_ratio)

                        o_fake = self.classifier(f_fake)
                        L_S_ce_G += self.loss(o_fake, y_fake)

                    if self.use_homo_label_aug==0 or self.use_homo_label_aug==2:
                        with torch.no_grad():
                            z = torch.rand((batch_size, self.latent_dim), dtype=torch.float32, device=self.device)
                            f_fake = self.G(z, y_real)
                            if self.use_FTNet:
                                f_fake = self.FTNet(f_fake, f_real, y_real)

                            if self.use_feature_inter:
                                f_fake = f_fake*f_ratio + f_real*(1-f_ratio)

                        o_fake = self.classifier(f_fake)
                        L_S_ce_G += self.loss(o_fake, y_real)

                if self.use_homo_label_aug==2:
                    L_S_ce_G /= 2

                L_S_ce = L_S_ce_G  / self.steps_using_G  * L_ratio + L_S_ce_real * (1-L_ratio)

                local_metrics[f'loss_S_{self.id}_CE_G'] = float(L_S_ce_G/self.steps_using_G)
            else:
                L_S_ce = L_S_ce_real

            L_S_ce.backward()
            self.optimizer.step()
            local_metrics[f'loss_S_{self.id}_CE'] = float(L_S_ce)

            ############################ Generator Training ###################################
            if self.use_G:
                self.model.eval()
                self.G.train()
                self.optimizer_G.zero_grad()

                #Check which one to use!!!!!!
                split = ((com_round*self.local_epochs + it) * batch_size) % self.class_num
                cls = cls_list[split:] + cls_list[:split]
                y_fake = torch.LongTensor(cls * (batch_size//self.class_num) + cls[:batch_size%self.class_num]).to(self.device)

                z = torch.rand((batch_size, self.latent_dim), dtype=torch.float32, device=self.device, requires_grad=True)
                f_fake = self.G(z, y_fake)
                # minimize the CE loss for fake features
                o_fake = self.classifier(f_fake)
                L_G_CE = self.loss(o_fake, y_fake)

                # maximize the distance between real and fake
                if 'MMD' in self.distance_metric_type_G:
                    with torch.no_grad():
                        _ = self.model(X)
                        f_real = self.ModelFeature.F
                    L_G_dist = self.distance_metric_G(f_fake, f_real)
                else:
                    f_cls_mean = self.cls_mean_feature[y_fake]
                    L_G_dist = self.distance_metric_G(f_fake, f_cls_mean)

                L_G = L_G_CE + (-1)*self.L_G_dist_lambda*L_G_dist
                L_G.backward()
                self.optimizer_G.step()

                local_metrics[f'loss_G_{self.id}_dist'] = float(L_G_dist)
                local_metrics[f'loss_G_{self.id}_CE'] = float(L_G_CE)

                if self.use_FTNet:
                    self.model.eval()
                    self.G.eval()
                    self.FTNet.train()
                    self.optimizer_FTNet.zero_grad()
                    L_FT_dist = 0
                    L_FT_CE = 0

                    # minimize the distance between real and fake_trans
                    if self.use_homo_label_aug==1 or self.use_homo_label_aug==2:
                        with torch.no_grad():
                            z = torch.rand((batch_size, self.latent_dim), dtype=torch.float32, device=self.device, requires_grad=True)
                            f_fake = self.G(z, y_fake)
                            f_cls_mean = self.cls_mean_feature[y_fake]

                        f_fake_trans = self.FTNet(f_fake.clone().detach(), f_cls_mean, y_fake)
                        o_fake_trans = self.classifier(f_fake_trans)

                        if self.cls_criterion_type_FT=='entropy':
                            L_FT_CE += (-1) * self.cls_criterion_FT(o_fake_trans)
                        elif self.cls_criterion_type_FT=='nonsatCE':
                            L_FT_CE += self.cls_criterion_FT(o_fake_trans, y_fake)

                        L_FT_dist += self.distance_metric_FT(f_fake_trans, f_cls_mean)

                    if self.use_homo_label_aug==0 or self.use_homo_label_aug==2:
                        with torch.no_grad():
                            _ = self.model(X)
                            f_real = self.ModelFeature.F
                            z = torch.rand((batch_size, self.latent_dim), dtype=torch.float32, device=self.device, requires_grad=True)
                            f_fake = self.G(z, y_real)

                        f_fake_trans = self.FTNet(f_fake.clone().detach(), f_real, y_real)
                        o_fake_trans = self.classifier(f_fake_trans)

                        if self.cls_criterion_type_FT=='entropy':
                            L_FT_CE += (-1) * self.cls_criterion_FT(o_fake_trans)
                        elif self.cls_criterion_type_FT=='nonsatCE':
                            L_FT_CE += self.cls_criterion_FT(o_fake_trans, y_real)

                        L_FT_dist += self.distance_metric_FT(f_fake_trans, f_real)

                    if self.use_homo_label_aug==2:
                        L_FT_CE /= 2
                        L_FT_dist /= 2

                    L_FT = L_FT_CE + self.L_FT_dist_lambda*L_FT_dist
                    L_FT.backward()
                    self.optimizer_FTNet.step()

                    local_metrics[f'loss_FT_{self.id}_CE'] = float(L_FT_CE)
                    local_metrics[f'loss_FT_{self.id}_dist'] = float(L_FT_dist)

            if it%5==0:
                tune.report(**local_metrics)

        if lr_decay:
            self.lr_scheduler.step()

        loss_path = self.model_path.replace('user', 'user_train_loss').replace('pt', 'txt')
        with open(loss_path, 'a+') as f:
            f.write(f"{local_metrics[f'loss_S_{self.id}_CE']} \n")

    def save_G(self):
        torch.save(self.G, self.model_path.replace('user', 'user_G'))
    def save_FTNet(self):
        torch.save(self.FTNet, self.model_path.replace('user', 'user_FTNet'))

    def load_G(self):
        #load_from_server
        self.G = torch.load(self.server_model_path.replace('server', 'server_G'))


