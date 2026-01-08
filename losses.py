import torch
import torch.nn as nn
    
class SEHALoss(nn.Module):
    def __init__(self, noisy_labels, loss_type='', momentum=0.99):
        super(SEHALoss, self).__init__()
        self.loss_type      = loss_type
        self.momentum       = momentum
        
        self.soft_labels1   = torch.Tensor(noisy_labels).cuda()
        self.soft_labels2   = torch.Tensor(noisy_labels).cuda()
    
    def intra_modal_loss(self, predict1, predict2, index, labels, epoch, opt):
        if epoch <= opt.tp:
            if opt.loss_type == 'CE':
                tmp1 = - (labels * predict1.log()).sum(1)
                tmp2 = - (labels * predict2.log()).sum(1)
            elif opt.loss_type == 'GCE':
                q = min(1., 0.01 * (epoch + 1))
                tmp1 = (1 - q) * (1. - torch.sum(labels.float() * predict1, dim=1) ** q).div(q)
                tmp2 = (1 - q) * (1. - torch.sum(labels.float() * predict2, dim=1) ** q).div(q)
            elif opt.loss_type == 'RC':
                tmp1 = torch.log(1-(labels * predict1)).sum(1)
                tmp2 = torch.log(1-(labels * predict2)).sum(1)
        else:
            self.soft_labels1[index] = self.momentum * self.soft_labels1[index] + \
                                        (1 - self.momentum) * predict1.detach()
            self.soft_labels2[index] = self.momentum * self.soft_labels2[index] + \
                                        (1 - self.momentum) * predict2.detach()
                                        
            tmp1 = -torch.sum(torch.log(predict1) * self.soft_labels1[index], dim=1)
            tmp2 = -torch.sum(torch.log(predict2) * self.soft_labels2[index], dim=1)
            
            tmp3 = -torch.sum(torch.log(predict1) * self.soft_labels2[index], dim=1)
            tmp4 = -torch.sum(torch.log(predict2) * self.soft_labels1[index], dim=1)
            tmp1 = tmp1 + tmp3
            tmp2 = tmp2 + tmp4
        
        return (tmp1 + tmp2).mean()
    
    def inter_modal_loss(self, features, predicts, tau=1., q=1.):
        n_view = len(features)
        batch_size = features[0].shape[0]
        all_features = torch.cat(features, dim=0)
        all_predicts = torch.cat(predicts, dim=0)
        
        similarity = all_features @ all_features.t()
        similarity = torch.exp(similarity / tau)
        similarity = similarity - torch.diag(similarity.diag())
        
        view_sums = [similarity[:, v*batch_size : (v+1)*batch_size] for v in range(n_view)]
        sim_sum1 = sum(view_sums)
        diag1 = torch.cat([sim_sum1[v*batch_size : (v+1)*batch_size].diag() for v in range(n_view)])
        p1 = diag1 / similarity.sum(dim=1)
        loss1 = -torch.log(p1 + 1e-8)
        
        view_rows = [similarity[v*batch_size : (v+1)*batch_size, :] for v in range(n_view)]
        sim_sum2 = sum(view_rows)
        diag2 = torch.cat([sim_sum2[:, v*batch_size : (v+1)*batch_size].diag() for v in range(n_view)])
        p2 = diag2 / similarity.sum(dim=1)
        loss2 = -torch.log(p2 + 1e-8)
        
        similarity = all_predicts @ all_predicts.t()
        similarity = torch.exp(similarity / tau)
        similarity = similarity - torch.diag(similarity.diag())
        
        view_sums = [similarity[:, v*batch_size : (v+1)*batch_size] for v in range(n_view)]
        sim_sum3 = sum(view_sums)
        diag3 = torch.cat([sim_sum3[v*batch_size : (v+1)*batch_size].diag() for v in range(n_view)])
        p3 = diag3 / similarity.sum(dim=1)
        loss3 = -torch.log(p3 + 1e-8) * 0.01
        
        view_rows = [similarity[v*batch_size : (v+1)*batch_size, :] for v in range(n_view)]
        sim_sum4 = sum(view_rows)
        diag4 = torch.cat([sim_sum4[:, v*batch_size : (v+1)*batch_size].diag() for v in range(n_view)])
        p4 = diag4 / similarity.sum(dim=1)
        loss4 = -torch.log(p4 + 1e-8) * 0.01
        
        return (loss1 + loss2 + loss3 + loss4).mean()
    
    def forward(self, features1, features2, predict1, predict2, index, labels=None, epoch=0, opt=None):
        term1 = self.intra_modal_loss(predict1, predict2, index, labels, epoch, opt)
        term2 = self.inter_modal_loss([features1, features2], [predict1, predict2], tau=opt.tau, q=getattr(opt, 'q', 1.0))
                
        return term1 + opt.lamda * term2