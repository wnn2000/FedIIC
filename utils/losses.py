import torch
import torch.nn as nn
import torch.nn.functional as F



class DALA(nn.Module):
    def __init__(self, cls_num_list, cls_loss, epoch, tau=1, weight=None, args=None):
        super(DALA, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        cls_loss = cls_loss.cuda()
        
        t = cls_p_list / (torch.pow(cls_loss, args.d)+1e-5)
        m_list = tau * torch.log(t)

        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class IntraSCL(nn.Module):
    def __init__(self, cls_num_list, temperature=0.1):
        super(IntraSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = torch.tensor(cls_num_list).cuda()
        self.cls_num_list = self.cls_num_list / torch.sum(self.cls_num_list)

    def forward(self, features, targets):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets = targets.repeat(2, 1)
        mask = torch.eq(targets, targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = features.mm(features.T)
        temp = self.cls_num_list.gather(0, targets.view(-1)).view(-1, 1)
        temp = temp.mm(temp.T)
        temp = torch.clamp(temp**0.5, min=0.07)
        logits = torch.div(logits, temp)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()

        return loss


class InterSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(InterSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, centers1, features, targets):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(
            len(self.cls_num_list), device=device).view(-1, 1)
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)
        batch_cls_count = torch.eye(len(self.cls_num_list))[
            targets].sum(dim=0).squeeze()

        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        logits_mask[:2*batch_size, :2*batch_size] = 0
        mask = mask * logits_mask

        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)
        logits = features[:2 * batch_size].mm(features.T)
        logits = torch.div(logits, self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()

        return loss