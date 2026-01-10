import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results
from models.lnln import build_model
from core.metric import MetricsTop 


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)  # 保留原有打印

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--seed', type=int, default=-1) 
# 新增阶段参数：不影响原有打印（opt打印会包含该参数，属于合理扩展且不修改原有打印格式）
parser.add_argument('--stage', type=str, default='train', choices=['train', 'pretrain', 'finetune'])
opt = parser.parse_args()
print(opt)  # 保留原有打印

def main():
    best_valid_results, best_test_results = {}, {}

    config_file = 'configs/train_sims.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)  # 保留原有打印

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))  # 保留原有打印

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    # 新增预训练权重路径：不影响原有打印
    pretrain_ckpt = os.path.join(ckpt_root, f'pretrain_seed_{seed}.pth')
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("ckpt root :", ckpt_root)  # 保留原有打印

    model = build_model(args).to(device)

    # 双阶段训练逻辑：根据stage分支，完全保留原有训练流程的打印
    if opt.stage == 'pretrain':
        # 自监督预训练：新增分支，不影响原有打印
        pretrain(model, args, ckpt_root, seed)
    elif opt.stage == 'finetune':
        # 有监督微调：加载预训练权重，不影响原有打印
        if os.path.exists(pretrain_ckpt):
            model.load_state_dict(torch.load(pretrain_ckpt, map_location=device), strict=False)
        finetune(model, args, ckpt_root, seed)
    else:
        # 原有训练流程：完全保留，打印内容不变
        dataLoader = MMDataLoader(args)

        optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=args['base']['lr'],
                                     weight_decay=args['base']['weight_decay'])
        scheduler_warmup = get_scheduler(optimizer, args)

        loss_fn = MultimodalLoss(args)

        metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

        for epoch in range(1, args['base']['n_epochs']+1):
            train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)

            if args['base']['do_validation']:
                valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
                best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=False)
                print(f'Current Best Valid Results: {best_valid_results}')  # 保留原有打印

            test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
            best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=True)
            print(f'Current Best Test Results: {best_test_results}\n')  # 保留原有打印

            scheduler_warmup.step()

def pretrain(model, args, ckpt_root, seed):
    """自监督预训练阶段：新增函数，不影响原有打印逻辑"""
    dataLoader = MMDataLoader(args)
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args['base']['lr'],
                                 weight_decay=args['base']['weight_decay'])
    scheduler_warmup = get_scheduler(optimizer, args)
    loss_fn = MultimodalLoss(args)

    for epoch in range(1, args['base']['n_epochs']//2 + 1):
        model.train()
        loss_dict = {'loss': 0, 'l_mask': 0, 'l_match': 0}
        for cur_iter, data in enumerate(dataLoader['train']):
            complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
            incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

            # 生成跨模态匹配标签
            match_label = torch.ones(len(data['labels']['M'])).long().to(device)
            flip_idx = torch.randperm(len(match_label))[:len(match_label)//2]
            match_label[flip_idx] = 0
            label = {'match_label': match_label}

            out = model(complete_input, incomplete_input, is_pretrain=True)
            loss = loss_fn(out, label, is_pretrain=True)

            loss['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()

            for key in loss_dict.keys():
                loss_dict[key] += loss[key].item()

        loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}
        # 预训练打印为新增内容，不干扰原有打印逻辑
        print(f'Pretrain Loss Epoch {epoch}: {loss_dict}')
        scheduler_warmup.step()

    # 保存预训练权重
    torch.save(model.state_dict(), os.path.join(ckpt_root, f'pretrain_seed_{seed}.pth'))

def finetune(model, args, ckpt_root, seed):
    """有监督微调阶段：复用原有训练逻辑，保留所有打印"""
    best_valid_results, best_test_results = {}, {}
    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args['base']['lr'] * 0.1,
                                 weight_decay=args['base']['weight_decay'])
    scheduler_warmup = get_scheduler(optimizer, args)

    loss_fn = MultimodalLoss(args)

    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    for epoch in range(1, args['base']['n_epochs']+1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)

        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')  # 保留原有打印

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=True)
        print(f'Current Best Test Results: {best_test_results}\n')  # 保留原有打印

        scheduler_warmup.step()

def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    # 完全保留原有函数逻辑，不修改任何打印
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)
        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}

        out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item()
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item()

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')  # 保留原有打印
    print(f'Train Results Epoch {epoch}: {results}')  # 保留原有打印

def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    # 完全保留原有函数逻辑，不修改任何内容（包括注释的打印）
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()
    
    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)
        effectiveness_labels = torch.cat([torch.ones(len(sentiment_labels)*8), torch.zeros(len(sentiment_labels)*8)]).long().to(device)
        label = {'sentiment_labels': sentiment_labels, 'completeness_labels': completeness_labels, 'effectiveness_labels': effectiveness_labels}
        
        with torch.no_grad():
            out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item()
                except:
                    loss_dict[key] += value
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    
    # print(f'Test Loss Epoch {epoch}: {loss_dict}')
    # print(f'Test Results Epoch {epoch}: {results}')

    return results


if __name__ == '__main__':
    main()
