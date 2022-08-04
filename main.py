#%%
# import libraries
import argparse
import datetime
import os
import torch
import time
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.distances import DotProductSimilarity

from utils.dataset import Prototype_Dataset
from utils.losses import KoLeoLoss
from utils.model import get_ViTImageEncoder

#%%
# setup directories
if __name__ == '__main__':
    if not os.path.exists('./models'):
        os.mkdir('./models')
    if not os.path.exists('./data'):
        os.mkdir('./data')

#%%
# settings and hyperparameters
if __name__ == '__main__':
    # argument parser (for command line arguments)
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    # training
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--reg_lambda', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--neg_ratio', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    # model
    parser.add_argument('--neg_margin', type=float, default=0.5)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default=str(int(time.time())))
    # environment settings
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--eval_epochs', type=int, default=5)
    parser.add_argument('--use_half', type=bool, default=True)
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()
    
    # create/load model
    model_dir = './models/' + args.model_name
    model_config_path = model_dir + '/model_config.txt'
    train_log_path = model_dir + '/train_log.csv'
    checkpoints_dir = model_dir + '/checkpoints'
    embeddings_dir = model_dir + '/embeddings'
    
    if os.path.exists(model_dir):
        print('\nLoading model from {}'.format(model_dir))
        # read model config to args
        with open(model_config_path, 'r') as f:
            for line in f:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if key in args:
                    type_fn = type(getattr(args, key))
                    args.__setattr__(key, type_fn(value))
    else:
        print('\nCreating model directory {}'.format(model_dir))
        os.mkdir(model_dir)
        os.mkdir(checkpoints_dir)
        os.mkdir(embeddings_dir)
        # write model config to args
        with open(model_config_path, 'w') as f:
            for key, value in vars(args).items():
                f.write(key + '=' + str(value) + '\n')
    
    if args.cuda:
        CUDA_AVAILABLE = torch.cuda.is_available()
    else:
        CUDA_AVAILABLE = False
        
    # print args
    print('\n', args, '\n')

#%%
# initialize dataloader (blister)
if __name__ == '__main__':
    # data transform for blister images
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR),
        torchvision.transforms.RandomAffine(8, translate=(.15, .15)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]), ])

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if CUDA_AVAILABLE else {}
    kwargs['batch_size'] = args.batch_size
    kwargs['shuffle'] = True
    
    train_dataset =\
        torchvision.datasets.ImageFolder(root=args.train_path, transform=data_transform)
    train_loader = DataLoader(train_dataset, **kwargs)

    test_dataset = \
        torchvision.datasets.ImageFolder(root=args.test_path, transform=data_transform)
    test_loader = DataLoader(test_dataset, **kwargs)

#%%
# train, evaluation functions
def train(model, optimizer, data_loader, contr_loss, koleo_loss, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    if CUDA_AVAILABLE:
        model = model.cuda()

    with tqdm(total=total_samples, desc='Training') as t:
        loss_history_epoch = []
        for i, (imgs, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            
            if CUDA_AVAILABLE:
                imgs = imgs.cuda()
            if args.use_half:
                imgs = imgs.half()
            
            img_embeds = model(imgs).float()
            
            loss = contr_loss(img_embeds, labels) + args.reg_lambda * koleo_loss(img_embeds)
            
            loss.backward()
            optimizer.step()

            loss_history_epoch.append(loss.item())
            with torch.no_grad():
                t.set_postfix(loss='{:.4f}'.format(loss.item()))
            t.update(imgs.shape[0])
        loss_epoch = np.mean(loss_history_epoch)
        t.set_postfix(loss='{:.4f}'.format(loss_epoch))
    
    loss_history.append(loss_epoch)
    
def evaluate(model, data_loader, contr_loss, koleo_loss, loss_history):
    total_samples = len(data_loader.dataset)
    model.eval()
    if CUDA_AVAILABLE:
        model = model.cuda()

    with tqdm(total=total_samples, desc='Evaluating triplet') as t:
        loss_history_epoch = []
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(data_loader):
                if CUDA_AVAILABLE:
                    imgs = imgs.cuda()
                if args.use_half:
                    imgs = imgs.half()
                    
                img_embeds = model(imgs).float()
                loss = contr_loss(img_embeds, labels) + args.reg_lambda * koleo_loss(img_embeds)

                loss_history_epoch.append(loss.item())
                t.set_postfix(loss='{:.4f}'.format(loss.item()))
                t.update(imgs.shape[0])
                
        loss_epoch = np.mean(loss_history_epoch)
        t.set_postfix(loss='{:.4f}'.format(loss_epoch))
    
    loss_history.append(np.mean(loss_epoch))

# get class embedding (dict) by averaging embeddings of all images in the class
def get_class_embed(model, dataset):
    train_labels = np.array([item[1] for item in dataset.imgs])
    labels_set = set(train_labels)
    prototype_dataset = Prototype_Dataset(dataset, len(labels_set))
    
    model.eval()
    if CUDA_AVAILABLE:
        model = model.cuda()
    
    cls_idx_2_embed = {}
    with tqdm(total=len(labels_set), desc='Getting class embeddings') as t:
        for cls_idx in labels_set:
            imgs = torch.stack(prototype_dataset[cls_idx])
            
            with torch.no_grad():
                if CUDA_AVAILABLE:
                    imgs = imgs.cuda()
                if args.use_half:
                    imgs = imgs.half()
                output = model(imgs)
                output = output.mean(dim=0)
            cls_idx_2_embed[cls_idx] = output
            t.update()

    return cls_idx_2_embed

# evaluate model in classification task
def evaluate_classification(model, dataset, data_loader: DataLoader, acc_history, cls_idx_2_embed = None, desc='Evaluating classification'):
    model.eval()
    if CUDA_AVAILABLE:
        model = model.cuda()
    
    if cls_idx_2_embed is None:
        cls_idx_2_embed = get_class_embed(model, dataset)
    
    num_pred = 0
    num_correct = 0
    acc = 0

    img_descriptors = []
    img_labels = []
    
    total_samples = len(data_loader.dataset)
    with tqdm(total=total_samples, desc=desc) as t:
        for _, (imgs, labels) in enumerate(data_loader):
            if args.use_half:
                imgs = imgs.half()
            if CUDA_AVAILABLE:
                imgs = imgs.cuda()
            
            with torch.no_grad():
                outputs = model(imgs)
            
            cls_embed = torch.stack(list(cls_idx_2_embed.values()))
            index_2_cls_label = {i: v for i, v in enumerate(list(cls_idx_2_embed.keys()))}
            distances = torch.cdist(outputs.float(), cls_embed.float(), p=2)
            
            pred_idx = distances.argmin(dim=1).cpu()
            pred_labels = pred_idx.apply_(index_2_cls_label.get)
            results = torch.eq(pred_labels, labels)
            
            num_pred += len(pred_labels)
            num_correct += results.sum().item()
            acc = num_correct / num_pred

            img_descriptors.extend(outputs.cpu().numpy())
            img_labels.extend(labels.cpu().numpy())
            
            t.update(imgs.shape[0])
            t.set_postfix(accuracy='{:.4f}'.format(acc))
    
    # if SAVE_EMBED:
    #     # save embeddings to csv
    #     img_descriptors = np.array(img_descriptors)
    #     img_labels = np.array(img_labels)
    #     img_descriptors_df = pd.DataFrame(img_descriptors)
    #     img_labels_df = pd.DataFrame(img_labels)
    #     img_descriptors_df.to_csv('./img_descriptors.tsv', sep='\t', index=False)
    #     img_labels_df.to_csv('./img_labels.tsv', sep='\t', index=False)

    acc_history.append(acc)
    
#%%
# initialize model
if __name__ == '__main__':
    model = get_ViTImageEncoder(pretrained=args.pretrained)
    if args.use_half:
        model = model.half()
    contr_loss = ContrastiveLoss(pos_margin=1, neg_margin=args.neg_margin, distance=DotProductSimilarity())
    koleo_loss = KoLeoLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

#%%
# run
if __name__ == '__main__':
    # read loss history from csv with pandas if exists
    train_loss_history, test_loss_history = [], []
    train_acc_history, test_acc_history = [], []
    if os.path.exists(train_log_path):
        train_log_df = pd.read_csv(train_log_path)
        train_loss_history = train_log_df['train_loss'].tolist()
        test_loss_history = train_log_df['test_loss'].tolist()
        train_acc_history = train_log_df['train_acc'].tolist()
        test_acc_history = train_log_df['test_acc'].tolist()
        
    if os.path.exists(model_dir + '/tmp.pt'):
        print('Loading model from ' + model_dir + '/tmp.pt')
        model.load_state_dict(torch.load(model_dir + '/tmp.pt'))
    else:
        print('Model not found at ' + model_dir + '/tmp.pt')
    
    # train/evaluate loop
    epoch = len(train_loss_history) + 1
    while epoch < args.n_epochs + 1:
        print('Epoch:{}/{}'.format(epoch, args.n_epochs))
        train(model, optimizer, train_loader, contr_loss, koleo_loss, train_loss_history)
        
        # evaluation with triplet loss and classification task
        if epoch % args.eval_epochs == 0:
            # evaluate with triplet loss
            evaluate(model, test_loader, contr_loss, koleo_loss, test_loss_history)
            
            # evaluate in classification task
            cls_2_embed = get_class_embed(model, train_dataset)
            evaluate_classification(model, None, train_loader, train_acc_history, cls_2_embed, desc='Evaluating classification (train)')
            evaluate_classification(model, None, test_loader, test_acc_history, cls_2_embed, desc='Evaluating classification (test)')
        else: 
            test_loss_history.append(None)
            train_acc_history.append(None)
            test_acc_history.append(None)
        
        # save loss history to csv with pandas
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_df = pd.DataFrame({'time': timestamp, 'train_loss': train_loss_history, 'test_loss': test_loss_history, 'train_acc': train_acc_history, 'test_acc': test_acc_history})
        train_log_df.to_csv(train_log_path)

        torch.save(model.state_dict(), model_dir + '/tmp.pt')
        print('Saved model to ' + model_dir + '/tmp.pt')
        
        # save model
        if epoch % args.save_epochs == 0:
            torch.save(model.state_dict(), checkpoints_dir + '/model_' + str(epoch) + '_epochs.pt')
            print('Saved model checkpoint to ' + checkpoints_dir + '/model_' + str(epoch) + '_epochs.pt')

        epoch += 1
        
#%%