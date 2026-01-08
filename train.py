import torch
import time
import copy
import logging
from pathlib import Path
import numpy as np
import random as rn
import torch.optim as optim
import torch.nn.functional as F
from model import CMNN, Embedding
from losses import SEHALoss
from load_data import get_loader
from evaluate import fx_calc_map_multilabel
from utils import get_training_args

def to_seed(seed=0):
    np.random.seed(seed)
    rn.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def setup_logging(args):
    log_dir = Path("results")
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / (
        f"{args.dataset}_{args.noise_mode}_"
        f"{args.noisy_ratio}_{args.bit}bit_log.txt"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', 
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_path

def train_model(model, emb, data_loaders, input_data_par, optimizer, configs):

    time_start = time.time()
    
    # Noisy Label Data
    clean_indexs = []
    noisy_indexs = []
    for imgs, txts, labels, ori_labels, index in data_loaders['train']:
        clean_index = np.argmax(labels, axis=1) == np.argmax(ori_labels, axis=1)
        clean_indexs.append(index[clean_index])
        noisy_index = np.argmax(labels, axis=1) != np.argmax(ori_labels, axis=1)
        noisy_indexs.append(index[noisy_index])
    clean_indexs = np.concatenate(clean_indexs)
    noisy_indexs = np.concatenate(noisy_indexs)
    clean_indexs = torch.tensor(clean_indexs, requires_grad=False).cuda()
    noisy_indexs = torch.tensor(noisy_indexs, requires_grad=False).cuda()
    clean_weights_list = []
    noisy_weights_list = []
    
    # Loss Function
    criterion = SEHALoss(noisy_labels=input_data_par['label_train'], loss_type=configs.loss_type, momentum=configs.momentum).cuda()
    
    # Training Record
    test_img_acc_history = []
    test_txt_acc_history = []
    mAP_history = []
    epoch_loss_history =[]
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Training Loop
    for epoch in range(configs.MAX_EPOCH):
        logging.info('\nEpoch {}/{}'.format(epoch, configs.MAX_EPOCH))
        logging.info('-' * 25)
        clean_weights = torch.zeros_like(clean_indexs, requires_grad=False).float().cuda()
        noisy_weights = torch.zeros_like(noisy_indexs, requires_grad=False).float().cuda()
        
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_corrects_img = 0.0
            running_corrects_txt = 0.0
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            for imgs, txts, labels, ori_labels, index in data_loaders[phase]:
                if torch.sum(imgs!=imgs) > 1 or torch.sum(txts!=txts) > 1:
                    logging.error("Data contains Nan.")

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        txts = txts.cuda()
                        labels = labels.cuda()
                        ori_labels = ori_labels.cuda()

                    optimizer.zero_grad()
                    W = emb(torch.eye(configs.data_class).cuda())
                    view1_feature, view2_feature = model(imgs, txts)
                    view1_predict = F.softmax(view1_feature.view([view1_feature.shape[0], -1]).mm(W.T), dim=1)
                    view2_predict = F.softmax(view2_feature.view([view2_feature.shape[0], -1]).mm(W.T), dim=1)

                    loss = criterion(view1_feature, view2_feature, view1_predict, view2_predict, 
                                     index, labels, epoch, configs)
                    
                    # if epoch >= configs.tp and phase == 'train' and configs.self_paced:
                    #     # clean
                    #     clean_index = torch.argmax(labels, dim=1) == torch.argmax(ori_labels, dim=1)
                    #     mask = torch.isin(clean_indexs, index[clean_index.cpu()].cuda())
                    #     clean_weights[torch.where(mask)[0]] = weight[clean_index]
                        
                    #     # noisy
                    #     noisy_index = torch.argmax(labels, dim=1) != torch.argmax(ori_labels, dim=1).int()
                    #     mask = torch.isin(noisy_indexs, index[noisy_index.cpu()].cuda())
                    #     noisy_weights[torch.where(mask)[0]] = weight[noisy_index]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item()
                clean_index = torch.argmax(labels, dim=1) == torch.argmax(ori_labels, dim=1)
                view1_predict = view1_predict[clean_index]
                view2_predict = view2_predict[clean_index]
                ori_labels = ori_labels[clean_index]
                running_corrects_img += torch.sum(torch.argmax(view1_predict, dim=1) == torch.argmax(ori_labels, dim=1))
                running_corrects_txt += torch.sum(torch.argmax(view2_predict, dim=1) == torch.argmax(ori_labels, dim=1))
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            t_imgs, t_txts, t_labels = [], [], []
            if phase == 'train':
                with torch.no_grad():
                    for imgs, txts, labels, ori_labels, index in data_loaders['valid']:
                        if torch.cuda.is_available():
                                imgs = imgs.cuda()
                                txts = txts.cuda()
                                labels = labels.cuda()
                        t_view1_feature, t_view2_feature = model(imgs, txts)
                        t_imgs.append(t_view1_feature.sign().cpu().numpy())
                        t_txts.append(t_view2_feature.sign().cpu().numpy())
                        t_labels.append(labels.cpu().numpy())
                t_imgs = np.concatenate(t_imgs)
                t_txts = np.concatenate(t_txts)
                t_labels = np.concatenate(t_labels)
                img2txt = fx_calc_map_multilabel(t_imgs, t_txts, t_labels, metric='hamming')
                txt2img = fx_calc_map_multilabel(t_txts, t_imgs, t_labels, metric='hamming')
                mAP_history.append((img2txt + txt2img) / 2.)
                clean_weights_list.append(clean_weights.detach().cpu().numpy())
                noisy_weights_list.append(noisy_weights.detach().cpu().numpy())

            logging.info(f"    - [{phase:<5}] Loss: {epoch_loss:>6.4f}  Img2Txt: {img2txt:>6.4f}  Txt2Img: {txt2img:>6.4f}  Lr: {optimizer.param_groups[0]['lr']:>6g}")

            if phase == 'valid' and (img2txt + txt2img) / 2. > best_acc:
                best_acc = (img2txt + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                test_img_acc_history.append(img2txt)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

    time_end = time.time()
    time_used = time_end - time_start
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_used // 60, time_used % 60))
    model.load_state_dict(best_model_wts)
    
    return model, mAP_history

def main():
    # Parse arguments
    args = get_training_args()
    setup_logging(args)
    logging.info(args)
    
    # Environmental setting
    device = torch.device("cuda:%d"%args.GPU if torch.cuda.is_available() else "cpu")
    to_seed(args.seed)
    
    # Data parameters
    logging.info("\n[SEHA]: Data loading starts...")
    dataset = args.dataset # IAPR MIRFlickr nuswide mscoco
    data_loader, input_data_par = get_loader(dataset, args.batch_size, args.noisy_ratio, args.noise_mode)
    args.data_class = input_data_par['num_class']
    logging.info('    - Train Numbers: {train:>4}  Valid Numbers: {valid:>4}  Test Numbers: {test:>4}  Classes Numbers: {classes:>4}'.format(
                 train=input_data_par["img_train"].shape[0], valid=input_data_par["img_valid"].shape[0], 
                 test=input_data_par["img_test"].shape[0], classes=input_data_par["label_train"].shape[1]))
    
    # Model parameters
    model_ft = CMNN(img_input_dim=input_data_par['img_dim'], 
                    text_input_dim=input_data_par['text_dim'], output_dim=args.bit, 
                    num_class=input_data_par['num_class']).to(device)

    emb = Embedding(args.data_class, args.bit).cuda()
    optimizer = optim.Adam([{'params': emb.parameters(), 'lr': args.lr},
                            {'params': model_ft.parameters(), 'lr': args.lr}])
    
    # Training
    logging.info("\n[SEHA]: Training starts...")
    model_ft, _ = train_model(model_ft, emb, data_loader, input_data_par, optimizer, args)

    # Evaluating
    view1_feature, view2_feature = model_ft(torch.tensor(input_data_par['img_test']).to(device), 
                                            torch.tensor(input_data_par['text_test']).to(device))
    
    label = input_data_par['label_test']
    view1_feature = view1_feature.sign().detach().cpu().numpy()
    view2_feature = view2_feature.sign().detach().cpu().numpy()

    # Performance on test set
    img_to_txt = fx_calc_map_multilabel(view1_feature, view2_feature, label, metric='hamming')
    txt_to_img = fx_calc_map_multilabel(view2_feature, view1_feature, label, metric='hamming')
    
    logging.info("\n[SEHA RESULT]:")
    logging.info("    - Image to Text MAP = {}".format(img_to_txt))
    logging.info("    - Text to Image MAP = {}".format(txt_to_img))
    logging.info("    - Average MAP = {}".format(((img_to_txt + txt_to_img) / 2.)))
    
if __name__ == '__main__':
    main()