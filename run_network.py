import yaml
import random
import time
from dataset import LabelEmbeddingData
from utils.load_data import *
from utils.metrics import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.tppred import TPMLC, TPMLC_single
from torch.optim import AdamW
from utils.sampling import Sampler
# from utils.visualization import *
from sklearn.metrics import f1_score, accuracy_score
from loss_functions import *
import time

class Model ():

    def __init__(self, args):
        """
        initialize the hyper-parameters
        """

        self.args = args

        # Load constants
        with open(args.cfg, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        # self.model = cfg['model']
        self.d_fea = cfg['d_fea']
        self.max_len = cfg['max_len']
        self.pts = cfg['pts']

        # network parameters
        self.seed = args.seed
        self.d_model = args.dm
        self.n_heads = args.nh
        self.n_layers_enc = args.nle
        self.n_layers_dec = args.nld
        self.drop = args.drop

        # shared training parameters
        self.batch_size = args.b

        # jointly training parameters
        self.epochs = args.e
        self.lr = args.lr
        self.w = args.w
        self.model_path = args.pth

        # retraining parameters
        self.re_method = args.s
        self.re_epochs = args.e2
        self.re_lr = args.lr2
        self.re_w = args.w2
        self.re_model_path = args.pth2

        # other parameters
        self.dataset_dir = args.src
        self.task_tag = ""
        self.result_folder = args.result_folder

        # If training all layers, the trained model will saved to self.model_path.
        # If retraining the classifiers, method will load the model self.model_path,
        # and save the retrained model to self.re_model_path

        self.names = [pt[:-4] for pt in self.pts]
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.n_class = len(self.pts)

        self.pt2idx = {}
        for i, pt in enumerate(self.names):
            self.pt2idx[pt] = i

        self.set_seed(seed=self.seed)

    def set_task(self, task=None):

        self.task_tag = task + "_" if task is not None else ""


    def train_epoch(self, model, optimizer, criterion, train_dataloder, val_dataloder, target = None):

        model.train()
        train_losses = []

        for i, data in enumerate(train_dataloder):
            optimizer.zero_grad()

            X, y, masks, label_input = data

            out, _, _, _ = model(X, masks, label_input)
            # out = model(X, masks)

            if target == None:
                loss = criterion(out, y.float())
            else:
                loss = criterion(out[:, target], y.float()[:, target])

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validating the model after each step
        model.eval()
        val_losses = []
        y_pred = []
        y_true = []

        with torch.no_grad():

            for i, data in enumerate(val_dataloder):
                X, y, masks, label_input = data
                out, _, _, _ = model(X, masks, label_input)
                # out = model(X, masks)

                if target == None:
                    loss = criterion(out, y.float())
                else:
                    loss = criterion(out[:, target], y.float()[:, target])

                val_losses.append(loss.item())
                y_true.extend(y.cpu().numpy())
                y_pred.extend(out.cpu().detach().numpy())

        # print("Epoch {}, train loss = {}, validation loss = {}".
        #       format(epoch, np.mean(train_losses), np.mean(val_losses)))

        # optimized by validation loss

        return float(np.mean(train_losses)), float(np.mean(val_losses)), y_true, y_pred

    def retrain_classifiers(self):
        """
        Retraining each specific classifier layer
        """
        print(f"Retraining classifier layers, task: {self.task_tag}")

        checkpoint = torch.load(self.model_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load training and validation datasets
        train_feas, train_labels, train_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'train'), True, *self.pts, seed=self.seed)

        val_feas, val_labels, val_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'val'), True, *self.pts, seed=self.seed)
        val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device),
                                   batch_size=self.batch_size, shuffle=False)

        print('dataset',os.path.join(self.dataset_dir, 'train'))

        criterion = torch.nn.BCELoss()

        # Reinitialize classifiers
        self.reset_classifiers(model)

        best_model = None

        for i, fn in enumerate(self.pts):
            name = fn.split('.')[0]
            print("Retrain classifier", name)

            # Freeze the model layers except the i-th classifier
            self.freeze_layers(model, i)

            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), self.re_lr, weight_decay=self.re_w)

            min_loss = 10000
            max_f1 = 0

            for epoch in range(self.re_epochs):

                sampler = Sampler(train_labels, method=self.re_method, lam=epoch / (self.re_epochs))
                sampler.set_target(i)

                train_dataloader = DataLoader(
                    dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device),
                    batch_size=self.batch_size, sampler=sampler)

                train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloader, val_dataloder, target=i)

                print("Epoch {}, train loss = {}, validation loss = {}".
                      format(epoch, train_loss, val_loss))

                if val_loss <= min_loss:
                
                    print('update loss', val_loss)
                    best_model = model
                    min_loss = val_loss
                
                    self.evaluation(np.array(y_true), np.array(y_pred), 'val')

        if self.re_model_path is not None:
            self.save_model(best_model, self.re_model_path)


    def train_all(self):

        print("Training all layers2, task name: ", self.task_tag)

        # Load training and validation features
        train_feas, train_labels, train_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'train'), True, *self.pts, seed=self.seed)

        val_feas, val_labels, val_pad_masks, _ = load_features(os.path.join(self.dataset_dir, 'val'), True, *self.pts, seed=self.seed)

        train_dataloder = DataLoader(dataset=LabelEmbeddingData(train_feas, train_labels, train_pad_masks, self.device),
                                     batch_size=self.batch_size, shuffle=True)
        val_dataloder = DataLoader(dataset=LabelEmbeddingData(val_feas, val_labels, val_pad_masks, self.device),
                                   batch_size=self.batch_size, shuffle=False)
        
        # phase 1
        model = TPMLC_single(self.d_fea, self.n_class, self.max_len, self.d_model, device=self.device, nhead=self.n_heads,
                      n_enc_layers=self.n_layers_enc, n_dec_layers=self.n_layers_dec, dropout=self.drop).to(self.device)
       
        criterion = torch.nn.BCELoss()
        
        optimizer = AdamW(model.parameters(), self.lr, weight_decay=self.w)

        # optimized values
        min_loss = 1000
        best_model = None
        
        for epoch in range(self.epochs):

            train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloder,
                                                                    val_dataloder)

            print("Epoch {}, train loss = {}, validation loss = {}".
                  format(epoch, train_loss, val_loss))

            # optimized by validation loss
            if val_loss <= min_loss:
                best_model = model
                min_loss = val_loss
                

        # save the model with min validation loss
        sv = self.model_path[:-4] + '_single.pth'
        if self.model_path is not None:
            self.save_model(best_model, sv)
    
        # phase 2

        checkpoint = torch.load(sv)
        rp_model = checkpoint['model']
        rp_model.load_state_dict(checkpoint['model_state_dict'])

        model = TPMLC(self.d_fea, self.n_class, self.max_len, self.d_model, device=self.device, nhead=self.n_heads,
                             n_enc_layers=self.n_layers_enc, n_dec_layers=self.n_layers_dec, dropout=self.drop).to(
            self.device)
        model_dict = model.state_dict()

        st = {} 
        for k, v in rp_model.named_parameters():
            if k.startswith('rp') and k in model_dict.keys():
                st[k] = v

        model_dict.update(st)
        model.load_state_dict(model_dict)


        # optimized values
        min_loss = 1000
        best_model = None

        self.freeze_layers_dec(model)
   
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.2, reduction='mean')
          
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), self.re_lr, weight_decay=self.w)
        best_val_thresholds = None
        for epoch in range(self.re_epochs):

            train_loss, val_loss, y_true, y_pred = self.train_epoch(model, optimizer, criterion, train_dataloder,
                                                                    val_dataloder)

            print("Epoch {}, train loss = {}, validation loss = {}".
                  format(epoch, train_loss, val_loss))

            # optimized by validation loss
            if val_loss <= min_loss:
                best_model = model
                min_loss = val_loss
                
                best_val_thresholds = self.evaluation(np.array(y_true), np.array(y_pred), 
                                                      tag='val', metric='F1+Jaccard')
        # save the model with min validation loss
        if self.model_path is not None:
            self.save_model(best_model, self.model_path)
        

        print("\n" + "="*40)
        print("Training Finished. Running Independent Test with Val Thresholds...")
        print("="*40)
        
        self.independent_test(thresholds=best_val_thresholds)

    def independent_test(self, pth=None, thresholds=None):
            """
            Independent test
            """ 

            model_path = pth if pth is not None else self.model_path
            print(f"Independent test {self.task_tag}, model path: {model_path}")

            # 1. Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            model = checkpoint['model']
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

          
            print("Starting feature extraction (Pre-trained Model running)...")
            if self.device.type == 'cuda':
                torch.cuda.synchronize() 
            
            t0_feat = time.time()
            
            test_feas, test_labels, test_pad_masks, test_seqs = load_features(os.path.join(self.dataset_dir, 'test'), True, *self.pts, seed=self.seed)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            t1_feat = time.time()
            feature_extraction_time = t1_feat - t0_feat
        
            test_dataloder = DataLoader(dataset=LabelEmbeddingData(test_feas, test_labels, test_pad_masks, self.device),
                                        batch_size=self.batch_size, shuffle=False) # 测试集建议设为 False

            # Predict
            y_pred = []
            y_true = []
            

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            t0_infer = time.time()
            
            with torch.no_grad():
                for i, data in enumerate(test_dataloder):
                    X, y, masks, label_input = data
                    out, _, _, _ = model(X, masks, label_input) 
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(out.cpu().detach().numpy())
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            t1_infer = time.time()
            downstream_inference_time = t1_infer - t0_infer
      
            total_samples = len(y_true)
            total_e2e_time = feature_extraction_time + downstream_inference_time
            
           
            feat_latency_per_sample = (feature_extraction_time / total_samples) * 1000
            infer_latency_per_sample = (downstream_inference_time / total_samples) * 1000
            e2e_latency_per_sample = (total_e2e_time / total_samples) * 1000

            print("\n" + "="*50)
            print(f"⏱️  INFERENCE TIME REPORT")
            print("="*50)
            print(f"Total Samples: {total_samples}")
            print(f"[1] Feature Extraction Time (LLM etc.): {feature_extraction_time:.4f} s ({feat_latency_per_sample:.2f} ms/sample)")
            print(f"[2] Downstream Inference Time (TPMLC) : {downstream_inference_time:.4f} s ({infer_latency_per_sample:.2f} ms/sample)")
            print("-" * 50)
            print(f"🚀 TOTAL End-to-End Time              : {total_e2e_time:.4f} s")
            print(f"🚀 TOTAL Latency per sample           : {e2e_latency_per_sample:.2f} ms/sample")
            print("="*50 + "\n")

            
            self.evaluation(np.array(y_true), np.array(y_pred), 
                            tag='test_final', 
                            metric='F1+Jaccard', 
                            thresholds=thresholds)
   


    def evaluation(self, y_true, y_pred, tag='val', metric='F1+Jaccard', per_label_thresh=True, thresholds=None):

        def compute_sample_jaccard(y_true_bin, y_pred_bin):
            
            y_true_bin = y_true_bin.astype(int)
            y_pred_bin = y_pred_bin.astype(int)
            
            return np.mean([
                np.sum((y_pred_bin[i] & y_true_bin[i])) / np.sum((y_pred_bin[i] | y_true_bin[i])) 
                if np.sum(y_pred_bin[i] | y_true_bin[i]) > 0 else 1.0
                for i in range(len(y_true_bin))
        ])

        def search_best_threshold_for_label(y_true_label, y_pred_label, metric='F1'):
            thresholds_range = np.arange(0.1, 0.9, 0.01)
            best_thresh = 0.5
            best_score = -1
            for thresh in thresholds_range:
                y_pred_cls_label = (y_pred_label >= thresh).astype(int)
                f1_macro = f1_score(y_true_label, y_pred_cls_label, average='macro')
               
                if f1_macro > best_score:
                    best_score = f1_macro
                    best_thresh = thresh
            return best_thresh
        

        best_thresh_final = None
        
        if thresholds is None:
            if not per_label_thresh:
                
                thresholds_range = np.arange(0.1, 0.9, 0.01)
                best_thresh = 0.5
                best_score = -1
                for thresh in thresholds_range:
                    y_pred_cls = (y_pred >= thresh).astype(int)
                    f1_macro = f1_score(y_true, y_pred_cls, average='macro')
                    jaccard = compute_sample_jaccard(y_true, y_pred_cls)
                    
                    if metric == 'F1+Jaccard': score = 0.5 * (f1_macro + jaccard)
                    else: score = f1_macro 

                    if score > best_score:
                        best_score = score
                        best_thresh = thresh
                
                print(f"[{tag}] Best global threshold: {best_thresh:.3f}")
                best_thresh_final = best_thresh
                y_pred_cls = (y_pred >= best_thresh).astype(int)
            else:
                
                best_thresh_per_label = []
                for i in range(y_true.shape[1]):
                    best_thresh = search_best_threshold_for_label(y_true[:, i], y_pred[:, i], metric)
                    best_thresh_per_label.append(best_thresh)
                
                print(f"[{tag}] Learned thresholds: {np.round(best_thresh_per_label, 2)}")
                best_thresh_final = np.array(best_thresh_per_label)
                
                y_pred_cls = np.zeros_like(y_pred)
                for i in range(y_true.shape[1]):
                    y_pred_cls[:, i] = (y_pred[:, i] >= best_thresh_final[i]).astype(int)
        
       
        else:
            print(f"[{tag}] Using provided thresholds for testing...")
            best_thresh_final = thresholds
            y_pred_cls = np.zeros_like(y_pred)
            
           
            if isinstance(thresholds, (float, int, np.float64, np.float32)):
                y_pred_cls = (y_pred >= thresholds).astype(int)
            else:
                for i in range(y_true.shape[1]):
                    y_pred_cls[:, i] = (y_pred[:, i] >= thresholds[i]).astype(int)
       
        macro_f1 = f1_score(y_true, y_pred_cls, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred_cls, average='micro', zero_division=0)
        subset_acc = accuracy_score(y_true, y_pred_cls)
        hamming = hamming_loss(y_true, y_pred_cls)
        jaccard = compute_sample_jaccard(y_true, y_pred_cls)
        aim_score = 0.5 * (macro_f1 + jaccard)
       
        try:
            binary_metrics(y_pred, y_true, self.names, best_thresh_final,
                        f'{self.result_folder}/{self.task_tag}{tag}_binary.csv', show=False)
            instances_overall_metrics(y_pred, y_true, best_thresh_final,
                                    f'{self.result_folder}/{self.task_tag}{tag}_sample.csv', show=False)
            label_overall_metrics(y_pred, y_true, best_thresh_final,
                                f'{self.result_folder}/{self.task_tag}{tag}_label.csv', show=False)
        except Exception:
            pass 

        return best_thresh_final
       

    def freeze_layers(self, model, i):
        """
        Freeze the specific classifier layer i
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                if name.split('.')[1] == str(i):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

    def freeze_layers_dec(self, model):
        """
        Freeze the decoder classifier layers
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                param.requires_grad = True
            else:
                if name.startswith('rp.decoder_layers') or name.startswith('rp.label'):
                    print("freeze", name)
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def reset_classifiers(self, model):
        """
        Reinitialize the classifier layers
        """
        for name, param in model.named_parameters():
            if name.startswith('fcs'):
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def save_model(self, model, path):

        torch.save({
            'model': model,
            'model_state_dict': model.state_dict(),
            'pt_order': self.names,
            'args': self.args
        }, f'{path}')


    def set_seed(self, seed=123):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


 

