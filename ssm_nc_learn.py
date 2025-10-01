from ssm_memory import *
from ssm_dataset import TemporalEdgeDataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from ssm_utlis import EarlyStopping,scatter_mean_update

def node_classification_train_single_sequence(model:MemoryModel,dataset:TemporalEdgeDataset, optimizer:torch.optim.Adam):

    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    total_loss = 0
    all_preds, all_labels = [], []
    # Only trained at window nodes! 
    t_warm = 100

    batch_loss = 0 
    batch_size = 10
    valid_grad = False
    batch_counter = 0
    # Reset Internal States 
    model.reset()

    if dataset.unique_batch:
        counter = range(len(dataset))
    else:
        counter = range(0,len(dataset)-2*dataset.T,dataset.T)
    for t in tqdm(counter,desc='Epoch Progress: '):
        data = dataset[t]
        model.zoh_update(data)

        if t>t_warm:
            states = model.ne_vec[data.target_src]
            time_embedding = model.time_encoder(data.target_enc)
            concatinated_state = torch.hstack([states,time_embedding])
            score = model.NodeClassificaiton(concatinated_state) 
            label = data.label_t
            loss = loss_fn(score, label[:,None]) 
            batch_loss += loss
            valid_grad = True
            all_preds.append(score.detach().cpu())
            all_labels.append(label.detach().cpu())

        #=========================================================================================================================
        if t>t_warm :
            batch_counter += 1

        if (valid_grad and t>t_warm and batch_counter>= batch_size) or (valid_grad and data.end_train) :
            batch_counter=0
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            model.detach()
            valid_grad = False
            total_loss += batch_loss.item()
            batch_loss = 0

          
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    auc = roc_auc_score(y_true, y_pred)
    return total_loss, auc

    
def node_classification_eval(model:MemoryModel,dataset:TemporalEdgeDataset):

    all_preds, all_labels = [], []
    # Only trained at window nodes! 
    # Reset Internal States 

    if dataset.unique_batch:
        counter = range(len(dataset))
    else:
        counter = range(0,len(dataset)-2*dataset.T,dataset.T)
    for t in tqdm(counter,desc='Epoch Progress: '):
        data = dataset[t]
        model.zoh_update(data)

        states = model.ne_vec[data.target_src]
        time_embedding = model.time_encoder(data.target_enc)
        concatinated_state = torch.hstack([states,time_embedding])
        score = model.NodeClassificaiton(concatinated_state) 
        label = data.label_t
        all_preds.append(score.detach().cpu())
        all_labels.append(label.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    auc = roc_auc_score(y_true, y_pred)
    return auc





def main(model, test_dataset,val_dataset,train_dataset, optimizer, num_epochs=10,name='trn', device='cuda'):
    path = name +  'best_model_nb.pt'
    early_stopping = EarlyStopping(patience=10, verbose=True, path=path)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.75, patience=5
    )
    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        loss, auc = node_classification_train_single_sequence(model,train_dataset,optimizer)
        with torch.no_grad():
            val_auc = node_classification_eval(model,val_dataset)

        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | AUC: {auc:.4f}| VAL-AUC: {val_auc:.4f}")
        scheduler.step(val_auc)
        print("Last LR:", scheduler.get_last_lr())
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    with torch.no_grad():
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        val_auc = node_classification_eval(model,test_dataset)
        print(f"VAL-AUC: {val_auc:.6f}")

    return val_auc

        
    