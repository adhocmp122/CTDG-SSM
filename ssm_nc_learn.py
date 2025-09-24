from ssm_memory import *
from ssm_dataset import TemporalEdgeDataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from ssm_utlis import EarlyStopping,scatter_mean_update

def node_classification_train_single_sequence(model:MemoryModel,dataset:TemporalEdgeDataset, optimizer:torch.optim.Adam, scaler=None,mode:str='eval'):
    device = dataset.device
    # Solver Values ============================================================================================================================
    nodes = torch.tensor([
        -0.1834346424956498,
        -0.5255324099163290,
        -0.7966664774136267,
        -0.9602898564975363,
        0.1834346424956498,
        0.5255324099163290,
        0.7966664774136267,
        0.9602898564975363,
    ], dtype=torch.float32).to(device)

    weights = torch.tensor([
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763,
    ], dtype=torch.float32).to(device)

    tau = 1.0  # change as needed
    t_nodes = 0.5 * tau * (nodes + 1.0)
    t_weights = 0.5 * tau * weights
    # ===============================================================================================================================================

    #model = MemoryModel(num_nodes=total_nodes, input_dim=edge_feat_dim + time_enc_dim, hidden_dim=64, delta=0.1, device='cuda').to('cuda')
    model.eval()
    if mode != 'eval':
        model.train()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer.zero_grad()
        total_loss = 0
        

    all_preds, all_labels = [], []
    val_preds, val_labels = [], []
    # Only trained at window nodes! 

    
    t_warm = 100
    validation_mode = False
    valid_back_prop = False
    batch_loss = 0 
    batch_size = 100
    batch_counter = 0
    # Reset Internal States 
    model.reset()
    # model.model_c_1 = dict()
    # model.model_c_2 = dict()
    # model.model_c_3 = dict() 
    # model.node_embedding = dict()
    # T = dataset.T
    if dataset.unique_batch:
        counter = range(len(dataset))
    else:
        counter = range(0,len(dataset)-2*dataset.T,dataset.T)
    for t in tqdm(counter,desc='Epoch Progress: '):
        data = dataset[t]

        # target_t_encoded = data.target_enc 
        validation_mode = data.validate

        # prev_c_1 = model.model_c_1[-1][active_ids]
        # prev_c_1 = model.dict_lookup(data.active_input_ids,model.m1_dict,model.hidden_dim)
        # prev_c_2 = model.dict_lookup(data.active_input_ids,model.m2_dict,model.hidden_dim)
        prev_c_1 = model.m1_vec[data.active_input_ids]
        if model.update_type == 'mamba':
            prev_c_2 = model.m2_vec[data.active_input_ids]

            # L_present - L_past

        # with amp.autocast(device_type='cuda', dtype=torch.float16):

        # ==========================================================================================================================
        if torch.isnan(prev_c_1).any():
            print("NaN in prev_c before matmul")
            print(prev_c_1)
            break


        # Old Model ===================================================================================================================
        if model.update_type == 'HiPPO':
            neigh_info = torch.zeros((len(data.x_sub),model.embd_dims),device=model.device)
            neigh_info[:data.target_src.numel(),:model.embd_dims] = model.names(data.target_src)
            neigh_info[data.target_src.numel():2*data.target_src.numel(),:model.embd_dims] = model.names(data.target_dst)
            
            # neigh_info = torch.zeros((len(data.x_sub),model.hidden_dim),device=model.device)
            # neigh_info[data.src_idx] = model.ne_vec[data.target_dst]
            # neigh_info[data.dst_idx] = model.ne_vec[data.target_src]
            x_in = torch.cat([data.x_sub,neigh_info],dim=1)
            reg = model.reg
            L_t = data.L_agg
            Is = torch.eye(len(L_t)).to(L_t.device)
            delta_L = data.delta_L_agg
            Id = torch.eye(len(model.A_hippo)).to(model.device)
            zt = model.TuneInputNC(x_in) # input Tuning 
            ztn = model.rms_1(zt)
            Bzt = model.B1(ztn)
            delta_1 = F.softplus(model.dt_proj_a(ztn)).expand(-1, model.hidden_dim) # (n x d) delta for each input
            # A_cont_1 = torch.exp(model.A_log_1)

            updated_c_1 =  (( prev_c_1[:,None,:] @ (Id[None,:,:]- delta_1[:,:,None] * model.A_hippo[None,:,:])).squeeze(1)
                + (Is-(reg*L_t)) @ Bzt * delta_1
                - (Is-(reg*L_t) ) @ (reg*delta_L) @ prev_c_1)
            
            updated_c_1 =  (( prev_c_1[:,None,:] @ (Id[None,:,:]- delta_1[:,:,None] * model.A_hippo[None,:,:])).squeeze(1)
                + model.delta * (Is-(reg*L_t)) @ Bzt
                - (Is-(reg*L_t) ) @ model.B_delta_L(model.rms_1(prev_c_1)))

            final_state = zt+F.gelu(updated_c_1,approximate='tanh') # (zt + F.gelu(updated_c_1,approximate='tanh'))


        # # Update Sates (Dual Mamba - 1 ) ============================================================================================
        elif model.update_type == 'mamba':
            reg = model.reg
            L_t = data.L_agg
            Is = torch.eye(len(L_t)).to(L_t.device)
            delta_L = data.delta_L_agg
            As_bar = Is - reg*delta_L + reg**2 * delta_L @ (L_t + delta_L/2) # - reg**3 * delta_L @ (L_t2 + L_t@delta_L + delta_L2/6 ) 
            dt = delta_L[None] * t_nodes[:,None,None]
            #dt2 = dt@dt # (8,n,n)
            LHS = Is[None] - reg*dt + reg**2*(dt@(L_t[None] + dt/2)) #- reg**3 * dt @ (L_t2[None] + L_t[None]@dt + dt2/6) 
            P_L_inv = Is - reg*L_t #+ reg**2 * L_t2
            # time_conj_x = torch.vstack([target_t_encoded,target_t_encoded])
            # x_in = torch.hstack([data.x_sub,time_conj_x])
            # neigh_embds = model.ne_vec[data.active_input_ids]#model.dict_lookup(data.active_ids,model.ne_dict,3*model.hidden_dim)
            neigh_info = torch.zeros((len(data.x_sub),model.embd_dims),device=model.device)
            neigh_info[:data.target_src.numel(),:model.embd_dims] = model.names(data.target_src)
            neigh_info[data.target_src.numel():2*data.target_src.numel(),:model.embd_dims] = model.names(data.target_dst)
            
            x_in = torch.cat([data.x_sub,neigh_info],dim=1)

            # node_neighbour_identity = torch.zeros((len(data.active_input_ids),model.hidden_dim),device=model.device)
            # node_neighbour_identity[data.dst_idx] = model.names(data.src_t)
            # node_neighbour_identity[data.src_idx] = model.names(data.dst_t)
            # print(neigh_info.shape,data.x_sub.shape)
            # x_in = torch.cat([data.x_sub,node_neighbour_identity],dim=1)

            zt = model.TuneInputNC(x_in) # input Tuning / Selective Scan
            
            ztn = model.rms_1(zt)
            # first_gate = F.sigmoid(model.B_gate_a(ztn))
            # Bzt = first_gate*model.B1(ztn)
            Bzt = model.B1(ztn)
            delta_1 = F.softplus(model.dt_proj_a(ztn)).expand(-1, model.hidden_dim) # (n x d) delta for each input
            A_cont_1 = -torch.exp(model.A_log_1) # ( 1 x d)
            At_bar_1 = torch.exp(delta_1 * A_cont_1) # ( n x d)

            C_1 = (P_L_inv @ (Bzt * delta_1))[None]*t_weights[:,None,None] 
            RHS_1 = torch.exp((delta_1 * A_cont_1)[None] * t_nodes[:,None,None] )
            integral_1 = torch.sum(LHS@C_1*RHS_1,dim=0)

            updated_c_1= As_bar@(prev_c_1 * At_bar_1) + integral_1
            
            u_2 = zt + F.gelu(updated_c_1,approximate='tanh')# + zt approximate for faster results, maybe less accurate

            # final_state = torch.hstack([ztn,u_2])
            # Mamba Layer -2  ====================================================================
            
            u_2n = model.rms_2(u_2)
            Second_gate = F.sigmoid(model.B_gate_b(u_2n))
            But = Second_gate*model.B2(u_2n)
            # But = model.B2(u_2n)
            delta_2 = F.softplus(model.dt_proj_a(u_2n)).expand(-1, model.hidden_dim)
            A_cont_2 = -torch.exp(model.A_log_2) # ( 1 x d)
            At_bar_2 = torch.exp(delta_2 * A_cont_2) # ( n x d)

            C_2 = (P_L_inv @ But * delta_2)[None]*t_weights[:,None,None]
            RHS_2 = torch.exp((delta_2 * A_cont_2)[None] * t_nodes[:,None,None] )
            integral_2 = torch.sum(LHS@C_2*RHS_2,dim=0)
            updated_c_2 = As_bar@(prev_c_2 * At_bar_2) + integral_2

            final_state = u_2 + F.gelu(updated_c_2,approximate='tanh') # torch.hstack([ztn,u_2,updated_c_3]) #u_2 + F.gelu(updated_c_3,approximate='tanh')) # approximate for faster results


        # model.m1_vec[data.active_input_ids] = updated_c_1 #.detach()
        # if model.update_type == 'mamba':
        #     model.m2_vec[data.active_input_ids] = updated_c_2 #.detach()
        # model.ne_vec[data.active_input_ids] = final_state #.detach()
        idx = data.active_input_ids  # indices [num_active]
        # Clamping for stability
        model.m1_vec = torch.clamp(scatter_mean_update(model.m1_vec, idx, updated_c_1),min=-1e+8,max=1e+8)
        if model.update_type == 'mamba':
            model.m2_vec = torch.clamp(scatter_mean_update(model.m2_vec, idx, updated_c_2),min=-1e+8,max=1e+8)
        model.ne_vec = torch.clamp(scatter_mean_update(model.ne_vec, idx, final_state),min=-1e+8,max=1e+8)

        # k = torch.arange(len(data.active_input_ids))
        # for src_k in torch.unique(data.active_input_ids):
        #     key = src_k.item()
        #     model.m1_dict[key] = updated_c_1[k[data.active_input_ids==src_k]] #.mean(dim=0) #.detach()
        #     model.m2_dict[key] = updated_c_3[k[data.active_input_ids==src_k]] #.mean(dim=0) #.detach()
        #     model.ne_dict[key] = final_state[k[data.active_input_ids==src_k]] #.mean(dim=0) #.detach()
            # if key not in model.node_embedding:
            #     model.node_embedding[key] = final_state[k[data.active_ids==src_k]].mean(dim=0)
            # else:

            #     model.node_embedding[key] = final_state[k[data.active_ids==src_k]].mean(dim=0)

        # Node Classification ===================================================================================================
                # Predict State ==========================================================================================================
        # L_out = data.L_out
        
        # memory = model.dict_lookup(data.active_ids,model.ne_dict,model.hidden_dim)
        # memory =  final_state[data.src_idx] #model.ne_vec[data.active_ids]

        # Graph Transformed 
        # memory[model.hidden_dim:] = (model.alpha*torch.eye(len(L_out)) + model.beta*L_out)@memory[model.hidden_dim:]

        # zex = data.target_enc.shape[0]
        if (t>t_warm and mode != 'eval') or (mode == 'eval' and validation_mode):
            states = model.ne_vec[data.target_src]
            projected_states = model.project_t(states,data.target_enc)
            # memory_target_src = model.ne_vec[data.target_src]#model.dict_lookup(data.target_src,memory).reshape(zex,-1)
            
            # embd_target_dst = memory[data.active_dst_mask]#model.dict_lookup(data.target_dst,memory).reshape(zex,-1)
            # memory_neg_dst = memory[data.active_neg_mask]#model.dict_lookup(data.neg_dst,memory).reshape(zex,-1)

            score = model.NodeClassificaiton(projected_states) 
            label = data.label_t

        if mode != 'eval' and t>t_warm:
            loss = loss_fn(score, label[:,None]) 

        if not validation_mode and mode != 'eval' and t>t_warm:
            batch_loss += loss
            valid_back_prop = True
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            all_preds.append(score.detach().cpu())
            all_labels.append(label.detach().cpu())

        elif validation_mode:
            model.eval()
            val_preds.append(score.detach().cpu())
            val_labels.append(label.detach().cpu())

        #=========================================================================================================================
        if t>t_warm and not validation_mode:
            batch_counter += 1
        if not validation_mode and mode != 'eval' and valid_back_prop and t>t_warm and batch_counter >= batch_size:
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            valid_back_prop = False
            model.detach()
            batch_counter = 0
            batch_loss = 0
            
        elif not validation_mode and data.next_validate and valid_back_prop :
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            valid_back_prop = False
            model.detach()
            batch_counter = 0
            batch_loss = 0

        if mode != 'eval' and t>t_warm and valid_back_prop:
            total_loss += batch_loss.item()

    if mode != 'eval':
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        
        val_pred = torch.cat(val_preds).numpy()
        val_true = torch.cat(val_labels).numpy()

        auc = roc_auc_score(y_true, y_pred)
        val_auc = roc_auc_score(val_true , val_pred)
        return total_loss, auc, val_auc

    elif mode == 'eval':
        
        val_pred = torch.cat(val_preds).numpy()
        val_true = torch.cat(val_labels).numpy()

        val_auc = roc_auc_score(val_true , val_pred)
        return val_auc
    






def main(model, full_dataset:TemporalEdgeDataset,train_dataset:TemporalEdgeDataset, optimizer, num_epochs:int=10,name:str='trn', device:str='cuda'):
    path = name +  'best_model_nb.pt'
    early_stopping = EarlyStopping(patience=10, verbose=True, path=path)
    scaler = None # amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.75, patience=5
    )
    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        loss, auc, val_auc = node_classification_train_single_sequence(model,train_dataset,optimizer,scaler,mode='train')
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
        val_auc = node_classification_train_single_sequence(model,full_dataset,optimizer,scaler,mode='eval')
        print(f"VAL-AUC: {val_auc:.6f}")

    return val_auc

        
    