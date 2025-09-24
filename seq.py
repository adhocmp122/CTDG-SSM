import torch
from torch.utils.data import Dataset
import numpy as np

class CTDGraphDataset(Dataset):
    def __init__(self, N, M, seed=42):
        """
        Args:
            N (int): number of edges per sequence
            M (int): number of sequences
        """
        self.N = N
        self.M = M
        np.random.seed(seed)

        self.data = []
        self.labels = []

        for _ in range(M):
            # node 0 signal = label
            y = np.random.choice([0, 1])
            self.labels.append(y)

            # node signals
            signals = {0: float(y)*2-1}
            for i in range(1, N + 1):
                signals[i] = np.random.normal(0, 1)/10

            # edges
            src = np.arange(0, N)
            dst = np.arange(1, N + 1)
            times = np.sort(np.random.uniform(0, 10, N))  # increasing times

            # endpoint signals
            x_src = np.array([signals[s] for s in src])
            x_dst = np.array([signals[d] for d in dst])

            self.data.append((src, dst, times, x_src, x_dst))

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        src, dst, times, x_src, x_dst = self.data[idx]
        y = self.labels[idx]

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "dst": torch.tensor(dst, dtype=torch.long),
            "t": torch.tensor(times, dtype=torch.float32),
            "x_src": torch.tensor(x_src, dtype=torch.float32),
            "x_dst": torch.tensor(x_dst, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32)
        }

from torch.utils.data import DataLoader, random_split

def get_loaders(N, M, batch_size=4, split=0.8, seed=42):
    dataset = CTDGraphDataset(N, M, seed=seed)

    # split sizes
    train_size = int(split * M)
    test_size = M - train_size

    # train/test split
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

N = 9
M = 1000
train_loader,test_loader = get_loaders(N=N,M=M,batch_size=1,split=0.7)

from ssm_memory import MemoryModel
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
test_acc = []
for _ in range(10):
    state_dim = 16
    time_dim = 4
    device = 'cpu'
    lr = 1e-3
    epochs = 100
    model = MemoryModel(num_nodes=N+1,input_dim=1,hidden_dim=state_dim,time_dim=time_dim,reg=1e-4,device=device)
    # early_stopping = EarlyStopping(patience=100, mode="min", path="best_model.pt")

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

    criterion = nn.BCEWithLogitsLoss()  # for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    Id = torch.eye(model.hidden_dim,device=model.device)
    for epoch in tqdm(range(epochs),desc='Epochs: '):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        hidden_state_list = []
        # yt = torch.zeros((len(train_loader),1),device=model.device)
        yt = []
        for bid,batch in enumerate(train_loader):
            src, dst, t = batch["src"].to(device)[0], batch["dst"].to(device)[0], batch["t"].to(device)[0]
            x_src, x_dst, y = batch["x_src"].to(device)[0], batch["x_dst"].to(device)[0], batch["y"].to(device)
            yt.append(y[0])
            reg = 1 #torch.log(1+ torch.exp(model.reg))
            
            m1_vec = torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
            m2_vec = torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
            ne_vec = torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
            for t in range(len(src)):
                model.reset()
                actives = torch.hstack([src[t],dst[t]])
                L_past = torch.eye(len(actives),device=model.device)
                Is = torch.eye(len(actives),device=model.device)
                src_t = src[:t] 
                dst_t = dst[:t]
                x_t = torch.randn((model.num_nodes,model.input_dim),device=model.device)*0
                # x_n =  torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
                # print(src[t])
                x_t[src[t]]=x_src[t]
                x_t[dst[t]]=x_dst[t]
                # x_t[src_t,:]=x_src[:t][:,None]
                # x_t[dst_t,:]=x_dst[:t][:,None]
                A = torch.eye(model.num_nodes,device=model.device)
                for i in range(len(src_t)):
                    # A[src[i],dst[i]]=1
                    A[dst[i],src[i]]=1
                    
                # D = A.sum(dim=1)[:,None]
                L = torch.eye(len(actives),device=model.device) #- A/D
                L[0,1]=-1
                L[1,0]=-1
                delta = 1/(N+1)
                prev_c_1 = m1_vec[actives]
                prev_c_2 = m2_vec[actives]
                P_L_inv =   torch.linalg.inv(Is + reg*L)
                delta_L = reg*(L - L_past)  
                zt = model.TuneInputSC(x_t[actives]) # input Tuning / Selective Scan
                ztn = zt #model.rms_1(zt)
                updated_c_1 = prev_c_1 @ (Id/2 - model.A_hippo*delta) + P_L_inv@zt@model.B_hippo*delta - P_L_inv@ model.BL_delta(prev_c_1)
                # u_2 = zt+F.gelu(updated_c_1,approximate='tanh')# + zt approximate for faster results, maybe less accurate
                final_state = updated_c_1
                # Mamba Layer -2  ====================================================================

                m1_vec[actives] = updated_c_1
                # m2_vec = updated_c_2
                ne_vec[actives] = final_state
                # L_past = L.clone()
                # print(ne_vec[0])
                # print(m1_vec[0])

            # print('outer_loop: ',ne_vec[0])
            
            
            hidden_state_list.append(ne_vec)
            # print(hidden_state_list[-1][0])
            logits = model.SeqClass(hidden_state_list[-1][-1])
            # print(logits)
            loss = criterion(logits, y)

            total_loss += loss #/ len(train_loader)
            preds = ( logits> 0).float()
            
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        acc = total_correct / total_samples
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Train Acc: {acc:.4f}")

        # if early_stopping(total_loss, model):
        #     print("Early stopping triggered")
        #     break

    print('Eval Mode===========')
    # model = early_stopping.load_best_model(model)
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    eval_state_list = []
    for bid,batch in enumerate(test_loader):
        src, dst, t = batch["src"].to(device)[0], batch["dst"].to(device)[0], batch["t"].to(device)[0]
        x_src, x_dst, y = batch["x_src"].to(device)[0], batch["x_dst"].to(device)[0], batch["y"].to(device)

        reg = 1 #torch.log(1+ torch.exp(model.reg))
        
        m1_vec = torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
        m2_vec = torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
        ne_vec = torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
        for t in range(len(src)):
            model.reset()
            actives = torch.hstack([src[t],dst[t]])
            L_past = torch.eye(len(actives),device=model.device)
            Is = torch.eye(len(actives),device=model.device)
            src_t = src[:t] 
            dst_t = dst[:t]
            x_t = torch.randn((model.num_nodes,model.input_dim),device=model.device)*0
            # x_n =  torch.zeros((model.num_nodes,model.hidden_dim),device=model.device)
            # print(src[t])
            x_t[src[t]]=x_src[t]
            x_t[dst[t]]=x_dst[t]
            # x_t[src_t,:]=x_src[:t][:,None]
            # x_t[dst_t,:]=x_dst[:t][:,None]
            A = torch.eye(model.num_nodes,device=model.device)
            for i in range(len(src_t)):
                # A[src[i],dst[i]]=1
                A[dst[i],src[i]]=1
                
            L = torch.eye(len(actives),device=model.device) #- A/D
            L[0,1]=-1
            L[1,0]=-1
            prev_c_1 = m1_vec[actives]
            prev_c_2 = m2_vec[actives]
            P_L_inv =   torch.linalg.inv(Is + reg*L)
            delta_L = reg*(L - L_past)  
            zt = model.TuneInputSC(x_t[actives]) # input Tuning / Selective Scan
            ztn = zt #model.rms_1(zt)
            delta = 1/(N+1)
            # updated_c_1= (prev_c_1 * At_bar_1) + Bt_bar
            # print(zt.shape,prev_c_1.shape,model.A_hippo.shape)
            # delta_n = model.dt_proj_a(model.names(torch.arange(len(ztn))))

            updated_c_1 = prev_c_1 @ (Id/2 - model.A_hippo*delta) + P_L_inv@zt@model.B_hippo*delta - P_L_inv@ model.BL_delta(prev_c_1)
            # u_2 = zt+F.gelu(updated_c_1,approximate='tanh')# + zt approximate for faster results, maybe less accurate
            final_state = updated_c_1
            m1_vec[actives] = updated_c_1
            ne_vec[actives] = final_state

        # print('outer_loop: ',ne_vec[0])
        
        
        eval_state_list.append(ne_vec)
        # print(hidden_state_list[-1][0])
        logits = model.SeqClass(eval_state_list[-1][-1])
        # print(logits)
        preds = ( logits> 0).float()
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    acc = total_correct / total_samples
    test_acc.append(acc)
    print(f" Eval Loss: {total_loss:.4f} | Eval Acc: {acc:.4f}")
len(test_acc)
print(f"ACC: {np.mean(test_acc):.4f} ± {np.std(test_acc,ddof=0):.4f}")
