import numpy as np
import torch


def run_VRNN(model, train_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_n = len(train_loader.dataset)
    maxLength = train_loader.dataset.maxLength
    latent_dim = model.latent_shape
    recurrent_dim = model.recurrent_shape
    batch_size = train_loader.batch_size
    
    zmus = np.zeros((maxLength, train_n, latent_dim))
    hs = np.zeros((maxLength, train_n, recurrent_dim))
    activatedBins = np.zeros((maxLength, train_n, 4))
    recon_loss = np.zeros((maxLength, train_n))
    lengths_tot = np.zeros((train_n))
    j = 0
    for i, (mmsis, timestamps, _, lengths, inputs, targets) in enumerate(train_loader):
        max_len = int(torch.max(lengths).item())
        endIndex = (batch_size*(i+1)) if (batch_size*(i+1)) <= train_n else train_n        
        lengths_tot[(batch_size*i):endIndex] = lengths.numpy()
        
        for target, length in zip(targets, lengths):
            activatedBins[:int(length),j,:] = torch.nonzero(target[:int(length),:], as_tuple=True)[1].reshape(int(length), 4).numpy() #activated bins in target
            j+=1
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        
        means = torch.zeros(max_len, len(lengths), latent_dim, device = device)
        hidden_states = torch.zeros(max_len, len(lengths), recurrent_dim, device = device)
        
        log_px, _, _, _, hidden_states, _, means, _ = model(inputs,targets,logits=None,z_mus=means,hs=hidden_states)

        curmask = torch.arange(max_len, device=device)[:, None] < lengths[None, :] #max_seq_len X Batch
                    
        log_px = torch.stack(log_px, dim=0) * curmask #max_seq_len X Batch
        
        recon_loss[:max_len, (batch_size*i):endIndex] = log_px.detach().cpu().numpy()
        zmus[:max_len,(batch_size*i):endIndex,:] = means.detach().cpu().numpy()
        hs[:max_len,(batch_size*i):endIndex,:] = hidden_states.detach().cpu().numpy()
             
    return recon_loss, activatedBins, zmus, hs, lengths_tot.astype(int)

def constuct_logprob_map(model, train_loader):

    if model.__class__.__name__ == 'VRNN':
        recon_loss, activatedBins, zmus, hs, lengths = run_VRNN(model, train_loader)

    trainset = train_loader.dataset

    #Split recon_loss into originating geographical bin
    lat_dim = len(trainset.params['binedges'][0]) - 1
    lon_dim = len(trainset.params['binedges'][1]) - 1
    map_logprob = dict()
    for row  in range(lat_dim):
        for col in range(lon_dim):
            map_logprob[ str(row)+","+str(col)] = []
                  
    for i, trackLength in enumerate(lengths):
        for t in range(trackLength):
            activatedlat = activatedBins[t,i,0] 
            activatedlon = activatedBins[t,i,1]-lat_dim
            
            map_logprob[ str(int(activatedlat))+","+str(int(activatedlon))].append(recon_loss[t,i])
            
    return map_logprob, zmus, hs, lengths
                
         

