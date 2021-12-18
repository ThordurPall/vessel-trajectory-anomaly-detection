import numpy as np
import torch

def evaluate_VRNN(model, test_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_n = len(test_loader.dataset)
    maxLength = test_loader.dataset.maxLength
    latent_dim = model.latent_shape
    recurrent_dim = model.recurrent_shape
    batch_size = test_loader.batch_size
    
    zmus = np.zeros((maxLength, test_n, latent_dim))
    hs = np.zeros((maxLength, test_n, recurrent_dim))
    activatedBins = np.zeros((maxLength, test_n, 4))
    recon_loss = np.zeros((maxLength, test_n))
    j = 0
    for i, (_, _, _, lengths, inputs, targets) in enumerate(test_loader):
        max_len = int(torch.max(lengths).item())
                
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
                
        endIndex = (batch_size*(i+1)) if (batch_size*(i+1)) <= test_n else test_n
        
        recon_loss[:max_len, (batch_size*i):endIndex] = log_px.detach().cpu().numpy()
        zmus[:max_len,(batch_size*i):endIndex,:] = means.detach().cpu().numpy()
        hs[:max_len,(batch_size*i):endIndex,:] = hidden_states.detach().cpu().numpy()
                 
    return recon_loss, activatedBins, zmus, hs


