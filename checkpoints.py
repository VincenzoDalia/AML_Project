import os
import torch 

from globals import CONFIG

def load_epoch_from_checkpoint(model, scheduler, optimizer):
    cur_epoch = 0

    if os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
    
    return cur_epoch


def save_checkpoint(epoch, model, scheduler, optimizer):
    checkpoint = {
        'epoch': epoch + 1,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'model': model.state_dict(),
    }
    torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))