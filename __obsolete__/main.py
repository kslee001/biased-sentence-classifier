from loadlibs import *
from loss_fn import NTXentLoss

from configs import CFG, TOOLS
import functions
import modules


if __name__ == '__main__':
    functions.seed_everything(CFG['SEED'])

    # arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqdm_off", dest = "tqdm_off", action='store_false')
    args = parser.parse_args()
    CFG['TQDM'] = bool(args.tqdm_off)
    if CFG['TQDM'] == False:
        print("-- tqdm turned off")
    
    # load model
    model = TOOLS['MODEL'](encoder=TOOLS['BACKBONE'], linear_dim=TOOLS['LINEAR_DIM'])    
    if torch.cuda.device_count()>=1:
        CFG['BATCH_SIZE'] *= torch.cuda.device_count()
        model = torch.nn.parallel.DataParallel(model)
    model = model.to(TOOLS['DEVICE'])
    
    # load data & dataloader
    data = functions.prepare_data()
    dataloaders = functions.prepare_loader(data) # list of 3 elements
    model_summary = summary(
        model=model, 
        input_data=next(iter(dataloaders[0]))[0],
        verbose=1 # 0 : no output / 1 : print model summary / 2 : full detail(weight, bias layers)
    ).__repr__()
    
    # settings    
    optimizer = TOOLS['OPTIMIZER'](model.parameters(), lr = CFG['LEARNING_RATE'])
    scheduler = TOOLS['SCHEDULER'](optimizer=optimizer,**TOOLS['SCHEDULER_ARGS'])
    warm_up = TOOLS['WARM_UP'](optimizer=optimizer, **TOOLS['WARM_UP_ARGS'])
    criterion = TOOLS['CRITERION'].to(TOOLS['DEVICE'])
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else torch.GradScaler()

    # train
    best_model = functions.train_fn(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        warm_up=warm_up, 
        criterion=criterion, 
        scaler=scaler,
        dataloaders=dataloaders, 
        configs=CFG, 
        device='cuda'
    )
