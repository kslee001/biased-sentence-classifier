from loadlibs import *
from loss_fn import NTXentLoss

from configs import CFG, TOOLS
import functions
import modules


if __name__ == '__main__':
    functions.seed_everything(CFG['SEED'])
    CFG['BATCH_SIZE'] = CFG['EXP_BATCH_SIZE']
    
    # arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--tqdm_off", dest = "tqdm_off", action='store_false')
    args = parser.parse_args()
    CFG['TQDM'] = bool(args.tqdm_off)
    if CFG['TQDM'] == False:
        print("-- tqdm turned off")
    
    # load model
    model = modules.ExpClassifier(encoder=TOOLS['BACKBONE'], linear_dim=TOOLS['LINEAR_DIM'])    
    if functions.is_parallel():
        CFG['BATCH_SIZE'] *= torch.cuda.device_count()
        model = torch.nn.parallel.DataParallel(model)
    model = model.to(TOOLS['DEVICE'])
    

    # load data & dataloader
    try:
        data = pd.read_csv("full_dataset.csv")
    except:
        data = functions.prepare_data()
        data.to_csv("full_dataset.csv", index=False)
    # data = data.sample(frac=1).reset_index(drop=True)[:20000]  # use small data
    dataloaders = functions.prepare_loader(data, dataset_type=modules.ExpDataset, drop_last=True) # list of 3 elements
    # input_data = next(iter(dataloaders[0]))[0]
    # input_ids, attention_mask = input_data['input_ids'].cuda(), input_data['attention_mask'].cuda()
    # embedding_adv = None
    # model_summary = summary(
    #     model=model, 
    #     input_data=[input_ids, attention_mask],
    #     dtypes = List[torch.float,torch.long],
    #     verbose=1 # 0 : no output / 1 : print model summary / 2 : full detail(weight, bias layers)
    # ).__repr__()
    
    
    
    # settings    
    optimizer = TOOLS['OPTIMIZER'](model.parameters(), lr = CFG['LEARNING_RATE'])
    scheduler = TOOLS['SCHEDULER'](optimizer=optimizer,**TOOLS['SCHEDULER_ARGS'])
    warm_up = TOOLS['WARM_UP'](optimizer=optimizer, **TOOLS['WARM_UP_ARGS'])
    class_criterion = TOOLS['CLASS_CRITERION'].to(TOOLS['DEVICE'])
    contrastive_criterion = NTXentLoss(
        device='cuda', 
        batch_size=CFG['BATCH_SIZE'], 
        temperature=0.07,
        use_cosine_similarity=True,
        ).to(TOOLS['DEVICE'])
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else torch.GradScaler()

    # train
    best_model = functions.adv_train_fn(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        warm_up=warm_up, 
        class_criterion=class_criterion, 
        contrastive_criterion=contrastive_criterion,
        scaler=scaler,
        dataloaders=dataloaders, 
        configs=CFG, 
        device='cuda',
        eps=0.05
    )
