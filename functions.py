from loadlibs import *
from configs import target_columns, CFG, DATA, TOOLS
import modules



def is_parallel():
    return torch.cuda.device_count()>1


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def prepare_data():
    gender_clean_dataset = []
    gender_dirty_dataset = []
    general_clean_dataset = []
    general_dirty_dataset = []
    dataset_columns = ['text', 'gender', 'toxic']

    """ first dataset : ucberkly measuring hate speech"""
    dataset = datasets.load_dataset(DATA['DATASET1'], 'binary')   
    df = dataset['train'].to_pandas()
    df = df[df['text'].isnull()==False]
    df = df[df['text']!= ""]
    gender_idx = df[(df['target_gender_men']==True)|(df['target_gender_women']==True)].index
    general_idx = sorted(set(df.index) - set(gender_idx))
    gender = df.loc[gender_idx]
    general = df.loc[general_idx]

    theshold = 0.2

    gender_dirty = gender[(gender['hatespeech']==2)&(gender['hate_speech_score']>theshold)]
    gender_clean = gender[(gender['hatespeech']<=1)&(gender['hate_speech_score']<-theshold)]
    general_dirty = general[(general['hatespeech']==2)&(general['hate_speech_score']>theshold)]
    general_clean = general[(general['hatespeech']<=1)&(general['hate_speech_score']<-theshold)]

    gender_dirty['gender']=True
    gender_clean['gender']=True
    general_dirty['gender']=False
    general_clean['gender']=False

    gender_dirty['toxic']=True
    general_dirty['toxic']=True
    gender_clean['toxic']=False
    general_clean['toxic']=False

    gender_dirty = gender_dirty[dataset_columns]
    gender_clean = gender_clean[dataset_columns]
    general_dirty = general_dirty[dataset_columns]
    general_clean = general_clean[dataset_columns]

    # append dataset
    gender_clean_dataset.append(gender_clean)
    gender_dirty_dataset.append(gender_dirty)
    general_clean_dataset.append(general_clean)
    general_dirty_dataset.append(general_dirty)


    """ second dataset : HateXplain """
    with open(DATA['DATASET2']) as f:
        json_object = json.load(f)
    keys = list(json_object.keys())

    gender_dirty = []
    general_dirty = []
    for post_id in keys:
        cur = json_object[post_id]
        gender_flag = False
        toxic_flag = False
        for annotator_idx in range(len(cur['annotators'])):
            txt = " ".join(cur["post_tokens"])
            if (txt == "") | (txt == " "):
                continue
            if cur['annotators'][annotator_idx]['label'] in ['offensive', 'hatespeech']:
                toxic_flag = True
            if 'Women' in cur['annotators'][annotator_idx]['target']:
                gender_flag = True             
        if toxic_flag :
            if gender_flag:
                gender_dirty.append(txt)
            else:
                general_dirty.append(txt)
    gender_dirty = pd.DataFrame(gender_dirty)
    gender_dirty['gender']=True
    gender_dirty['toxic']=True
    gender_dirty.columns = dataset_columns

    general_dirty = pd.DataFrame(general_dirty)
    general_dirty['general']=True
    general_dirty['toxic']=True
    general_dirty.columns = dataset_columns

    # append dataset
    gender_dirty_dataset.append(gender_dirty)
    general_dirty_dataset.append(general_dirty)


    """ third dataset : kaggle unintended bias """
    identity = pd.read_csv(DATA['DATASET3'][0])
    gender_idx = identity[(identity['gender']=='male') | (identity['gender']=='female') | (identity['gender']=='male female')].id.drop_duplicates().tolist()
    train = pd.read_csv(DATA['DATASET3'][1])
    train = train[train['comment_text'].isnull()==False]
    train = train[train['comment_text']!= ""]
    general_idx = sorted(set(train.id.tolist()) - set(gender_idx))
    gender = train[train['id'].isin(gender_idx)]
    general = train[train['id'].isin(general_idx)]
    gender_clean = gender[gender['toxicity_annotator_count']<5]
    gender_dirty = gender[gender['toxicity_annotator_count']>10]
    general_clean = general[general['toxicity_annotator_count']<5]
    general_dirty = general[general['toxicity_annotator_count']>10]

    gender_clean['text']=gender_clean['comment_text']
    gender_dirty['text']=gender_dirty['comment_text']
    general_clean['text']=general_clean['comment_text']
    general_dirty['text']=general_dirty['comment_text']

    gender_clean['gender']=True
    gender_dirty['gender']=True
    general_clean['gender']=False
    general_dirty['gender']=False

    gender_clean['toxic']=False
    gender_dirty['toxic']=True
    general_clean['toxic']=False
    general_dirty['toxic']=True

    gender_clean = gender_clean[dataset_columns]
    gender_dirty = gender_dirty[dataset_columns]
    general_clean = general_clean[dataset_columns]
    general_dirty = general_dirty[dataset_columns]

    gender_clean_dataset.append(gender_clean)
    gender_dirty_dataset.append(gender_dirty)
    general_clean_dataset.append(general_clean)
    general_dirty_dataset.append(general_dirty)

    gender_clean_dataset = pd.concat(gender_clean_dataset, axis=0)
    gender_dirty_dataset = pd.concat(gender_dirty_dataset, axis=0)
    general_clean_dataset = pd.concat(general_clean_dataset, axis=0)
    general_dirty_dataset = pd.concat(general_dirty_dataset, axis=0)

    print("gender clean dataset  : ", len(gender_clean_dataset))
    print("gender dirty dataset  : ", len(gender_dirty_dataset))
    print("general clean dataset : ", len(general_clean_dataset))
    print("general dirty dataset : ", len(general_dirty_dataset))

    general_clean_dataset = general_clean_dataset.sample(frac=1).reset_index(drop=True)[:len(general_clean_dataset)//10]
    print("--reduce the size of general clean dataset : ", len(general_clean_dataset))

    full_dataset = pd.concat([gender_clean_dataset, gender_dirty_dataset, general_clean_dataset, general_dirty_dataset], axis = 0).reset_index(drop=True)
    print("full dataset size : ", len(full_dataset))
    return full_dataset


def prepare_loader(data, dataset_type=modules.BaseDataset, drop_last=False):
    train, valid = train_test_split(data, test_size=CFG['TEST_SIZE'], random_state=CFG['SEED'])
    valid, test = train_test_split(valid, test_size=0.5, random_state=CFG['SEED'])
    
    train_dataset=dataset_type(
        data=train,
        tokenizer=TOOLS['TOKENIZER'],
        mode='train',
    )
    valid_dataset=dataset_type(
        data=valid,
        tokenizer=TOOLS['TOKENIZER'],
        mode='valid',
    )
    test_dataset=dataset_type(
        data=test,
        tokenizer=TOOLS['TOKENIZER'],
        mode='test'
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=True,
        drop_last=drop_last,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=True,
        drop_last=drop_last,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=CFG['BATCH_SIZE'],
        shuffle=False,
        drop_last=False,
    )
    
    return train_loader, valid_loader, test_loader



def train_fn(model, optimizer, scheduler, warm_up, criterion, scaler,
             dataloaders, configs, device): 
    def forward_step(batch, mode='train'):
        encoded, y = batch
        input_ids, attention_mask = encoded['input_ids'].to(device), encoded['attention_mask'].to(device)
        y = y.float().to(device)
        
        optimizer.zero_grad()
        yhat = model(input_ids, attention_mask)
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss = criterion(yhat, y)

        if mode == 'train':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return yhat, loss
    
    def validation():
        model.eval()
        preds = []
        labels = []
        val_loss = []
        
        with torch.no_grad():
            valid_iterator = tq(valid_loader) if configs['TQDM'] else valid_loader
            for batch in valid_iterator:
                encoded, y = batch
                yhat, loss = forward_step(batch, 'valid')
                yhat = yhat.detach().cpu().numpy() > 0.5
                yhat = yhat.astype(int).tolist()
                
                val_loss.append(loss.item())                
                labels.extend(y.cpu().detach().numpy().tolist())
                preds.extend(yhat)
                
        print("true labels (~10) : ", labels[:10])
        print("pred labels (~10) : ", preds[:10])
        acc = [labels[idx] == preds[idx] for idx in range(len(labels))]
        acc = sum(acc)/len(acc)
        return np.mean(val_loss), acc
    
    
    model = model.to(device)
    train_loader, valid_loader, test_loader = dataloaders
    
    best_acc = 0
    best_loss = 999999
    best_model = None
    
    for epoch in range(1, configs['EPOCHS']):
        model.train()
        train_loss = []
        
        # train
        train_iterator = tq(train_loader) if configs['TQDM'] else train_loader
        for batch in train_iterator:
            _, loss = forward_step(batch)
            train_loss.append(loss.item())
            
        train_loss = np.mean(train_loss)
        
        # validation
        val_loss, acc = validation()
        
        if acc >= best_acc:
            best_acc = acc
            best_model = model
            
        print(f"-- EPOCH {epoch} --")
        print("val_loss : ", round(val_loss, 4))
        print("accuracy : ", round(acc, 4))
        
        if epoch < configs['WARM_UP_EPOCHS']:
            warm_up.step()
        else:
            scheduler.step(acc)

    return best_model
            
            

    
def adv_train_fn(model, optimizer, scheduler, warm_up, class_criterion, contrastive_criterion, scaler,
             dataloaders, configs, device, eps=0.05): 
    if device == 'cuda':
        device = f"{device}:{torch.cuda.current_device()}"
        
    def forward_step(encoded, y, embedding_grad):
        optimizer.zero_grad()
        input_ids, attention_mask = encoded['input_ids'].to(device), encoded['attention_mask'].to(device)
        yhat, feature = model(input_ids, attention_mask, embedding_grad)
        return yhat, feature
    
    def validation():
        model.eval()
        preds = []
        labels = []
        val_loss = []
        
        with torch.no_grad():
            valid_iterator = tq(valid_loader) if configs['TQDM'] else valid_loader
            for batch in valid_iterator:
                encoded, y = batch[0], batch[1].float().to(device)
                yhat, feature = forward_step(encoded, y, None)
                loss = class_criterion(yhat, y)
                yhat = yhat.detach().cpu().numpy() > 0.5
                yhat = yhat.astype(int).tolist()
                
                val_loss.append(loss.item())                
                labels.extend(y.cpu().detach().numpy().tolist())
                preds.extend(yhat)
                
        print("true labels (~10) : ", labels[:10])
        print("pred labels (~10) : ", preds[:10])
        acc = [labels[idx] == preds[idx] for idx in range(len(labels))]
        acc = sum(acc)/len(acc)
        return np.mean(val_loss), acc
    
    
    model = model.to(device)
    train_loader, valid_loader, test_loader = dataloaders
    
    best_acc = 0
    best_loss = 999999
    best_model = None
    
    for epoch in range(1, configs['EPOCHS']):
        model.train()
        train_loss = []
        
        # train
        train_iterator = tq(train_loader) if configs['TQDM'] else train_loader
        for batch in train_iterator:
            encoded, y = batch[0], batch[1].float().to(device)
            yhat, feature = forward_step(encoded, y, None)
            
            """
                embedding perturbation stage
            """
            model.zero_grad()
            # first loss calculation (negative loss)
            init_pred = yhat.detach().cpu().numpy() > 0.5
            init_pred = init_pred.astype(int).tolist()
            # adversarial attack
            negative_loss = -class_criterion(yhat, y)
            negative_loss.backward()
            
            # get adversarial sample
            if is_parallel():
                temp_embeddings = torch.nn.Embedding(
                    num_embeddings=model.module.embeddings.word_embeddings.num_embeddings,
                    embedding_dim =model.module.embeddings.word_embeddings.embedding_dim,
                )
                embedding_adv = model.module.embeddings.word_embeddings.weight.detach().clone()
                grad = model.module.embeddings.word_embeddings.weight.grad
            else:
                temp_embeddings = torch.nn.Embedding(
                    num_embeddings=model.embeddings.word_embeddings.num_embeddings,
                    embedding_dim =model.embeddings.word_embeddings.embedding_dim,
                )
                embedding_adv = model.embeddings.word_embeddings.weight.detach().clone()
                grad = model.embeddings.word_embeddings.weight.grad
                
            embedding_adv = torch.nn.Parameter(embedding_adv - eps*grad)
            temp_embeddings.weight = embedding_adv

            """
                classification stage
            """
            n_yhat, n_feature = forward_step(encoded, y, None)
            a_yhat, a_feature = forward_step(encoded, y, temp_embeddings)
            criterion_loss1 = class_criterion(n_yhat, y)
            criterion_loss2 = class_criterion(a_yhat, y)
        
            contrastive_loss = contrastive_criterion(n_feature, a_feature)
            loss = criterion_loss1*0.25 + criterion_loss2*0.25 + contrastive_loss*0.5
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss.append(loss.item())
            
        train_loss = np.mean(train_loss)
        
        # validation
        val_loss, acc = validation()
        
        if acc >= best_acc:
            best_acc = acc
            best_model = model
            
        print(f"-- EPOCH {epoch} --")
        print("val_loss : ", round(val_loss, 4))
        print("accuracy : ", round(acc, 4))
        
        if epoch < configs['WARM_UP_EPOCHS']:
            warm_up.step()
        else:
            scheduler.step(acc)

    return best_model





def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def recursive_to_device(inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.to(target_device)
        if isinstance(inputs, (list, tuple)):
            return type(inputs)(recursive_to_device(e) for e in inputs)
        if isinstance(inputs, dict):
            return type(inputs)((k, recursive_to_device(v)) for k, v in inputs.items())
        return inputs

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            ret = {}
            for k in out:
                if isinstance(out[k], torch.Tensor):
                    ret[k] = Gather.apply(target_device, dim, *[d[k] for d in outputs])
                else:
                    ret[k] = type(out[k])([recursive_to_device(e) for d in outputs for e in d[k]])
            return type(out)(ret)
        return type(out)(map(recursive_to_device, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        recursive_to_device = None
        gather_map = None
    return res




# -------------------------------------------------
def __OBSOLETE_prepare_data():
    datasets.logging.set_verbosity(verbosity=False)
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')   
    df = dataset['train'].to_pandas()

    gender_df = df[(df["target_gender"]==True)][target_columns]
    sexual_df = df[(df["target_sexuality"]==True)][target_columns]

    gender_ids = gender_df.index.tolist()
    sexual_ids = sexual_df.index.tolist()

    gender_ids = gender_ids + sexual_ids
    general_ids = list(set(df.index.tolist())-set(gender_ids))

    # gender
    gender = df.loc[gender_ids][target_columns]
    gender_good = gender[(gender['hate_speech_score']<0)].reset_index(drop=True)
    gender_bad = gender[(gender['hate_speech_score']>0)].reset_index(drop=True)

    # general
    general = df.loc[general_ids][target_columns]
    general_good = general[(general['hate_speech_score']<0)].reset_index(drop=True)
    general_bad = general[(general['hate_speech_score']>0)].reset_index(drop=True)

    print()
    print(f"gender  good comments : {len(gender_good)}")
    print(f"gender  bad  comments : {len(gender_bad)}")
    print(f"general good comments : {len(general_good)}")
    print(f"general bad  comments : {len(general_bad)}")
    print()

    gender_good['gender'] = True
    gender_bad['gender'] = True
    general_good['gender'] = False
    general_bad['gender'] = False

    gender_bad['biased'] = True
    general_bad['biased'] = True
    gender_good['biased'] = False
    general_good['biased'] = False

    full_data = pd.concat([gender_good, gender_bad, general_good, general_bad], axis=0).reset_index(drop=True)
    full_data.drop(columns=['target_sexuality', 'target_gender'], inplace=True)

    print(f"full data : {len(full_data)}")
    
    return full_data



