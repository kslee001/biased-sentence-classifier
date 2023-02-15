from loadlibs import *

def prepare_data(configs):
    gender_clean_dataset = []
    gender_dirty_dataset = []
    general_clean_dataset = []
    general_dirty_dataset = []
    dataset_columns = ['text', 'gender', 'toxic']

    """ first dataset : ucberkly measuring hate speech"""
    dataset = datasets.load_dataset(configs['DATASET1'], 'binary')   
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
    with open(configs['DATASET2']) as f:
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
    identity = pd.read_csv(configs['DATASET3'][0])
    gender_idx = identity[(identity['gender']=='male') | (identity['gender']=='female') | (identity['gender']=='male female')].id.drop_duplicates().tolist()
    train = pd.read_csv(configs['DATASET3'][1])
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
    if configs['DATA_SIZE'] != 1.0:
        print(f"--reduce the size of full dataset to {configs['DATA_SIZE']}")
        full_dataset = full_dataset.sample(frac=1).reset_index(drop=True)[:int(len(full_dataset)*configs['DATA_SIZE'])]
    print("full dataset size : ", len(full_dataset))
    return full_dataset



# def prepare_loader(data, configs, drop_last=False):
#     train, valid = train_test_split(data, test_size=configs['TEST_SIZE'], random_state=configs.['SEED'])
#     valid, test = train_test_split(valid, test_size=0.5, random_state=configs['SEED'])
    
#     train_dataset=dataset_type(
#         data=train,
#         tokenizer=TOOLS['TOKENIZER'],
#         mode='train',
#     )
#     valid_dataset=dataset_type(
#         data=valid,
#         tokenizer=TOOLS['TOKENIZER'],
#         mode='valid',
#     )
#     test_dataset=dataset_type(
#         data=test,
#         tokenizer=TOOLS['TOKENIZER'],
#         mode='test'
#     )
    
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset,
#         batch_size=configs['BATCH_SIZE'],
#         shuffle=True,
#         drop_last=drop_last,
#     )
#     valid_loader = torch.utils.data.DataLoader(
#         dataset=valid_dataset,
#         batch_size=configs['BATCH_SIZE'],
#         shuffle=True,
#         drop_last=drop_last,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         dataset=test_dataset,
#         batch_size=configs['BATCH_SIZE'],
#         shuffle=False,
#         drop_last=False,
#     )
    
#     return train_loader, valid_loader, test_loader