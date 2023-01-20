from loadlibs import *
import modules
import loss_fn


CFG = {
    'SEED' : 1203,
    'BATCH_SIZE' : 64,
    'EXP_BATCH_SIZE':64, 
    'EPOCHS' : 10,
    'WARM_UP_EPOCHS' : 5,
    'TQDM' : True,
    
    'LEARNING_RATE' : 0.005,
    'TEST_SIZE' : 0.25,
}

DATA = {
    'DATA_SIZE' : 0.2,
    'DATASET1' : 'ucberkeley-dlab/measuring-hate-speech', # thru dataset module
    'DATASET2' : '/home/gyuseonglee/workspace/biascfr/data/HateXplain.json',
    'DATASET3' : [
        '/home/gyuseonglee/workspace/biascfr/data/unintended/identity_individual_annotations.csv',
        '/home/gyuseonglee/workspace/biascfr/data/unintended/all_data.csv',
        ],
}

TOOLS = {
    'DEVICE' : 'cuda',
    
    'MODEL' : modules.BasicClassifier,
    # 'BACKBONE' : ElectraModel.from_pretrained("google/electra-small-discriminator"),
    # 'TOKENIZER' : ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator"),
    'BACKBONE' : AutoModel.from_pretrained('distilbert-base-uncased'),
    'TOKENIZER' : DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'),
    'LINEAR_DIM' : 768,
    
    
    'OPTIMIZER' : torch.optim.Adam,
    'CRITERION' : torch.nn.BCEWithLogitsLoss(),
    'CLASS_CRITERION' : torch.nn.BCEWithLogitsLoss(),
    'CONTRASTIVE_CRITERION' : None,
    'SCHEDULER' : torch.optim.lr_scheduler.ReduceLROnPlateau,
    'SCHEDULER_ARGS' : {
        'mode':'max',
        'factor':0.75,
        'patience':1,
        'threshold_mode':'abs',
        'min_lr':CFG['LEARNING_RATE']/0.1,
        'verbose':True,
        },
    'WARM_UP' : torch.optim.lr_scheduler.LinearLR,
    'WARM_UP_ARGS' : {
        'start_factor':CFG['LEARNING_RATE']/CFG['WARM_UP_EPOCHS'],
        'end_factor' : CFG['LEARNING_RATE'],
        'total_iters': CFG['WARM_UP_EPOCHS'],
        }
    
}



target_columns = [
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "hatespeech",
    "hate_speech_score",
    "text",
    "target_sexuality",
    "target_gender",
]








