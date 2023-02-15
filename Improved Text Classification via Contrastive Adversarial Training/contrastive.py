from loadlibs import *
import functions
from loss_fn import NTXentLoss

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# model
model_name = 'distilbert-base-uncased'



configs = dict()
# train setting
configs['SEED'] = 1203
configs['DATA_SIZE'] = 0.2
configs['TEST_SIZE'] = 0.25
configs['BATCH_SIZE'] = 64
configs['EPOCHS'] = 20
configs['LEARNING_RATE'] = 0.0001

# devices
configs['DEVICE'] = 'cuda'
configs['NUM_GPUS'] = torch.cuda.device_count()
configs['NUM_WORKERS'] = 4

# dataset
configs['DATA_DIR'] = "./full_dataset.csv"
configs['DATASET1'] = 'ucberkeley-dlab/measuring-hate-speech'
configs['DATASET2'] = '/home/n7/gyuseong/workspace/biascfr/data/HateXplain.json'
configs['DATASET3'] = [
            '/home/n7/gyuseong/workspace/biascfr/data/unintended/identity_individual_annotations.csv',
            '/home/n7/gyuseong/workspace/biascfr/data/unintended/all_data.csv',
        ]

# module
configs['TOKENIZER'] = DistilBertTokenizerFast.from_pretrained(model_name)
configs['ENCODER'] = AutoModel.from_pretrained(model_name)
configs['LINEAR_DIM'] = 768

# save files
folder_name = f"./checkpoints/{model_name}_{configs['SEED']}"



class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mode='train'):
        self.text = data.text.values
        if mode in ['train', 'valid']:
            self.gender = data.gender.astype(int).values.astype(int)
            self.toxic = data.toxic.astype(int).values.astype(int)
        else:
            self.gender = None
            self.toxic = None
        self.tokenizer = tokenizer
        self.mode = mode
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.text[idx],
            add_special_tokens=True,
            max_length = 256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        ids = encoded["input_ids"].squeeze(0)
        mask = encoded["attention_mask"].squeeze(0)
        encoded = {'input_ids':ids, 'attention_mask':mask}
        
        if self.mode in ['train', 'valid']:
            return (encoded, torch.tensor([self.gender[idx], self.toxic[idx]]).float())
        else:
            return encoded
       
       
class SentenceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data, tokenizer):
        super().__init__()
        self.data = data
        train, valid = train_test_split(self.data, test_size=configs['TEST_SIZE'], random_state=configs['SEED'])
        valid, test = train_test_split(valid, test_size=0.5, random_state=configs['SEED'])
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.valid = valid
        self.test = test
        self.tokenizer = tokenizer
        self.train_dataset = SentenceDataset(
            data=self.train,
            tokenizer=self.tokenizer,
            mode='train',
        )
        self.val_dataset = SentenceDataset(
            data=self.valid,
            tokenizer=self.tokenizer,
            mode='valid',
        )
        self.test_dataset = SentenceDataset(
            data=self.test,
            tokenizer=self.tokenizer,
            mode='test',
        )   

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )
            
            
class MyModule(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.ctr_loss_fn = NTXentLoss(
            batch_size=configs['BATCH_SIZE'], 
            temperature=0.07,
            use_cosine_similarity=True,
        )
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        
    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.model, "madgradw", lr=self.hparams.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_config]
    
    def forward(self, input_ids, attention_mask, adv):
        return self.model(input_ids, attention_mask, adv)
    
    def training_step(self, batch, batch_idx):
        encoded, y = batch
        input_ids, attention_mask = encoded['input_ids'], encoded['attention_mask']
        yhat, features = self(input_ids, attention_mask, False)
        
        """  forward  """
        self.model.zero_grad()
        # first loss calculation (negative loss)
        init_pred = yhat.detach().cpu().numpy() > 0.5
        init_pred = init_pred.astype(int).tolist()
        # adversarial attack
        negative_loss = -self.cls_loss_fn(yhat, y)
        negative_loss.backward()
        
        # get adversarial sample
        grad = self.model.embeddings.word_embeddings.weight.grad.detach().cpu().clone()    
        adv_grad = torch.nn.Parameter(self.model.eps*grad).to(f"cuda:{torch.cuda.current_device()}")
        self.model.embeddings_adv_grad = adv_grad

        n_yhat, n_feature = self(input_ids, attention_mask, False)
        a_yhat, a_feature = self(input_ids, attention_mask, True)
        
        """  loss 1 : classification loss  """
        criterion_loss1 = self.cls_loss_fn(n_yhat, y)
        criterion_loss2 = self.cls_loss_fn(a_yhat, y)
        
        """  loss 2 : contrastive loss  """
        contrastive_loss = self.ctr_loss_fn(n_feature, a_feature)

        """  backward  """
        loss = criterion_loss1*0.25 + criterion_loss2*0.25 + contrastive_loss*0.5
        
        self.train_accuracy(n_yhat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )        
        return loss 
        
    def validation_step(self, batch, batch_idx):
        encoded, y = batch
        input_ids, attention_mask = encoded['input_ids'], encoded['attention_mask']
        yhat, features = self(input_ids, attention_mask, False)
        loss = self.cls_loss_fn(input=yhat, target=y)
        self.val_accuracy(yhat, y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_epoch=True)
    

class ExpClassifier(torch.nn.Module):
    def __init__(self, encoder, linear_dim=1024):
        super().__init__()
        self.activate = torch.nn.GELU(approximate='tanh')
        self.head_mask = [None]*encoder.config.num_hidden_layers
        
        self.embeddings = encoder.embeddings
        self.embeddings_adv = encoder.embeddings
        self.embeddings_adv_grad = torch.zeros(0)
        self.transformer = encoder.transformer
        self.CrossEntropyLinear = torch.nn.Linear(linear_dim, 2)
        self.ContrastiveLinear = torch.nn.Sequential(
            torch.nn.Linear(linear_dim, 256),
            self.activate,
            torch.nn.Linear(256, 64)
        )
        
        self.eps = 0.05
        
    def forward(self, input_ids, attention_mask, embedding_adv=False):        
        # adversarial attack
        if embedding_adv == True:
            with torch.no_grad():
                device_idx = torch.cuda.current_device()
                self.embeddings_adv.weights = self.embeddings.word_embeddings.weight.to(f"cuda:{device_idx}") - self.embeddings_adv_grad.to(f"cuda:{device_idx}")
                x = self.embeddings_adv(input_ids)
        else: 
            x = self.embeddings(input_ids)
            
        x = self.transformer(
            x=x, 
            attn_mask=attention_mask,
            head_mask=self.head_mask,
            output_attentions=[None],
            output_hidden_states=None,
            return_dict=True,
        )['last_hidden_state'][:, 0, :]
        
        yhat = self.CrossEntropyLinear(x)
        feature = self.ContrastiveLinear(x)
        
        return yhat, feature
    
       
def main(): 
    # load data
    try:
        data = pd.read_csv(configs['DATA_DIR'])
    except:
        data = functions.prepare_data(configs)
        data.to_csv(configs['DATA_DIR'], index=False)

    logger.info("training start")
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info("load model")
    model = ExpClassifier(encoder=configs['ENCODER'], linear_dim=configs['LINEAR_DIM'])
    datamodule = SentenceDataModule(
        batch_size=configs['BATCH_SIZE'], 
        num_workers=configs['NUM_WORKERS'], 
        data=data, 
        tokenizer=configs['TOKENIZER']
    )
    module = MyModule(model=model, lr=configs['LEARNING_RATE'])
    
    checkpoints = ModelCheckpoint(dirpath=folder_name, monitor="val_acc", mode="max")
    callbacks = [checkpoints, RichProgressBar(), LearningRateMonitor()]
    # callbacks = []
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trainer = pl.Trainer(
        gpus=configs['NUM_GPUS'],
        accelerator="gpu", strategy="ddp",
        logger=WandbLogger(name=f"{model_name}_{now}", project="sentence_cfr"),
        callbacks=callbacks,
        max_epochs=configs['EPOCHS'],
        precision=16,
        # fast_dev_run=True,
    )
    
    logger.info("start training")
    trainer.fit(module, datamodule=datamodule)
    logger.info("training end")
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    torch.distributed.destroy_process_group()
    
    
if __name__ == "__main__":
    main()