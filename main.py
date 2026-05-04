import yaml
import torch
from data.assistments_dataset import AssistmentsDataset
from data.kt_dataset import KTSequenceDataset
from models.mirt import MIRTModel, MIRTModelEM
from train.train_mirt import train_mirt
from train.train_mirt_elo import train_mirt_elo
from train.train_mgirt_elo import train_mgirt_elo
from train.train_sakt import train_sakt
from eval.evaluate_mirt import evaluate_mirt
from eval.evaluate_kt import evaluate_kt
from eval.evaluate_mgirt import evaluate_mgirt
import numpy as np
from sklearn.model_selection import train_test_split

# Load config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def split_dataset(dataset, test_size=0.2, seed=42):
    idxs = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(idxs, test_size=test_size, random_state=seed)
    class Subset(torch.utils.data.Dataset):
        def __init__(self, dataset, idxs):
            self.dataset = dataset
            self.idxs = idxs
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, i):
            return self.dataset[self.idxs[i]]
    return Subset(dataset, train_idx), Subset(dataset, test_idx)

def main():
    config = load_config('config.yaml')['experiment']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = AssistmentsDataset(config['dataset_path'])
    train_set, test_set = split_dataset(dataset, test_size=0.2, seed=config['seed'])
    num_students = len(dataset.students)
    num_items = len(dataset.items)
    # print('Training standard MIRT...')
    # mirt, _ = train_mirt(train_set, emb_dim=config['emb_dim'], batch_size=config['batch_size'], lr=config['lr'], epochs=config['epochs'])
    # print('Evaluating standard MIRT:')
    # print('Train:')
    # evaluate_mirt(mirt, train_set)
    # print('Test:')
    # evaluate_mirt(mirt, test_set)
    # print('\nTraining EM MIRT...')
    # mirt_em = MIRTModelEM(num_students, num_items, emb_dim=config['emb_dim'])
    # mirt_em.em_train(train_set, batch_size=config['batch_size'], lr=config['lr'], epochs=config['epochs'])
    # print('Evaluating EM MIRT:')
    # print('Train:')
    # evaluate_mirt(mirt_em, train_set)
    # print('Test:')
    # evaluate_mirt(mirt_em, test_set)

    # if config.get('run_mirt_elo'):
    #     elo_cfg = config.get('mirt_elo', {})
    #     print('\nTraining MIRT (Elo-style updates)...')
    #     mirt_elo, _ = train_mirt_elo(
    #         train_set,
    #         emb_dim=elo_cfg.get('emb_dim', config['emb_dim']),
    #         lr=elo_cfg.get('lr', 0.05),
    #         epochs=elo_cfg.get('epochs', 1),
    #         lr_decay=elo_cfg.get('lr_decay', 0.0),
    #         decay_mode=elo_cfg.get('decay_mode', 'sqrt'),
    #         l2=elo_cfg.get('l2', 0.0),
    #         init_std=elo_cfg.get('init_std', 0.01),
    #         device=device,
    #     )
    #     print('Evaluating MIRT (Elo-style):')
    #     print('Train:')
    #     evaluate_mirt(mirt_elo, train_set)
    #     print('Test:')
    #     evaluate_mirt(mirt_elo, test_set)

    if config.get('run_mgirt_elo'):
        mg_cfg = config.get('mgirt_elo', {})
        print('\nTraining MG-IRT (Elo-style updates)...')
        mgirt, _ = train_mgirt_elo(
            train_set,
            emb_dim=mg_cfg.get('emb_dim', config['emb_dim']),
            alpha=mg_cfg.get('alpha', 1.0),
            beta=mg_cfg.get('beta', 1.0),
            lr_item=mg_cfg.get('lr_item', 0.01),
            lr_pos=mg_cfg.get('lr_pos', 0.05),
            lr_neg=mg_cfg.get('lr_neg', 0.05),
            epochs=mg_cfg.get('epochs', 1),
            lr_decay=mg_cfg.get('lr_decay', 0.0),
            decay_mode=mg_cfg.get('decay_mode', 'sqrt'),
            l2=mg_cfg.get('l2', 0.0),
            init_std=mg_cfg.get('init_std', 0.01),
            normalize_items=mg_cfg.get('normalize_items', False),
            device=device,
        )
        print('Evaluating MG-IRT (Elo-style):')
        print('Train:')
        evaluate_mgirt(mgirt, train_set)
        print('Test:')
        evaluate_mgirt(mgirt, test_set)

    if config.get('run_sakt'):
        sakt_cfg = config.get('sakt', {})
        max_seq_len = sakt_cfg.get('max_seq_len', 50)
        emb_dim = sakt_cfg.get('emb_dim', 64)
        num_heads = sakt_cfg.get('num_heads', 4)
        dropout = sakt_cfg.get('dropout', 0.2)
        train_ratio = sakt_cfg.get('train_ratio', 0.8)
        lr = sakt_cfg.get('lr', 1e-3)
        batch_size = sakt_cfg.get('batch_size', 128)
        epochs = sakt_cfg.get('epochs', 10)

        print('\nTraining SAKT (chronological split)...')
        sakt_model, sakt_train = train_sakt(
            config['dataset_path'],
            max_seq_len=max_seq_len,
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            train_ratio=train_ratio,
            device=device,
        )
        sakt_test = KTSequenceDataset(
            config['dataset_path'],
            max_seq_len=max_seq_len,
            train_ratio=train_ratio,
            split='test',
        )
        print('Evaluating SAKT:')
        print('Train:')
        evaluate_kt(sakt_model, sakt_train, device=device)
        print('Test:')
        evaluate_kt(sakt_model, sakt_test, device=device)

if __name__ == '__main__':
    main()
