"""
config_and_utils.py - Fixed version with proper initialization
"""
import logging
import os
import random
import numpy as np
import pandas as pd
import torch
import time
from sklearn.metrics import roc_curve, auc, top_k_accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import nltk
logger = logging.getLogger(__name__)
# Download the 'punkt' tokenizer models


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Config:
    """Unified configuration for both traditional and graph-based methods."""

    def __init__(self,dataset_path=None):
        # Initialize default values that will be updated later
        self.num_users = 500  # Default, will be updated after data loading
        self.node_feature_dim = 311  # Default spaCy dimension (300 + 11 features)
        
        # Ablation flags for experiments
        self.ablate_pos = False        # Disable POS tag features
        self.ablate_dep = False        # Disable dependency edge features
        self.pooling = "mean"          # Pooling: 'mean', 'max', or 'attention'

        # Dataset
        #self.DATASET_PATH = "/Users/shaivpatel/Downloads/update500 2/dataset/synthetic_prompt_biometrics_500x20.csv"
        default_path = "/Users/shaivpatel/Downloads/update500 2/dataset/synthetic_prompt_biometrics_2500x30.csv"
        self.DATASET_PATH = dataset_path if dataset_path else default_path
        # Reproducibility
        self.RANDOM_SEED = 42

        # Data splitting
        self.SEEN_USERS_RATIO = 0.8
        self.NUM_TRAIN_PROMPTS = 16
        self.NUM_VAL_PROMPTS = 2
        self.NUM_TEST_PROMPTS = 2

        # Traditional method hyperparameters
        self.TRADITIONAL = {
            'epochs': 70,  # Increased for better convergence
            'batch_size': 32,
            'lr':  1e-4,
            'lr_scheduler': 'ReduceLROnPlateau',
            'hidden_dim': 384,
            'early_stopping_patience': 5,  # Increased for better capacity
            'dropout': 0.1,
            'encoder_name': 'sentence-transformers/nli-roberta-base-v2',      
            'use_sentence_transformers': True 
        }
        #sentence-transformers/nli-roberta-base-v2
        #intfloat/e5-base
        #all-mpnet-base-v2
        #intfloat/e5-large
        #BAAI/bge-small-en-v1.5
        self.GRAPH = {
            'epochs': 75,  # Increased for better convergence
            'batch_size': 32,  # Reduced for memory efficiency
            'lr': 1e-4,
            'early_stopping_patience': 5,
            'gnn_hidden_channels': 256,
            'embedding_dim': 384,
            'dropout': 0.1,
            'training_mode': 'classification',  # or 'triplet'
            'triplet_margin': 0.5
        }
    
    # Device configuration
    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

def set_reproducibility(seed=42):
    """Set seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        import copy
        self.best_weights = copy.deepcopy(model.state_dict())

def create_consistent_splits(df, config):
    """Create consistent train/validation/test splits for both methods"""
    set_reproducibility(config.RANDOM_SEED)
    
    # Clean and validate data
    df = df.dropna(subset=['prompt']).reset_index(drop=True)
    df['prompt'] = df['prompt'].astype(str)
    
    # Get users with sufficient data
    user_counts = df.groupby('user_id').size()
    min_prompts = config.NUM_TRAIN_PROMPTS + config.NUM_VAL_PROMPTS + config.NUM_TEST_PROMPTS
    valid_users = user_counts[user_counts >= min_prompts].index.tolist()
    
    print(f"Total users: {len(df['user_id'].unique())}")
    print(f"Users with sufficient data: {len(valid_users)}")
    
    # Update config with actual number of users
    config.num_users = len(valid_users)
    
    # Split users into seen/unseen
    np.random.shuffle(valid_users)
    num_seen = int(len(valid_users) * config.SEEN_USERS_RATIO)
    seen_users = valid_users[:num_seen]
    unseen_users = valid_users[num_seen:]
    
    # Create train/validation/test splits for seen users
    train_data, val_data, test_seen_data = [], [], []
    
    for user_id in seen_users:
        user_df = df[df['user_id'] == user_id].copy()
        user_df = user_df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        
        # Sample training prompts
        train_prompts = user_df.iloc[:config.NUM_TRAIN_PROMPTS]
        
        # Sample validation prompts
        val_prompts = user_df.iloc[config.NUM_TRAIN_PROMPTS:config.NUM_TRAIN_PROMPTS + config.NUM_VAL_PROMPTS]
        
        # Sample test prompts
        test_prompts = user_df.iloc[config.NUM_TRAIN_PROMPTS + config.NUM_VAL_PROMPTS:
                                    config.NUM_TRAIN_PROMPTS + config.NUM_VAL_PROMPTS + config.NUM_TEST_PROMPTS]
        
        if len(train_prompts) > 0:
            train_data.append(train_prompts)
        if len(val_prompts) > 0:
            val_data.append(val_prompts)
        if len(test_prompts) > 0:
            test_seen_data.append(test_prompts)
    
    # Combine data
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
    test_seen_df = pd.concat(test_seen_data, ignore_index=True) if test_seen_data else pd.DataFrame()
    unseen_df = df[df['user_id'].isin(unseen_users)].copy()
    
    # Create consistent user ID mappings
    user_id_map = {user_id: idx for idx, user_id in enumerate(valid_users)}

    # Validation to prevent data leakage
    print("\nValidating data splits to prevent leakage...")
    
    seen_user_set = set(seen_users)
    unseen_user_set = set(unseen_users)
    
    train_users = set(train_df['user_id'].unique())
    val_users = set(val_df['user_id'].unique())
    test_seen_users = set(test_seen_df['user_id'].unique())

    assert seen_user_set.isdisjoint(unseen_user_set), \
        "CRITICAL LEAKAGE: Overlap found between seen and unseen user sets!"

    assert train_users.isdisjoint(unseen_user_set), \
        "CRITICAL LEAKAGE: Training data contains users from the unseen set!"
        
    print("Data splits are valid. No user overlap detected.")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test_seen': test_seen_df,
        'unseen': unseen_df,
        'user_id_map': user_id_map,
        'seen_users': seen_users,
        'unseen_users': unseen_users
    }

class BiometricEvaluator:
    """Unified evaluation framework for both methods"""
    
    def __init__(self):
        self.results = {}
        self.timing_results = {}
    
    def evaluate_method(self, embeddings_train, labels_train, embeddings_test, 
                       labels_test, embeddings_unseen, method_name, train_time=None):
        """Complete evaluation for a biometric method"""
        
        # Handle empty inputs
        if len(embeddings_train) == 0 or len(embeddings_test) == 0:
            print(f"Warning: Empty embeddings for {method_name}")
            return 0, 0, 0, 0.5
        
        # Create prototypes for seen users
        unique_labels = np.unique(labels_train)
        prototypes = {}
        for label in unique_labels:
            mask = labels_train == label
            prototypes[label] = np.mean(embeddings_train[mask], axis=0)
        
        prototype_matrix = np.array([prototypes[l] for l in unique_labels])
        
        # === IDENTIFICATION EVALUATION ===
        similarities = cosine_similarity(embeddings_test, prototype_matrix)
        predictions = np.argsort(-similarities, axis=1)
        
        # Map test labels to prototype indices
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        mapped_test_labels = []
        valid_test_embeddings = []
        
        for i, label in enumerate(labels_test):
            if label in label_to_idx:
                mapped_test_labels.append(label_to_idx[label])
                valid_test_embeddings.append(i)
        
        if len(mapped_test_labels) == 0:
            print(f"Warning: No valid test labels for {method_name}")
            return 0, 0, 0, 0.5
        
        mapped_test_labels = np.array(mapped_test_labels)
        valid_predictions = predictions[valid_test_embeddings]
        
        # Calculate top-k accuracy
        top1_acc = np.mean([mapped_test_labels[i] in valid_predictions[i][:1] 
                           for i in range(len(mapped_test_labels))])
        top3_acc = np.mean([mapped_test_labels[i] in valid_predictions[i][:3] 
                           for i in range(len(mapped_test_labels))])
        top5_acc = np.mean([mapped_test_labels[i] in valid_predictions[i][:5] 
                           for i in range(len(mapped_test_labels))])
        
        # === VERIFICATION EVALUATION ===
        # Scores for seen users (genuine)
        valid_similarities = similarities[valid_test_embeddings]
        scores_seen = valid_similarities[np.arange(len(mapped_test_labels)), mapped_test_labels] 
        target_fars = [1e-2, 1e-3, 1e-4, 1e-5]

        # Scores for unseen users (impostor)
        if len(embeddings_unseen) > 0:
            similarities_unseen = cosine_similarity(embeddings_unseen, prototype_matrix)
            scores_unseen = np.max(similarities_unseen, axis=1)
            
            # ROC analysis
            labels = np.concatenate([np.ones_like(scores_seen), np.zeros_like(scores_unseen)])
            scores = np.concatenate([scores_seen, scores_unseen])
            
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)

            # --- NEW: CALCULATE EER & D-PRIME ---
            fnr = 1 - tpr
            eer_index = np.nanargmin(np.abs(fnr - fpr)) 
            eer = fpr[eer_index]

            mu_g, mu_i = np.mean(scores_seen), np.mean(scores_unseen)
            std_g, std_i = np.std(scores_seen), np.std(scores_unseen)
            # Handle potential divide by zero if std dev is 0
            d_prime_denominator = np.sqrt(0.5 * (std_g ** 2 + std_i ** 2))
            d_prime = (abs(mu_g - mu_i) / d_prime_denominator) if d_prime_denominator > 0 else 0.0 
            # --- END NEW ---
            
            # Calculate FRR at specific FAR values
            from scipy.interpolate import interp1d
            interp_tpr = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(tpr[0], tpr[-1]))

            verification_results = {}
            for far in target_fars:
                tar_at_far = float(interp_tpr(far))  # TAR = TPR at this FAR
                frr_at_far = 1.0 - tar_at_far        # FRR = 1 - TAR
                verification_results[f'TAR_at_FAR_{far:.0e}'] = tar_at_far
                verification_results[f'FRR_at_FAR_{far:.0e}'] = frr_at_far

                # Optional: Log or print for visibility
                logger.info(f'TAR @ FAR={far:.3f}: {tar_at_far:.4f}')
                logger.info(f'FRR @ FAR={far:.3f}: {frr_at_far:.4f}')
            
            raw_scores = {
                'seen': scores_seen,
                'unseen': scores_unseen,
                'fpr': fpr,
                'tpr': tpr
            }

        else:
            # No unseen users for verification evaluation
            logger.warning(f"No unseen embeddings for {method_name}. Verification metrics (ROC-AUC, FRR) will be NaN.")
            roc_auc = np.nan
            eer = np.nan
            d_prime = np.nan
            verification_results = {
                f'FRR_at_FAR_{far:.0e}': np.nan for far in target_fars
            }
            raw_scores = {
                'seen': scores_seen,
                'unseen': np.array([]),
                'fpr': np.array([]),
                'tpr': np.array([])
            }
        
        # Store results
        self.results[method_name] = {
            'identification': {
                'top1': top1_acc,
                'top3': top3_acc, 
                'top5': top5_acc
            },
            'verification': {
                'roc_auc': roc_auc,
                'eer': eer,          
                'd_prime': d_prime,   
                **verification_results
            },
            'raw_scores': raw_scores
        }
        
        if train_time:
            self.timing_results[method_name] = {'train_time': train_time}
        
        return top1_acc, top3_acc, top5_acc, roc_auc
    



    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*60)
        print("        PROMPT BIOMETRICS RESEARCH RESULTS")
        print("="*60)
        
        print(f"\n{'Method':<15} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'ROC-AUC':<8} {'EER':<8} {'d-prime':<8} {'Train Time':<12}")
        print("-" * 85)
        
        for method, results in self.results.items():
            id_results = results['identification']
            ver_results = results['verification']
            train_time = self.timing_results.get(method, {}).get('train_time', 0)
            
            print(f"{method:<15} {id_results['top1']:<8.3f} {id_results['top3']:<8.3f} "
                  f"{id_results['top5']:<8.3f} {ver_results['roc_auc']:<8.3f} "
                  f"{ver_results['eer']:<8.3f} {ver_results['d_prime']:<8.3f} "
                  f"{train_time:<12.1f}s")

def log_experiment_setup(config, data_splits):
    """Log experimental setup for reproducibility"""
    print(f"\n=== EXPERIMENTAL SETUP ===")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"Dataset: {config.DATASET_PATH}")
    print(f"Seen users: {len(data_splits['seen_users'])}")
    print(f"Unseen users: {len(data_splits['unseen_users'])}")
    print(f"Training samples: {len(data_splits['train'])}")
    print(f"Validation samples: {len(data_splits['validation'])}")
    print(f"Test samples (seen): {len(data_splits['test_seen'])}")
    print(f"Test samples (unseen): {len(data_splits['unseen'])}")
    print("=" * 30)
