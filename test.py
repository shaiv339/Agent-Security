"""
run_experiments.py

The main driver script for running a full suite of experiments for
the SBERT + Stylometry biometric model.

This script will:
1.  Load a dataset from a command-line argument.
2.  Run a full Ablation Study (Combined, SBERT-Only, Stylo-Only).
3.  Run a full Robustness Analysis (Paraphrase, Synonym, Homoglyph attacks).
4.  Print a final summary report.

Usage:
    python run_experiments.py --dataset_path /path/to/your/dataset.csv
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
from attacks import HomoglyphAttacker, ParaphraseAttacker, SynonymAttacker
# Import our upgraded local files
from config_and_utils import (BiometricEvaluator, Config,
                              create_consistent_splits, set_reproducibility)
from traditional_method import TraditionalBiometrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_training(config, data_splits, ablation_mode='combined'):
    """
    Trains one instance of the TraditionalBiometrics model.
    Returns the trained model and its training data (embeddings/labels).
    """
    logger.info(f"--- Training Model: Mode '{ablation_mode}' ---")
    
    # 1. Initialize model with the correct ablation mode
    traditional_model = TraditionalBiometrics(config, ablation_mode=ablation_mode)
    
    # 2. Extract features for train, val, and unseen
    train_features, _, _ = traditional_model.extract_features(data_splits['train']['prompt'])
    train_labels = np.array([data_splits['user_id_map'][uid] 
                           for uid in data_splits['train']['user_id']])
    
    val_features, _, _ = traditional_model.extract_features(data_splits['validation']['prompt'])
    val_labels = np.array([data_splits['user_id_map'][uid] 
                         for uid in data_splits['validation']['user_id']])
    
    unseen_features, _, _ = traditional_model.extract_features(data_splits['unseen']['prompt'])

    # 3. Train the model
    training_time = traditional_model.train(train_features, train_labels, val_features, val_labels)
    
    # 4. Generate embeddings needed for evaluation
    train_embeddings = traditional_model.generate_embeddings(train_features)
    unseen_embeddings = traditional_model.generate_embeddings(unseen_features)
    
    logger.info(f"Training for '{ablation_mode}' complete. Time: {training_time:.2f}s")
    
    return traditional_model, train_embeddings, train_labels, unseen_embeddings, training_time

def evaluate_model(model, evaluator, eval_name, train_embeds, train_labels, unseen_embeds, test_prompts, test_labels, train_time):
    """
    Evaluates a trained model on a given set of test prompts.
    """
    logger.info(f"Evaluating: {eval_name}")
    
    # 1. Extract features from the test prompts
    test_features, _, _ = model.extract_features(test_prompts)
    
    # 2. Generate embeddings for the test prompts
    test_embeddings = model.generate_embeddings(test_features)
    
    # 3. Run the evaluation
    evaluator.evaluate_method(
        train_embeds,
        train_labels,
        test_embeddings,
        test_labels,
        unseen_embeds,
        eval_name,
        train_time
    )

def main(args):
    """
    Main orchestration function.
    """
    start_time = time.time()
    logger.info(f"--- STARTING FULL EXPERIMENT SUITE ---")
    logger.info(f"Dataset: {args.dataset_path}")
    
    # 1. Load Config and Data
    config = Config(dataset_path=args.dataset_path)
    set_reproducibility(config.RANDOM_SEED)
    df = pd.read_csv(config.DATASET_PATH)
    data_splits = create_consistent_splits(df, config)

    # 2. Prepare Data and Attackers
    logger.info("Initializing adversarial attackers (this may take a moment)...")
    attackers = {
        "Paraphrase": ParaphraseAttacker(),
        "Synonym": SynonymAttacker(),
        "Homoglyph": HomoglyphAttacker()
    }
    logger.info("Attackers loaded.")

    # Get original test prompts (the "unattacked" baseline)
    original_prompts = data_splits['test_seen']['prompt']
    original_labels = np.array([data_splits['user_id_map'][uid] 
                                for uid in data_splits['test_seen']['user_id']])

    # 3. Run Ablation Study
    # This involves training three separate models
    final_evaluator = BiometricEvaluator()
    champion_model = None
    champion_train_data = None
    unseen_embeddings_cache = {} # Cache unseen embeddings
    
    if not args.skip_ablation:
        logger.info(f"\n{'='*25} STARTING ABLATION STUDY {'='*25}")
        ablation_modes = ['combined', 'sbert_only', 'stylo_only']
        
        for mode in ablation_modes:
            model, train_embeds, train_labels, unseen_embeds, train_time = run_single_training(
                config, 
                data_splits, 
                ablation_mode=mode
            )
            
            # Evaluate this model on the *original* test data
            eval_name = f"Baseline ({mode})"
            evaluate_model(
                model, final_evaluator, eval_name,
                train_embeds, train_labels, unseen_embeds,
                original_prompts, original_labels,
                train_time
            )
            
            # Save the "combined" model as our champion for the attack phase
            if mode == 'combined':
                champion_model = model
                champion_train_data = (train_embeds, train_labels, train_time)
                unseen_embeddings_cache['combined'] = unseen_embeds

            # Cache unseen embeddings to avoid re-calculating
            unseen_embeddings_cache[mode] = unseen_embeds
            
        logger.info(f"--- ABLATION STUDY COMPLETE ---")

    else:
        logger.warning("Skipping Ablation Study. Will train 'combined' model for attacks.")
        # We still need to train the champion model for the attack phase
        model, train_embeds, train_labels, unseen_embeds, train_time = run_single_training(
            config, 
            data_splits, 
            ablation_mode='combined'
        )
        champion_model = model
        champion_train_data = (train_embeds, train_labels, train_time)
        unseen_embeddings_cache['combined'] = unseen_embeds


    # 4. Run Robustness (Attack) Analysis
    # We use the *one* champion model trained in 'combined' mode
    
    if not args.skip_attacks:
        logger.info(f"\n{'='*25} STARTING ROBUSTNESS ANALYSIS {'='*25}")
        
        # Get data from our champion model
        train_embeds, train_labels, train_time = champion_train_data
        unseen_embeds = unseen_embeddings_cache['combined']
        
        attack_configs = {
            "Synonym": {'attack_rate': 0.2},
            "Homoglyph": {'attack_rate': 0.15},
            "Paraphrase": {}
        }

        for attack_name, attacker in attackers.items():
            logger.info(f"--- Attack Run: {attack_name} ---")
            
            # 1. Generate attacked prompts
            kwargs = attack_configs[attack_name]
            attacked_prompts_list = attacker.attack(original_prompts.tolist(), **kwargs)
            attacked_prompts_series = pd.Series(attacked_prompts_list)
            
            # 2. Evaluate the *champion model* on the attacked prompts
            eval_name = f"Attacked ({attack_name})"
            evaluate_model(
                champion_model, final_evaluator, eval_name,
                train_embeds, train_labels, unseen_embeds,
                attacked_prompts_series, original_labels,
                train_time
            )
            
        logger.info(f"--- ROBUSTNESS ANALYSIS COMPLETE ---")
    else:
        logger.warning("Skipping Robustness Analysis.")

    # 5. Final Summary Report
    logger.info(f"\n\n{'='*30} FINAL SUMMARY REPORT {'='*30}")
    logger.info(f"Dataset: {args.dataset_path}")
    final_evaluator.generate_report()
    
    total_time = time.time() - start_time
    logger.info(f"--- FULL EXPERIMENT SUITE COMPLETE ---")
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full experiment suite for the SBERT+Stylometry model.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the .csv dataset file."
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the ablation study (Combined, SBERT-Only, Stylo-Only)."
    )
    parser.add_argument(
        "--skip-attacks",
        action="store_true",
        help="Skip the robustness analysis (Paraphrase, Synonym, Homoglyph)."
    )
    
    args = parser.parse_args()
    
    # Run the main experiment driver
    main(args)
