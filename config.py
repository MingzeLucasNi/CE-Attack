import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Cross-Entropy Attack on NLP models")
    
    parser.add_argument("--text", type=str, required=True, help="Text to attack")
    parser.add_argument("--true_label", type=str, required=True, help="True label of the text")
    parser.add_argument("--dataset", type=str, required=True, choices=["ag_news", "emotion", "sst2"], help="Dataset name")
    parser.add_argument("--model_type", type=str, required=True, choices=["classifier", "nmt"], help="Type of model")
    parser.add_argument("--attack_type", type=str, required=True, choices=["soft", "hard"], help="Type of attack")
    parser.add_argument("--max_iterations", type=int, default=100, help="Maximum iterations for the attack")
    parser.add_argument("--num_candidates", type=int, default=10, help="Number of candidates to generate in each iteration")
    parser.add_argument("--rho", type=float, default=0.2, help="Proportion of top candidates to retain for updating theta")
    
    return parser.parse_args()