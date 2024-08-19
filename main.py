from config import get_args
from dataset_preparation import load_model
from attack import CrossEntropyAttack

def main():
    # Parse command-line arguments
    args = get_args()

    # Load the appropriate model based on the dataset and model type
    model = load_model(args.dataset, args.model_type)

    # Instantiate the CrossEntropyAttack class
    cea = CrossEntropyAttack(
        max_iterations=args.max_iterations, 
        num_candidates=args.num_candidates, 
        rho=args.rho
    )

    # Perform the attack
    adversarial_example = cea.attack_text(
        args.text, 
        args.true_label, 
        attack_type=args.attack_type, 
        model=model, 
        nmt=(args.model_type == "nmt")
    )
    
    print(f"Adversarial Example ({args.attack_type}):", adversarial_example)

if __name__ == "__main__":
    main()