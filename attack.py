import numpy as np
import random
from uti import generate_substitution_sets, calculate_sentence_similarity, calculate_rouge_score

# Define functions for soft-label, hard-label, and NMT performance measures

def soft_label_performance(predictions, true_class, num_classes):
    confidence = predictions[true_class]['score']
    return max(1 - confidence, 1 - 1/num_classes)

def hard_label_performance(predictions, true_class):
    return 1 if predictions[0]['label'] != true_class else 0

def nmt_performance(translation, reference, bleu_score_function):
    bleu = bleu_score_function(reference, translation)
    sem = calculate_sentence_similarity(reference, translation)
    return 1 - bleu * sem

# Cross-Entropy Attack implementation

class CrossEntropyAttack:
    def __init__(self, max_iterations=100, num_candidates=10, rho=0.2):
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        self.rho = rho

    def attack_text(self, text, true_label, attack_type="soft", model=None, nmt=False):
        if model is None:
            raise ValueError("Model must be provided for the attack.")

        substitution_sets, _ = generate_substitution_sets(text, mlm_model=model)
        theta = [np.ones(len(subs)) / len(subs) for subs in substitution_sets]
        gamma = 0.5

        for t in range(self.max_iterations):
            candidates = []
            for _ in range(self.num_candidates):
                candidate = [random.choices(subs, theta[i])[0] for i, subs in enumerate(substitution_sets)]
                candidates.append(candidate)

            # Evaluate performance for each candidate
            performances = []
            for candidate in candidates:
                candidate_text = " ".join(candidate)
                if nmt:
                    translation = model(candidate_text)[0]['translation_text']
                    ref_translation = model(text)[0]['translation_text']
                    performance = nmt_performance(translation, ref_translation, bleu_score_function=lambda ref, hyp: 0.7)
                else:
                    predictions = model(candidate_text)
                    if attack_type == "soft":
                        performance = soft_label_performance(predictions, true_label, len(predictions))
                    else:
                        performance = hard_label_performance(predictions, true_label)
                performances.append(performance)

            # Update gamma and theta
            sorted_indices = np.argsort(performances)
            top_candidates = [candidates[i] for i in sorted_indices[int((1 - self.rho) * self.num_candidates):]]

            for i in range(len(substitution_sets)):
                for j in range(len(substitution_sets[i])):
                    theta[i][j] = sum(1 for candidate in top_candidates if candidate[i] == substitution_sets[i][j])
                theta[i] /= sum(theta[i])

        # Select the final