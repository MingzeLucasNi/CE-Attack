import numpy as np
import random
from transformers import pipeline
from uti import *
# Sample models for testing
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
nmt_model = pipeline('translation_en_to_fr')

# Define functions for soft-label, hard-label, and NMT performance measures

def soft_label_performance(predictions, true_class, num_classes):
    confidence = predictions[true_class]['score']
    return max(1 - confidence, 1 - 1/num_classes)

def hard_label_performance(predictions, true_class):
    return 1 if predictions[0]['label'] != true_class else 0

def nmt_performance(translation, reference, bleu_score_function):
    bleu = bleu_score_function(reference, translation)
    sem = semantic_similarity(reference, translation)  # Placeholder for semantic similarity function
    return 1 - bleu * sem

# Cross-Entropy Attack implementation

class CrossEntropyAttack:
    def __init__(self, max_iterations=100, num_candidates=10, rho=0.2):
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        self.rho = rho

    def generate_substitutions(self, word, mlm_model, thesaurus):
        # Generate substitutions from masked language model (MLM) and thesaurus
        masked_input = "[MASK]".join(word)
        mlm_suggestions = mlm_model(masked_input)
        thesaurus_suggestions = thesaurus.get(word, [])
        return list(set(mlm_suggestions).intersection(thesaurus_suggestions))

    def attack_text(self, text, true_label, attack_type="soft", model=classifier, nmt=False):
        words = text.split()
        substitution_sets = [self.generate_substitutions(w, mlm_model=classifier, thesaurus={}) for w in words]
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
                    translation = nmt_model(candidate_text)[0]['translation_text']
                    ref_translation = nmt_model(text)[0]['translation_text']
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

            for i in range(len(words)):
                for j in range(len(substitution_sets[i])):
                    theta[i][j] = sum(1 for candidate in top_candidates if candidate[i] == substitution_sets[i][j])
                theta[i] /= sum(theta[i])

        # Select the final adversarial example
        final_words = [subs[np.argmax(t)] for subs, t in zip(substitution_sets, theta)]
        return " ".join(final_words)


# Example usage

# Soft-label attack on sentiment classifier
text = "This movie was absolutely amazing, with stellar performances and a gripping plot."
true_label = "POSITIVE"
cea = CrossEntropyAttack()
adversarial_example = cea.attack_text(text, true_label, attack_type="soft", model=classifier)
print("Adversarial Example (Soft-Label):", adversarial_example)

# Hard-label attack on sentiment classifier
adversarial_example_hard = cea.attack_text(text, true_label, attack_type="hard", model=classifier)
print("Adversarial Example (Hard-Label):", adversarial_example_hard)

# NMT attack
nmt_text = "The quick brown fox jumps over the lazy dog."
adversarial_nmt = cea.attack_text(nmt_text, true_label=None, nmt=True)
print("Adversarial Example (NMT):", adversarial_nmt)