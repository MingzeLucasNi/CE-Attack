from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from sklearn.neighbors import NearestNeighbors
import torch
from torch.nn import functional as F
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool

import copy
import OpenHowNet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from transformers import RobertaTokenizer

# Initialize global variables
detokenizer = TreebankWordDetokenizer()
hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)

# Utility functions

def calculate_rouge_score(input_sentence: str, reference_sentence: str, rouge_type: str = "rouge1") -> float:
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(input_sentence, reference_sentence)[rouge_type].precision
    return scores

def calculate_sentence_similarity(input_sentence, reference_sentence, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    sentence_transformer = SentenceTransformer(model_name)
    cosine_similarity_function = nn.CosineSimilarity(dim=0, eps=1e-6)

    sentence_embeddings = sentence_transformer.encode([input_sentence, reference_sentence])
    similarity_score = cosine_similarity_function(torch.from_numpy(sentence_embeddings[0]), torch.from_numpy(sentence_embeddings[1])).item()

    return similarity_score

def victim_model(text, label, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model(inputs).logits.squeeze()
    right_prob = F.softmax(outputs, dim=-1)[label]
    prob = F.softmax(outputs, dim=-1).squeeze()
    predicted_label = torch.argmax(prob).item()
    True_label = label

    return True_label, right_prob.item(), predicted_label

def get_synonyms(word):
    return hownet_dict_advanced.get_nearest_words(word, language='en', K=10, merge=True)

class MLM:
    def __init__(self, K):
        self.K = K
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.model = AutoModelForMaskedLM.from_pretrained('roberta-large')
    
    def cdts(self, mask_text):
        mask_id = 50264
        inputs = self.tokenizer.encode(mask_text, return_tensors="pt")
        mask_position = (inputs.squeeze() == mask_id).nonzero().squeeze()
        mask_logits = self.model(inputs).logits.squeeze()[mask_position]
        top_k = torch.sort(mask_logits, descending=True).indices[:self.K]
        return [self.tokenizer.decode([id]) for id in top_k]

def generate_substitution_sets(sentence, mlm_model, num_cpu=4):
    tokens = word_tokenize(sentence)
    mlm_subs = []
    synonym_subs = []
    final_substitution_sets = []
    attackable = []

    # Generate MLM candidates for each word by masking it and running it through the MLM model
    for i in range(len(tokens)):
        masked_sentence = copy.deepcopy(tokens)
        masked_sentence[i] = '<mask>'
        masked_sentence = ' '.join(masked_sentence)
        
        mlm_candidates = mlm_model.cdts(masked_sentence)
        
        if tokens[i] in mlm_candidates:
            mlm_candidates.remove(tokens[i])
        
        mlm_subs.append(set(mlm_candidates))

    # Generate synonyms for each word in the sentence using multiprocessing
    with Pool(num_cpu) as pool:
        synonym_subs = pool.map(get_synonyms, tokens)

    # Combine MLM candidates and synonyms into a single substitution set for each word
    for i in range(len(tokens)):
        combined_subs = mlm_subs[i].union(set(synonym_subs[i]))
        final_substitution_sets.append(combined_subs)
        
        attackable.append(len(combined_subs) > 0)

    return final_substitution_sets, attackable

class victim_models:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.detokenizer = TreebankWordDetokenizer()
        self.word_tokenize = word_tokenize
    
    def logits(self, text, label):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model(inputs).logits.squeeze()
        return outputs[label]

    def mr_words_saliency(self, v, text, label):
        tokens = self.word_tokenize(text)
        ori_logit = self.logits(text, label)
        saliency = []
        for i in v:
            rem_text = copy.deepcopy(tokens)
            rem_text.pop(i)
            rem_logit = self.logits(self.detokenizer.detokenize(rem_text), label)
            saliency.append(ori_logit - rem_logit)
        return saliency

    def words_saliency(self, text, label):
        tokens = self.word_tokenize(text)
        ori_logit = self.logits(text, label)
        saliency = []
        for i in range(len(tokens)):
            rem_text = copy.deepcopy(tokens)
            rem_text.pop(i)
            rem_logit = self.logits(self.detokenizer.detokenize(rem_text), label)
            saliency.append(ori_logit - rem_logit)
        return saliency

    def prob(self, text, label):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model(inputs).logits.squeeze()
        return F.softmax(outputs, dim=0)[label]

    def predict(self, text):
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        outputs = self.model(inputs).logits.squeeze()
        prob = F.softmax(outputs, dim=0).squeeze()
        return torch.argmax(prob)