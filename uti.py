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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
detokenizer=TreebankWordDetokenizer()
hownet_dict_advanced = OpenHowNet.HowNetDict(init_sim=True)



def calculate_rouge_score(input_sentence: str, reference_sentence: str, rouge_type: str = "rouge1") -> float:
    """
    Calculate the ROUGE score between the input sentence and the reference sentence.
    
    Args:
        input_sentence: The input sentence to be compared with the reference sentence.
        reference_sentence: The reference sentence that the input sentence is compared with.
        rouge_type: The type of ROUGE score to calculate, default is "rouge1".
    
    Returns:
        The ROUGE precision score.
    """
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(input_sentence, reference_sentence)[rouge_type].precision
    return scores


def calculate_sentence_similarity(input_sentence, reference_sentence, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Calculate the cosine similarity between the input sentence and the reference sentence using a sentence transformer model.
    
    Args:
        input_sentence (str): The input sentence.
        reference_sentence (str): The reference sentence.
        model_name (str): The name of the sentence transformer model to use, default is 'sentence-transformers/all-MiniLM-L6-v2'.
    
    Returns:
        float: The cosine similarity score between the input and reference sentences.
    """
    sentence_transformer = SentenceTransformer(model_name)
    cosine_similarity_function = nn.CosineSimilarity(dim=0, eps=1e-6)

    sentence_embeddings = sentence_transformer.encode([input_sentence, reference_sentence])
    similarity_score = cosine_similarity_function(torch.from_numpy(sentence_embeddings[0]), torch.from_numpy(sentence_embeddings[1])).item()

    return similarity_score



def victim_model(text, label, model_name):
    """
    Get the probability for a specific label from a text using a pre-trained model.
    
    Args:
        text (str): The input text.
        label (int): The label for which the probability is to be retrieved.
        model_name (str): The name of the pre-trained model.
    
    Returns:
        torch.Tensor: The probability for the specified label.
    """
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
    c=hownet_dict_advanced.get_nearest_words(word, language='en',K=10, merge=True)
    return c




class MLM:
    def __init__(self,K):
        self.K=K
        self.tokenizer=RobertaTokenizer.from_pretrained("roberta-large")
        self.model = AutoModelForMaskedLM.from_pretrained('roberta-large')
    def cdts(self,mask_text):
        '''
        input: mask_text(string), number of candidates K (int)
        return: list of candidates (list)
        '''
        mask_id=50264
        inputs=self.tokenizer.encode(mask_text, return_tensors="pt")
        mask_position=(inputs.squeeze()==mask_id).nonzero().squeeze()
        mask_logits=self.model(inputs).logits.squeeze()[mask_position]
        top_k=torch.sort(mask_logits,descending=True).indices[:self.K]
        cand=[]
        for id in top_k:
            c=self.tokenizer.decode([id])
            cand.append(c)
        return cand
    
def generate_substitution_sets(sentence, mlm_model, num_cpu=4):
    """
    Generates a substitution set for each word in the sentence by combining MLM predictions and synonyms.
    
    Args:
        sentence (str): The input sentence.
        mlm_model (MLM): An instance of the MLM class.
        num_cpu (int): Number of CPUs to use for parallel processing.
    
    Returns:
        list: A list of sets containing potential substitutions for each word in the sentence.
    """
    # tokens = sentence.split()  # Tokenize the sentence into words
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
        
        # Ensure the original token is not in the MLM candidates list
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
        
        # Check if the word has any attackable synonyms
        attackable.append(len(combined_subs) > 0)

    return final_substitution_sets, attackable


class victim_models:
    def __init__(self,model_name):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name)
        self.detokenizer=TreebankWordDetokenizer()
        self.word_tokenize=word_tokenize
    
    def logits(self,text,label):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        logits=outputs[label]
        return logits

    def mr_words_saliency(self,v,text,label):
        tokens=self.word_tokenize(text)
        ori_logit=self.logits(text,label)
        saliency=[]
        for i in v:
            rem_text=copy.deepcopy(tokens)
            rem_text.pop(i)
            rem_logit=self.logits(self.detokenizer.detokenize(rem_text),label)
            s=ori_logit-rem_logit
            saliency.append(s)
        return saliency


    def words_saliency(self,text,label):
        tokens=self.word_tokenize(text)
        ori_logit=self.logits(text,label)
        saliency=[]
        for i in range(len(tokens)):
            rem_text=copy.deepcopy(tokens)
            rem_text.pop(i)
            rem_logit=self.logits(self.detokenizer.detokenize(rem_text),label)
            s=ori_logit-rem_logit
            saliency.append(s)
        return saliency

    def prob(self,text,label):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        prob=F.softmax(outputs,dim=0)[label]
        return prob

    def predict(self,text):
        inputs=self.tokenizer.encode(text, return_tensors="pt")
        outputs=self.model(inputs).logits.squeeze()
        prob=F.softmax(outputs,dim=0).squeeze()
        pre_label=torch.argmax(prob)
        return pre_label
