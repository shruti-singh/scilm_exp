from collections import Counter
from nltk.corpus import wordnet
from random import shuffle
from textblob import TextBlob

from .dataset_stats import DataStats

import nltk
import random
import re


class CorruptionTechniques:
    
    @staticmethod
    def rotate_abstract(sent_text):
        # 1. Rotate abstract
        # break the abstract sentences list at the middle of total sentences -> 
        mid = len(sent_text)//2
        rotated_abstract = " ".join(sent_text[mid:]) + " " + " ".join(sent_text[0:mid])
        rotated_abstract = rotated_abstract.replace("  ", " ")
        return rotated_abstract
    
    @staticmethod
    def rand_shuffle_abstract(sent_text):
        # 2. Random Shuffle abstract
        shuffled_sentences = sent_text.copy()
        shuffle(shuffled_sentences)
        shuff_abs = " ".join(shuffled_sentences)
        return shuff_abs
    
    @staticmethod
    def rand_deletion(abstract, all_words_list):
        sample_len = int(len(all_words_list)*0.3)
        sampled_words = random.sample(all_words_list, sample_len)

        abstract_copy = abstract
        for w in sampled_words:
            abstract_copy = abstract_copy.replace(w, "", 1)
        return abstract_copy
    
    @staticmethod
    def pos_deleted_abstract(abs_word_text, postag):
        new_absract = []
        allpostags = nltk.pos_tag(abs_word_text, tagset='universal')
        for x in allpostags:
            if x[1] != postag:
                new_absract.append(x[0])
        return " ".join(new_absract)
    
    @staticmethod
    def pos_deleted_title(title_word_text, postag):
        pos_found = False
        new_title = []

        allpostags = nltk.pos_tag(title_word_text, tagset='universal')
        for x in allpostags:
            if x[1] != postag:
                new_title.append(x[0])
            else:
                pos_found = True
        if pos_found:
            return " ".join(new_title), True
        return title_word_text, False
    
    @staticmethod
    def del_partial_nn_phrases(abstract, org_data_path, top_50percent_nns=None):
        if not top_50percent_nns:
            top_50percent_nns = DataStats(org_data_path).noun_phrase_scores_abstract()
        blob = TextBlob(abstract)
        nnph = blob.noun_phrases
        abs_copy = abstract
        for n in nnph:
            if n in top_50percent_nns:
                abs_copy = abs_copy.replace(n, "")
        return abs_copy
    
    @staticmethod
    def del_all_nn_phrases(abstract):
        blob = TextBlob(abstract)
        nnph = blob.noun_phrases
        abs_copy = abstract
        for n in nnph:
            abs_copy = abs_copy.replace(n, "")
        return abs_copy
    
    @staticmethod
    def delete_posbased_quartile(sents, qnum):
        total_length = len(sents)
        qual_len = total_length//3
        local_quartile_idx = list(range((qnum-1)*qual_len, (qnum*qual_len)))

        new_abs = []
        for _, s in enumerate(sents):
            if _ in local_quartile_idx:
                continue
            else:
                new_abs.append(s)
        return " ".join(new_abs)
    
    @staticmethod
    def uppercase_nn(word_text, non_nn=False):
        """Uppercase all nouns in the text. If non_nn=True, uppercase all words except the NNs."""
        new_title = []
        nn_found = False

        allpostags = nltk.pos_tag(word_text, tagset='universal')
        for x in allpostags:
            if x[1] == 'NOUN':
                if not non_nn:
                    new_title.append(x[0].upper())
                    nn_found = True
                else:
                    new_title.append(x[0])
            else:
                if non_nn:
                    new_title.append(x[0].upper())
                    nn_found = True
                else:
                    new_title.append(x[0])
                    
        if nn_found:
            return " ".join(new_title), True
        return " ".join(word_text), False
    
    @staticmethod
    def char_deletion_noun(word_text):
        """Delete 1-2 characters from all nouns in the text - ideally paper title"""

        allpostags = nltk.pos_tag(word_text, tagset='universal')
        new_title = []
        nn_found = False
        
        for x in allpostags:
            if x[1] == 'NOUN':
                org_word = x[0]
                chars_to_del = 1
                if len(org_word) > 5:
                    chars_to_del = 2
                if len(org_word) > 3:
                    for idx in range(0, chars_to_del):
                        rem_idx = random.sample(range(0, len(org_word)), 1)[0]
                        org_word = org_word[:rem_idx] + org_word[rem_idx+1:]
                new_title.append(org_word)
                nn_found = True
            else:
                new_title.append(x[0])

        if nn_found:
            return " ".join(new_title), True
        return " ".join(word_text), False
    
    @staticmethod
    def repeat_title_nn(word_text):
        """Repeat a noun from the title in the title.
        Eg: Neural networks for NLP -> Neural networks networks for NLP NLP"""

        allpostags = nltk.pos_tag(word_text, tagset='universal')
        title_words = []
        new_title = []
        nn_words = []
        
        for x in allpostags:
            if x[1] == 'NOUN':
                nn_words.append(x[0])
            title_words.append(x[0])

        if nn_words:
            repeat_nn = random.sample(nn_words, 1)[0]
            new_nn = repeat_nn + ' ' + repeat_nn
            for w in title_words:
                if w == repeat_nn:
                    new_title.append(new_nn)
                else:
                    new_title.append(w)
            return " ".join(new_title), True
        return " ".join(word_text), False
    
    @staticmethod
    def repeat_abs_in_title(abs_word_text, title):
        """Add nn words from the abstract to the title"""

        allpostags = nltk.pos_tag(abs_word_text, tagset='universal')
        nn_words = []
        
        for x in allpostags:
            if x[1] == 'NOUN':
                nn_words.append(x[0])

        if nn_words:
            nn_counter = Counter(nn_words)
            most_common_nns = nn_counter.most_common(n=len(nn_counter)//2)
            nn_phrase = " ".join([x[0] for x in most_common_nns])
            return title+' '+nn_phrase, True
        return title, False
    
    @staticmethod
    def retain_only_abstract_nns(abs_word_text):
        """Delete all words from the abstract except the nouns. Can be used with as mod_abs+title or just mod_abs."""

        allpostags = nltk.pos_tag(abs_word_text, tagset='universal')
        nn_words = []
        
        for x in allpostags:
            if x[1] == 'NOUN':
                nn_words.append(x[0])
        if nn_words:
            return " ".join(nn_words), True
        return " ".join(abs_word_text), False
    
    # @staticmethod
    # def retain_only_non_nns(abstract):
    #     """Delete all nouns in the text. Can be used with as mod_abs+title or just mod_abs."""
    #     words = nltk.word_tokenize(abstract)
    #     allpostags = nltk.pos_tag(words, tagset='universal')
    #     non_nn_words  = []
    #
    #     for x in allpostags:
    #         if x[1] != 'NOUN':
    #             non_nn_words.append(x[0])
    #     if non_nn_words:
    #         return " ".join(non_nn_words), True
    #     return abstract, False
    
    @staticmethod
    def replace_adj_with_antonyms(abs_word_text):
        """Replace all adjectives with antonyms"""

        allpostags = nltk.pos_tag(abs_word_text, tagset='universal')
        perturbed_text = []
        adj_replacement = False
        
        for x in allpostags:
            if x[1] == 'ADJ':
                adj_word = x[0]
                antonyms_list = []
                for syn in wordnet.synsets(adj_word):
                    if syn:
                        for lemm in syn.lemmas():
                            if lemm.antonyms():
                                antonyms_list.append(lemm.antonyms()[0].name())
                if antonyms_list:
                    adj_antonym = Counter(antonyms_list).most_common(n=1)[0][0]
                    perturbed_text.append(adj_antonym)
                    adj_replacement = True
                else:
                    perturbed_text.append(adj_word)
            else:
                perturbed_text.append(x[0])
        
        if adj_replacement:
            return " ".join(perturbed_text), True
        return " ".join(abs_word_text), False
    
    @staticmethod
    def perturb_whitespace(text):
        """Replace 50% whitespace chars randomly with 2-5 whitespace chars"""
        whitespace_indices = [i.start() for i in re.finditer(' ', text)]
        to_perturb_indices = random.sample(whitespace_indices, len(whitespace_indices)//2)
        
        if to_perturb_indices:
            perturbed_text = ''
            prev_perturbed_index = 0
            for idx in to_perturb_indices:
                num_ws_chars = random.randint(2, 5)
                perturbed_text += text[prev_perturbed_index:idx] + ' '*num_ws_chars
                prev_perturbed_index = idx + 1
            perturbed_text += text[prev_perturbed_index:]
            return perturbed_text, True
        else:
            return text, False
