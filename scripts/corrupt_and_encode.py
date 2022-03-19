from collections import defaultdict
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from  cogdl import oagbert
import torch

from .corruption_techniques import CorruptionTechniques
from .dataset_stats import DataStats

import argparse
import nltk
import pickle

        
class CorruptAndEncode:
    
    def __init__(self, data, emb_dir, model='specter'):
        self.data = data
        self.direc = emb_dir + '/' + model
        self.abs_useful_pos = ['ADJ', 'NOUN', 'VERB', 'DET', 'PRON', 'ADV', 'NUM']
        self.title_useful_pos = ['ADJ', 'NOUN', 'VERB', 'DET']

        # Init the tokenizer and encoding the model
        if model == 'specter':
            self.load_specter()
        elif model == 'scibert':
            self.load_scibert()
        elif model == 'oagbert':
            self.load_oagbert()
        return

    def update_encoding_model(self, model_name):
        if model_name == 'specter':
            self.load_specter()
        elif model_name == 'scibert':
            self.load_scibert()
        elif model_name == 'oagbert':
            self.load_oagbert()
        else:
            print('Invalid model name! Please try again with a valid  model name.')
    
    def load_specter(self):
        self.model_name = 'specter'
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter')
        return

    def load_scibert(self):
        self.model_name = 'scibert'
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
        return

    def load_oagbert(self):
        self.model_name = 'oagbert'
        self.tokenizer, self.model = oagbert("oagbert-v2")
        return

    def encode_batch_wise(self, document_list, to_save_loc, dict_name, org_pid_seq=None, is_doc_list=True, batch_size=20):
        if self.model_name == 'specter':
            return self.encode_batch_wise_using_specter(document_list, to_save_loc, dict_name, org_pid_seq, is_doc_list, batch_size)
        elif self.model_name == 'scibert':
            return self.encode_batch_wise_using_scibert(document_list, to_save_loc, dict_name, org_pid_seq, is_doc_list)
        elif self.model_name == 'oagbert':
            return self.encode_batch_wise_using_oagbert(document_list, to_save_loc, dict_name, org_pid_seq, is_doc_list)

    def encode_batch_wise_using_oagbert(self, document_list, to_save_loc, dict_name, org_pid_seq=None,
                                        is_doc_list=True):
        document_emb_dict = {}

        for _, d in enumerate(document_list):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            if is_doc_list:
                paper_title = (d.get('title') or '')
                paper_abstract = (d.get('abstract') or '')
                document_id = org_pid_seq[_]
            else:
                # document_list is actually a dictionatry. org_pid_seq is not provided
                paper_title = (document_list[d].get('title') or '')
                paper_abstract = (document_list[d].get('abstract') or '')
                document_id = d

            #check if title+abs is empty
            title_abs = paper_title + paper_abstract
            if not title_abs.strip():
                continue

            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, \
            masked_positions, num_spans = self.model.build_inputs(
                title=paper_title, abstract=paper_abstract, venue=[], authors=[], concepts=[], affiliations=[])
            sequence_output, pooled_output = self.model.bert.forward(
                input_ids=torch.LongTensor(input_ids).unsqueeze(0),
                token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
                attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=torch.LongTensor(position_ids).unsqueeze(0),
                position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0)
            )

            embedding = pooled_output.detach().numpy()
            document_emb_dict[document_id] = embedding

        document_emb_dict.__setitem__('name', dict_name)

        with open(to_save_loc, 'wb') as f:
            pickle.dump(document_emb_dict, f)

        return document_emb_dict

    def encode_batch_wise_using_scibert(self, document_list, to_save_loc, dict_name, org_pid_seq=None,
                                        is_doc_list=True):
        document_emb_dict = {}

        for _, d in enumerate(document_list):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            if is_doc_list:
                text_to_encode = (d.get('title') or '') + self.tokenizer.sep_token + (d.get('abstract') or '')
                document_id = org_pid_seq[_]
            else:
                # document_list is actually a dictionatry. org_pid_seq is not provided
                text_to_encode = (document_list[d].get('title') or '') + self.tokenizer.sep_token + (document_list[d].get('abstract') or '')
                document_id = d

            inputs = self.tokenizer(text_to_encode, padding=True, truncation=True, return_tensors="pt", max_length=512)
            result = self.model(**inputs)
            embedding = result.last_hidden_state[:, 0, :].detach().numpy()
            document_emb_dict[document_id] = embedding

        document_emb_dict.__setitem__('name', dict_name)

        with open(to_save_loc, 'wb') as f:
            pickle.dump(document_emb_dict, f)

        return document_emb_dict

    def encode_batch_wise_using_specter(self, document_list, to_save_loc, dict_name, org_pid_seq=None,
                                        is_doc_list=True, batch_size=20):
        # List to contain small batch of douments and the corresponding doc ids
        doc_batch_list = []
        batch_ids = []

        document_emb_dict = {}

        for _, d in enumerate(document_list):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            if is_doc_list:
                doc_batch_list.append((d.get('title') or '') + self.tokenizer.sep_token + (d.get('abstract') or ''))
                batch_ids.append(org_pid_seq[_])
            else:
                # document_list is actually a dictionatry. org_pid_seq is not provided
                doc_batch_list.append((document_list[d].get('title') or '') + self.tokenizer.sep_token + (document_list[d].get('abstract') or ''))
                batch_ids.append(d)

            if _%batch_size == 0:
                inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
                result = self.model(**inputs)
                embeddings = result.last_hidden_state[:, 0, :]

                for ii, k in enumerate(batch_ids):
                    document_emb_dict[k] = embeddings[ii].detach().numpy()

                doc_batch_list = []
                batch_ids = []

        if batch_ids:
            inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
            result = self.model(**inputs)
            embeddings = result.last_hidden_state[:, 0, :]

            for _, k in enumerate(batch_ids):
                document_emb_dict[k] = embeddings[_].detach().numpy()

        document_emb_dict.__setitem__('name', dict_name)

        # To freeup memory in case of reuse in jupyter
        doc_batch_list = []
        batch_ids = []

        with open(to_save_loc, 'wb') as f:
            pickle.dump(document_emb_dict, f)

        return document_emb_dict

    def naive_corrupt_and_encode(self):
        # Simply remove title/abstract
        paper_tabs = {}
        paper_titles = {}
        paper_abstracts_only = {}
                
        for _, p in enumerate(self.data):
            paper_titles[p] = {'title': self.data[p]['title']}
        self.encode_batch_wise(paper_titles, '{}/emb_titles.pkl'.format(self.direc), dict_name='T', is_doc_list=False)
        paper_titles = None
        
        for _, p in enumerate(self.data):
            paper_tabs[p] = {'title': self.data[p]['title'], 'abstract': self.data[p]['abstract']}
        self.encode_batch_wise(paper_tabs, '{}/emb_tabs.pkl'.format(self.direc), dict_name='T_A', is_doc_list=False)
        paper_tabs = None

        for _, p in enumerate(self.data):
            paper_abstracts_only[p] = {'abstract': self.data[p]['abstract']}
        self.encode_batch_wise(paper_abstracts_only, '{}/emb_abstracts.pkl'.format(self.direc), dict_name='A', is_doc_list=False)
        paper_abstracts_only = None
        return
    
    # Abstract related corruption
    def corrupt_abstract(self):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        paper_rotate = {}
        paper_randshuffle = {}
        paper_sorted_asc = {}
        paper_sorted_desc = {}
        paper_rand_delete = {}

        paper_pos_del = defaultdict(dict)
        
        paper_nn_ph_del = {}
        paper_nn_ph_del_partial = {}

        paper_q1_del = {}
        paper_q2_del = {}
        paper_q3_del = {}

        paper_abs_nn_upper = {}
        paper_abs_nonnn_upper = {}

        paper_char_del_title_abs = {}

        paper_title_repeat_nn = {}
        paper_abs_repeat_title = {}

        paper_abs_nn_only = {}

        paper_abs_adj_anto = {}

        paper_tabs_ws = {}

        datastats_papers = DataStats(self.data)
        abslen_dict = datastats_papers.paper_abs_length()
        indpaper_pos = datastats_papers.abstract_postag_hist()
        top_50percent_nns = datastats_papers.noun_phrase_scores_abstract()

        for _, p in enumerate(self.data):

            if _ % 5000 == 0:
                print("Data perturbed: ", _)

            title = self.data[p]['title']
            abstract = self.data[p]['abstract']
            
            abs_sent_text = nltk.sent_tokenize(abstract)
            abs_word_text = nltk.word_tokenize(abstract)
            
            # 1. Rotate abstract
            rotated_abstract = CorruptionTechniques.rotate_abstract(abs_sent_text)
            paper_rotate[p] = {'title': title, 'abstract': rotated_abstract}

            # 2. Random Shuffle abstract
            shuff_abs = CorruptionTechniques.rand_shuffle_abstract(abs_sent_text)
            paper_randshuffle[p] = {'title': title, 'abstract': shuff_abs}

            # 3 & 4. Arrange in ascending/descending order of sentence length (i.e. word count)
            asc_word_count = sorted(abs_sent_text, key=lambda x: len(nltk.word_tokenize(x)))
            desc_word_count = sorted(abs_sent_text, key=lambda x: len(nltk.word_tokenize(x)), reverse=True)

            paper_sorted_asc[p] = {'title': title, 'abstract': " ".join(asc_word_count)}
            paper_sorted_desc[p] = {'title': title, 'abstract': " ".join(desc_word_count)}
            
            # 5. Rand deletion 30% words
            all_word_tags = []
            all_words = []
            for s in abs_sent_text:
                words = nltk.word_tokenize(s)
                postags = nltk.pos_tag(words, tagset='universal')
                all_word_tags += [x[1] for x in postags]
                all_words += words
            paper_rand_delete[p] = {'title': title, 'abstract': CorruptionTechniques.rand_deletion(abstract, all_words)}

            # 6. POS based deletion: ['ADJ', 'NOUN', 'VERB', 'DET', 'PRON', 'ADV', 'NUM']
            for postag in self.abs_useful_pos:
                if postag in indpaper_pos[p]:
                    paper_pos_del[postag][p] = {'title': title, 'abstract': CorruptionTechniques.pos_deleted_abstract(abs_word_text, postag)}

            # 7 and 8. Noun phrase deletion - complete and partial
            nn_deleted = CorruptionTechniques.del_all_nn_phrases(abstract)
            if nn_deleted != abstract:
                paper_nn_ph_del[p] = {'title': title, 'abstract': nn_deleted}

            nn_partial_deleted = CorruptionTechniques.del_partial_nn_phrases(abstract, self.data, top_50percent_nns)
            if nn_partial_deleted != abstract:
                paper_nn_ph_del_partial[p] = {'title': title, 'abstract': nn_partial_deleted}

            # 9. Sentence positon based deletion
            if abslen_dict[p] >= 3:
                paper_q1_del[p] = {'title': title, 'abstract': CorruptionTechniques.delete_posbased_quartile(abs_sent_text, 1)}
                paper_q2_del[p] = {'title': title, 'abstract': CorruptionTechniques.delete_posbased_quartile(abs_sent_text, 2)}
                paper_q3_del[p] = {'title': title, 'abstract': CorruptionTechniques.delete_posbased_quartile(abs_sent_text, 3)}

            # ================================================================================
            # Reuse variables to save space :/
            abs_nn_upper = CorruptionTechniques.uppercase_nn(abs_word_text)
            if abs_nn_upper[1]:
                paper_abs_nn_upper[p] = {'title': title, 'abstract': abs_nn_upper[0]}

            abs_nn_upper = CorruptionTechniques.uppercase_nn(abs_word_text, non_nn=True)
            if abs_nn_upper[1]:
                paper_abs_nonnn_upper[p] = {'title': title, 'abstract': abs_nn_upper[0]}

            paper_char_del_title_abs[p] = {'title': CorruptionTechniques.char_deletion_noun(nltk.word_tokenize(title))[0],
                                           'abstract': CorruptionTechniques.char_deletion_noun(abs_word_text)[0]}

            title_nn_repeat = CorruptionTechniques.repeat_title_nn(nltk.word_tokenize(title))
            if title_nn_repeat[1]:
                paper_title_repeat_nn[p] = {'title': title_nn_repeat[0], 'abstract': abstract}

            title_nn_repeat = CorruptionTechniques.repeat_abs_in_title(abs_word_text, title)
            if title_nn_repeat[1]:
                paper_abs_repeat_title[p] = {'title': title_nn_repeat[0], 'abstract': abstract}

            abs_nns = CorruptionTechniques.retain_only_abstract_nns(abs_word_text)
            if abs_nns[1]:
                paper_abs_nn_only[p] = {'title': title, 'abstract': abs_nns[0]}

            abs_antonym = CorruptionTechniques.replace_adj_with_antonyms(abs_word_text)
            if abs_antonym[1]:
                paper_abs_adj_anto[p] = {'title': title, 'abstract': abs_antonym[0]}

            paper_tabs_ws[p] = {'title': CorruptionTechniques.perturb_whitespace(title)[0],
                                'abstract': CorruptionTechniques.perturb_whitespace(abstract)[0]}

            
        print('Started encoding ...')
        # Encode using specter
        self.encode_batch_wise(paper_randshuffle, '{}/emb_rand_shuff.pkl'.format(self.direc), dict_name='T_AShuff', is_doc_list=False)
        paper_randshuffle = None
        self.encode_batch_wise(paper_rotate, '{}/emb_rotate.pkl'.format(self.direc), dict_name='T_ARot', is_doc_list=False)
        paper_rotate = None
        self.encode_batch_wise(paper_sorted_asc, '{}/emb_asc.pkl'.format(self.direc), dict_name='T_ASortAsc', is_doc_list=False)
        paper_sorted_asc = None
        self.encode_batch_wise(paper_sorted_desc, '{}/emb_desc.pkl'.format(self.direc), dict_name='T_ASortDesc', is_doc_list=False)
        paper_sorted_desc = None
        self.encode_batch_wise(paper_rand_delete, '{}/emb_rand_del.pkl'.format(self.direc), dict_name='T_ADelRand', is_doc_list=False)
        paper_rand_delete = None
        self.encode_batch_wise(paper_nn_ph_del, '{}/emb_nnph_del.pkl'.format(self.direc), dict_name='T_ADelNNPH', is_doc_list=False)
        paper_nn_ph_del = None
        self.encode_batch_wise(paper_nn_ph_del_partial, '{}/emb_nnph_pardel.pkl'.format(self.direc), dict_name='T_ADelTopNNPH', is_doc_list=False)
        paper_nn_ph_del_partial = None
        self.encode_batch_wise(paper_q1_del, '{}/emb_q1_del.pkl'.format(self.direc), dict_name='T_ADelQ1', is_doc_list=False)
        paper_q1_del = None
        self.encode_batch_wise(paper_q2_del, '{}/emb_q2_del.pkl'.format(self.direc), dict_name='T_ADelQ2', is_doc_list=False)
        paper_q2_del = None
        self.encode_batch_wise(paper_q3_del, '{}/emb_q3_del.pkl'.format(self.direc), dict_name='T_ADelQ3', is_doc_list=False)
        paper_q3_del = None
        for univ_pos in self.abs_useful_pos:
            self.encode_batch_wise(paper_pos_del[univ_pos], '{}/emb_pos_del_{}.pkl'.format(self.direc, univ_pos), dict_name='T_ADel'+univ_pos, is_doc_list=False)

        self.encode_batch_wise(paper_abs_nn_upper, '{}/emb_a_nn_upper.pkl'.format(self.direc), dict_name='T_ANNU', is_doc_list=False)
        paper_abs_nn_upper = None
        self.encode_batch_wise(paper_abs_nonnn_upper, '{}/emb_a_nonnn_upper.pkl'.format(self.direc), dict_name='T_ANonNNU', is_doc_list=False)
        paper_abs_nonnn_upper = None
        self.encode_batch_wise(paper_char_del_title_abs, '{}/emb_ta_chardel.pkl'.format(self.direc), dict_name='T_A_DelNNChar', is_doc_list=False)
        paper_char_del_title_abs = None
        self.encode_batch_wise(paper_title_repeat_nn, '{}/emb_t_nn_rep.pkl'.format(self.direc), dict_name='TRepNNT_A', is_doc_list=False)
        paper_title_repeat_nn = None
        self.encode_batch_wise(paper_abs_repeat_title, '{}/emb_a_nn_rep.pkl'.format(self.direc), dict_name='TRepNNA_A', is_doc_list=False)
        paper_abs_repeat_title = None
        self.encode_batch_wise(paper_abs_nn_only, '{}/emb_abs_nn_preserve.pkl'.format(self.direc), dict_name='T_ADelNonNN', is_doc_list=False)
        paper_abs_nn_only = None
        self.encode_batch_wise(paper_abs_adj_anto, '{}/emb_abs_adj_ant.pkl'.format(self.direc), dict_name='T_ARepADJ', is_doc_list=False)
        paper_abs_adj_anto = None
        self.encode_batch_wise(paper_tabs_ws, '{}/emb_tabs_ws.pkl'.format(self.direc), dict_name='T_A_WS', is_doc_list=False)
        paper_tabs_ws = None
        return
    
    def corrupt_title(self):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        paper_pos_del_title = defaultdict(dict)
        only_nn_title = defaultdict(dict)
        paper_title_nn_upper = defaultdict(dict)
        paper_title_nonnn_upper = defaultdict(dict)

        for _, p in enumerate(self.data):

            if _ % 5000 == 0:
                print("Data perturbed: ", _)

            title = self.data[p]['title']
            abstract = self.data[p]['abstract']

            title_word_text = nltk.word_tokenize(title)
            
            # 9. POS deletion from title
            for postag in self.title_useful_pos:
                pos_del_title, pos_found = CorruptionTechniques.pos_deleted_title(title_word_text, postag)
                if pos_found:
                    paper_pos_del_title[postag][p] = {'title': pos_del_title, 'abstract': abstract} 
                    if postag == 'NOUN':
                        only_nn_title[p] = {'title': pos_del_title}

            title_nn_upper = CorruptionTechniques.uppercase_nn(title_word_text)
            if title_nn_upper[1]:
                paper_title_nn_upper[p] = {'title': title_nn_upper[0], 'abstract': abstract}

            title_nn_upper = CorruptionTechniques.uppercase_nn(title_word_text, non_nn=True)
            if title_nn_upper[1]:
                paper_title_nonnn_upper[p] = {'title': title_nn_upper[0], 'abstract': abstract}
        
        print('Started encoding ...')
        for univ_pos in self.title_useful_pos:
            self.encode_batch_wise(paper_pos_del_title[univ_pos], '{}/emb_pos_del_title_{}.pkl'.format(self.direc, univ_pos), dict_name='TDel'+univ_pos+"_A", is_doc_list=False)
        
        self.encode_batch_wise(only_nn_title, '{}/emb_nn_del_titleonly.pkl'.format(self.direc), dict_name='TDelNN', is_doc_list=False)
        only_nn_title = None
        self.encode_batch_wise(paper_title_nn_upper, '{}/emb_t_nn_upper.pkl'.format(self.direc), dict_name='TNNU_A', is_doc_list=False)
        paper_title_nn_upper = None
        self.encode_batch_wise(paper_title_nonnn_upper, '{}/emb_t_nonnn_upper.pkl'.format(self.direc), dict_name='TNonNNU_A', is_doc_list=False)
        paper_title_nonnn_upper = None
        return
