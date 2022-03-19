from collections import Counter
from textblob import TextBlob

import argparse
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pickle

class DataStats:
    def __init__(self, data, output_dir=None):
        if type(data) == 'str':
            self.data = pickle.load(open(args.data_path, 'rb'))
        else:
            self.data = data

        # Todo: Save figures here
        if output_dir:
            self.output_dir = output_dir

        self.title_useful_pos = ['ADJ', 'NOUN', 'VERB', 'DET']
        self.abstract_useful_pos = ['ADJ', 'NOUN', 'VERB', 'DET', 'PRON', 'ADV', 'NUM']
        return

    def paper_abs_length(self):
        print("Abstract length stats:")
        abs_lengths = []
        abslen_dict = {}

        for k in self.data:
            sents = nltk.sent_tokenize(self.data[k]['abstract'])
            abs_lengths.append(len(sents))
            abslen_dict[k] = len(sents)

        print("Mean: {}\nMedia: {}\nMax len: {}\nMin len: {}".format(np.mean(abs_lengths), np.median(abs_lengths), np.max(abs_lengths), np.min(abs_lengths)))

        self.abs_length_boxplot(abs_lengths)
        return abslen_dict

    def abs_length_boxplot(self, abs_lengths):
        fig1, ax1 = plt.subplots(dpi=100)
        ax1.set_title('Abstract length - # of sentences')
        ax1.boxplot(abs_lengths, sym='+')
        plt.show()
        return

    def paper_title_length(self):
        print("Title length stats:")
        title_lengths = []
        titlelen_dict = {}

        for k in self.data:
            words = nltk.word_tokenize(self.data[k]['title'])
            title_lengths.append(len(words))
            titlelen_dict[k] = len(words)

        print("Mean: {}\nMedia: {}\nMax len: {}\nMin len: {}".format(np.mean(title_lengths), np.median(title_lengths), np.max(title_lengths), np.min(title_lengths)))

        self.title_length_boxplot(title_lengths)
        return titlelen_dict

    def title_length_boxplot(self, title_lengths):
        fig1, ax1 = plt.subplots(dpi=100)
        ax1.set_title('Title length - # of wordss')
        ax1.boxplot(title_lengths, sym='+')
        plt.show()
        return

    def title_postag_hist(self):
        pos_per_paper_dict = {x: [] for x in self.abstract_useful_pos} # Format: {'NOUN': [2,3,5], "ADJ":[12,1,0]}

        for k, v in self.data.items():
            title = v['title']
            sents = nltk.sent_tokenize(title)
            all_words_postags_list = []
            for s in sents:
                words = nltk.word_tokenize(s)
                postags = nltk.pos_tag(words, tagset='universal')
                all_words_postags_list += [x[1] for x in postags]
            overall_title_pos_cnt = Counter(all_words_postags_list)
            for pos in pos_per_paper_dict:
                pos_per_paper_dict[pos].append(overall_title_pos_cnt[pos])

        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=False, tight_layout=False, dpi=200)

        for _, post in enumerate(self.abstract_useful_pos):
            num_bins = 10
            if (max(pos_per_paper_dict[post])+1-min(pos_per_paper_dict[post])) < 10:
                num_bins = range(min(pos_per_paper_dict[post]), max(pos_per_paper_dict[post])+1, 1)
            freq, bins, pats = axs[_//3, _%3].hist(pos_per_paper_dict[post], bins=num_bins, histtype='step', stacked=True, fill=False, color="red")
            for i in range(len(bins)-1):
                axs[_//3, _%3].text(bins[i],freq[i]+10,str(round((freq[i]*100)/len(self.data), 1))+"%", fontdict={"color": "red"})
            plt.setp(axs[_//3, _%3], xlabel=post)

        fig.suptitle("Histogram of count of pos tags in {} paper titles.\nFor bin=[1, 2), a bin height 'x' represents \
         x paper titles have postag 'p' occurence = 1.".format(len(self.data)))

        return

    def abstract_postag_hist(self):
        pos_per_paper_dict = {x: [] for x in self.abstract_useful_pos} # Format: {'NOUN': [2,3,5], "ADJ":[12,1,0]}
        indpaper_pos = {}

        for k, v in self.data.items():
            abstract = v['abstract']
            sents = nltk.sent_tokenize(abstract)
            all_words_postags_list = []
            for s in sents:
                words = nltk.word_tokenize(s)
                postags = nltk.pos_tag(words, tagset='universal')
                all_words_postags_list += [x[1] for x in postags]
            overall_title_pos_cnt = Counter(all_words_postags_list)
            indpaper_pos[k] = list(set(all_words_postags_list))
            for pos in pos_per_paper_dict:
                pos_per_paper_dict[pos].append(overall_title_pos_cnt[pos])

        fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=False, tight_layout=False, dpi=250)

        for _, post in enumerate(self.abstract_useful_pos):
            num_bins = 10
            if (max(pos_per_paper_dict[post])+1-min(pos_per_paper_dict[post])) < 10:
                num_bins = range(min(pos_per_paper_dict[post]), max(pos_per_paper_dict[post])+1, 1)
            freq, bins, pats = axs[_//3, _%3].hist(pos_per_paper_dict[post], bins=num_bins, histtype='step', stacked=True, fill=False, color="red")
            for i in range(len(bins)-1):
                axs[_//3, _%3].text(bins[i],freq[i]+10,str(round((freq[i]*100)/len(self.data), 1))+"%", fontdict={"color": "red"})
            plt.setp(axs[_//3, _%3], xlabel=post)

        fig.suptitle("Histogram of count of pos tags in {} paper abstracts.\nFor bin=[1, 2), a bin height 'x' represents 'x' abstracts have postag 'p' occurence = 1.".format(len(self.data)))

        return indpaper_pos

    def noun_phrase_scores_abstract(self):
        noun_ph_tf = Counter()
        noun_ph_idf = Counter()

        for k in self.data:
            blob = TextBlob(self.data[k]['abstract'])
            noun_ph_tf.update(blob.noun_phrases)
            noun_ph_idf.update(list(set(blob.noun_phrases)))

        lnidf = {} # ln2 idf scores
        for k in noun_ph_idf:
            lnidf[k] = np.log2(1+( (len(self.data)-noun_ph_idf[k]+0.5)/(noun_ph_idf[k]+0.5) ))

        bm25_scores = {} # tf is not normalized

        for k in noun_ph_tf:
            bm25_scores[k] = noun_ph_tf[k] * lnidf[k]

        sorted_nn_phs = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        print("Top-10 noun phrases: ", sorted_nn_phs[0:10])
        print("\nBottom-10 noun phrases: ", sorted_nn_phs[-10:])
        print("\n\nRange of score: {}-{}".format(sorted_nn_phs[-1][1], sorted_nn_phs[0][1]))
        print("Total noun phrases: ", len(bm25_scores))

        top_50percent_nns = []
        top50percent = len(sorted_nn_phs)//2

        for i in range(0, top50percent):
            top_50percent_nns.append(sorted_nn_phs[i][0])

        return top_50percent_nns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='path to the dataset title+abs pickle file')
    # TODO: Save images to dir
    parser.add_argument('--output_dir', help='path to the output dir to save stat images')
    args = parser.parse_args()

    if args.data_path.endswith('org_data.pkl'):
        data = pickle.load(open(args.data_path, 'rb'))
        return DataStats(data)
    else:
        print("The datapath should contain file of format datasets/{}/org_data.pkl. Please check path again. Exiting.")
        return

if __name__ == "__main__":
    main()

# Sample run: python code/dataset_stats.py --data_path datasets/ACL/org_data.py