from convokit import Corpus, download

if __name__ == '__main__':
    corpus = Corpus(filename=download("switchboard-corpus"))
    corpus.print_summary_stats()

    processed_corpus = Corpus(filename=download("switchboard-processed-corpus"))
    processed_corpus.print_summary_stats()

    # response acknowledgements
    bk_utterances = []
    count = 0
    for utt in processed_corpus.iter_utterances():
        meta = utt.meta
        if 'bk' in meta.get('tags'):
            if len(utt.text) < 20:
                bk_utterances.append(utt.text)
                count += 1
                if count == 50:
                    break

    for uttr in bk_utterances:
        print(uttr)
