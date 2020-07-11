import pickle
from typing import List
import re as re
def predict(documents: List[str]):
#def predict(documents: str):
    document_classes = {
        'UNK': documents
    }
    print (type (documents))
    print (documents)
    #print(re.split('\W+',documents))
    #word_classes ={'UNK': re.split('\W+',documents)}#re.split('\W+',documents)
    txt= documents[0]
    print(txt)
    #word_classes ={'UNK': [['bad', 'place']]}
    word_classes ={'UNK': [re.split('\W+',txt)]}
    print (word_classes)
    with open('models/model.pkl', 'rb') as input_file:
        model = pickle.load(input_file)

    document_words = word_classes['UNK']

    predictions = []
    for document in document_words:
        positive_prob = model['POS_PROB']
        negative_prob = model['NEG_PROB']
        for word in document:
            if word in model['COND_POS_PROBS']:
                positive_prob += model['COND_POS_PROBS'][word]['logprob']
            else:
                positive_prob += model['COND_POS_PROBS'][-1]['logprob']

            if word in model['COND_NEG_PROBS']:
                negative_prob += model['COND_NEG_PROBS'][word]['logprob']
            else:
                negative_prob += model['COND_NEG_PROBS'][-1]['logprob']

        if positive_prob >= negative_prob:
            predictions.append('POS')
        else:
            predictions.append('NEG')

    return predictions
