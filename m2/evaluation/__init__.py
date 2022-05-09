from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .spice.spice import Spice
from .tokenizer import PTBTokenizer


def compute_scores(gts, gen):
    metrics = (Bleu(), Meteor(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    
    spice = Spice()
    score, scores = spice.compute_score(gts, gen)
    all_score["SPICE"] = score
    all_scores["SPICE"] = scores

    return all_score, all_scores
