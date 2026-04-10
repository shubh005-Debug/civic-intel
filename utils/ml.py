"""ML model loader and prediction - no NLTK required"""
import pickle, re, os

STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once',
    'here','there','when','where','why','how','all','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now',
    'd','ll','m','o','re','ve','y','ain'
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_cat_model = _pri_model = _fake_model = None

def _load():
    global _cat_model, _pri_model, _fake_model
    if _cat_model is None:
        with open(f'{MODEL_DIR}/category_model.pkl','rb') as f: _cat_model  = pickle.load(f)
        with open(f'{MODEL_DIR}/priority_model.pkl','rb') as f: _pri_model  = pickle.load(f)
        with open(f'{MODEL_DIR}/fake_model.pkl',   'rb') as f: _fake_model = pickle.load(f)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return ' '.join(t for t in text.split() if t not in STOP_WORDS and len(t) > 2)

def predict(text):
    _load()
    clean = preprocess(text)
    cat      = _cat_model.predict([clean])[0]
    cat_prob = float(max(_cat_model.predict_proba([clean])[0]))
    pri      = _pri_model.predict([clean])[0]
    pri_prob = float(max(_pri_model.predict_proba([clean])[0]))
    fake     = int(_fake_model.predict([clean])[0])
    fake_prob= float(max(_fake_model.predict_proba([clean])[0]))
    return {
        'category': cat,
        'category_confidence': round(cat_prob * 100, 1),
        'priority': pri,
        'priority_confidence': round(pri_prob * 100, 1),
        'is_fake': fake,
        'fake_confidence': round(fake_prob * 100, 1),
    }
