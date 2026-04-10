"""
Train all ML models - uses only sklearn + pandas (no NLTK required)
Run once: python train_model.py
"""
import pandas as pd, numpy as np, pickle, os, re, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Basic English stop words (no NLTK needed)
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
    'd','ll','m','o','re','ve','y','ain','couldn','didn','doesn','hadn',
    'hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn',
    'wasn','weren','won','wouldn'
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    return ' '.join(t for t in tokens if t not in STOP_WORDS and len(t) > 2)

# ── DATASET ───────────────────────────────────────────────────────────────
data = [
  # (text, category, priority)
  # ROAD
  ("Massive pothole on MG Road near bus stop causing daily accidents and injuries", "Road", "Critical"),
  ("Road near school completely broken vehicles getting damaged daily", "Road", "Critical"),
  ("Flyover bridge has developed deep cracks extremely dangerous for commuters", "Road", "Critical"),
  ("Main road collapsed near market emergency situation vehicles stuck", "Road", "Critical"),
  ("Large crater on road near hospital patients unable to reach on time", "Road", "Critical"),
  ("Bridge cracks getting wider emergency situation on city highway", "Road", "Critical"),
  ("Road construction incomplete creating huge traffic jams for weeks", "Road", "Moderate"),
  ("Potholes filled with rainwater causing accidents during monsoon season", "Road", "Moderate"),
  ("Highway divider broken vehicles crossing into wrong lane dangerous", "Road", "Moderate"),
  ("No road markings zebra crossing completely faded causing accidents", "Road", "Moderate"),
  ("Road repair work stopped midway worse condition than before", "Road", "Moderate"),
  ("Road waterlogging for weeks creating breeding ground for mosquitos", "Road", "Moderate"),
  ("Speed breakers broken not visible at night minor inconvenience", "Road", "Normal"),
  ("Street slightly uneven near the park needs minor patching work", "Road", "Normal"),
  ("Road divider paint faded needs repainting for better visibility", "Road", "Normal"),
  ("Footpath tiles broken near the residential colony needs repair", "Road", "Normal"),
  ("Pothole on highway causing truck accidents deaths reported", "Road", "Critical"),
  ("Road flooded after rain cars submerged accident danger", "Road", "Critical"),
  # WATER
  ("Water supply cut for 5 days families suffering no drinking water available", "Water", "Critical"),
  ("Sewage water mixing with drinking water pipeline health emergency in colony", "Water", "Critical"),
  ("Main water pipe burst flooding entire street dangerous situation", "Water", "Critical"),
  ("Contaminated water causing severe illness multiple families hospitalized", "Water", "Critical"),
  ("No water for five days hospital operations affected patients suffering", "Water", "Critical"),
  ("Dirty brown water coming from tap not safe to drink children sick", "Water", "Critical"),
  ("Water pipeline leaking on street huge water wastage for two weeks", "Water", "Moderate"),
  ("No water pressure cannot fill overhead tank for daily needs", "Water", "Moderate"),
  ("Water tanker not coming despite repeated requests three days without water", "Water", "Moderate"),
  ("Borewell motor broken entire colony has no water supply", "Water", "Moderate"),
  ("Overhead tank dirty not cleaned since months water quality poor", "Water", "Moderate"),
  ("Water supply timing changed without notice residents missing their turn", "Water", "Moderate"),
  ("Water meter faulty billing is incorrect needs calibration check", "Water", "Normal"),
  ("Minor water seepage from pipeline on footpath small puddle forming", "Water", "Normal"),
  ("Water pressure slightly low during morning hours minor issue", "Water", "Normal"),
  ("Tap water smells bad could be contamination need testing", "Water", "Critical"),
  ("Water pipeline broken main road flooding area", "Water", "Critical"),
  # ELECTRICITY
  ("Power outage 12 hours no response from electricity department critical area", "Electricity", "Critical"),
  ("Transformer blast near residential area causing fire risk emergency", "Electricity", "Critical"),
  ("Exposed live wires near children playground extremely dangerous", "Electricity", "Critical"),
  ("Frequent power cuts affecting hospital equipment lives at risk", "Electricity", "Critical"),
  ("Underground cable fault causing area wide blackout for days", "Electricity", "Critical"),
  ("Electric pole tilted about to fall on pedestrians emergency situation", "Electricity", "Critical"),
  ("High voltage fluctuation damaging expensive home appliances daily", "Electricity", "Moderate"),
  ("Short circuit in main power line causing repeated area outages", "Electricity", "Moderate"),
  ("Street lights not working area completely dark at night unsafe", "Electricity", "Moderate"),
  ("Electric wires dangling low on street dangerous for tall vehicles", "Electricity", "Moderate"),
  ("Generator not working in government building during load shedding", "Electricity", "Moderate"),
  ("Power supply unstable causing issues for work from home employees", "Electricity", "Moderate"),
  ("Electricity bill too high despite low usage needs audit", "Electricity", "Normal"),
  ("Street light flickering at night minor annoyance needs bulb change", "Electricity", "Normal"),
  ("Minor fluctuation during peak evening hours not causing damage", "Electricity", "Normal"),
  ("Electric shock from switchboard dangerous fire hazard building", "Electricity", "Critical"),
  ("Power line fell on road people trapped electrocution risk", "Electricity", "Critical"),
  # SANITATION
  ("Garbage not collected for two weeks entire street stinking disease risk", "Sanitation", "Critical"),
  ("Sewage manhole uncovered child fell into it yesterday emergency", "Sanitation", "Critical"),
  ("Hospital waste dumped in open area near homes severe health hazard", "Sanitation", "Critical"),
  ("Drainage overflow flooding streets sewage water entering homes", "Sanitation", "Critical"),
  ("Dead animals on road not removed for days causing disease spread", "Sanitation", "Critical"),
  ("Open garbage dump near school attracting flies causing illness in children", "Sanitation", "Critical"),
  ("Drainage blocked sewage water overflowing on main road", "Sanitation", "Moderate"),
  ("Rats and mosquitoes breeding in garbage pile near busy market", "Sanitation", "Moderate"),
  ("Garbage truck not coming to our area for ten days", "Sanitation", "Moderate"),
  ("Waste dumped near water body causing water pollution concern", "Sanitation", "Moderate"),
  ("Drainage overflow flooding streets during even light rainfall", "Sanitation", "Moderate"),
  ("Street sweeper not visited our lane for weeks bad smell", "Sanitation", "Moderate"),
  ("Public toilet slightly dirty needs cleaning attention", "Sanitation", "Normal"),
  ("Garbage bin overflowing needs extra pickup on weekends", "Sanitation", "Normal"),
  ("Minor drainage smell near community park nuisance for residents", "Sanitation", "Normal"),
  ("Sewage pipe burst raw sewage flowing on street disease outbreak", "Sanitation", "Critical"),
  ("Toxic waste illegally dumped contaminating groundwater emergency", "Sanitation", "Critical"),
  # PUBLIC SAFETY
  ("Chain snatching incidents happening every day near temple women scared", "PublicSafety", "Critical"),
  ("Drug addicts gathering near park scaring children and women residents afraid", "PublicSafety", "Critical"),
  ("Unlicensed firearms seen with local goons near market serious threat", "PublicSafety", "Critical"),
  ("Armed robbers active in residential area multiple houses targeted", "PublicSafety", "Critical"),
  ("Robbery attempt at ATM machine at night area unsafe", "PublicSafety", "Critical"),
  ("Missing child reported please help find urgent situation", "PublicSafety", "Critical"),
  ("Suspicious group loitering near school every afternoon children at risk", "PublicSafety", "Critical"),
  ("Eve teasing and harassment outside college gate daily occurrence", "PublicSafety", "Critical"),
  ("Public fighting and violence near liquor shop daily residents scared", "PublicSafety", "Critical"),
  ("Theft reported at multiple homes in colony within same week", "PublicSafety", "Moderate"),
  ("Stray dogs attacking pedestrians near bus stand multiple injuries", "PublicSafety", "Moderate"),
  ("Illegal gambling den operating in residential area bringing criminals", "PublicSafety", "Moderate"),
  ("Accident prone area multiple accidents weekly no signage installed", "PublicSafety", "Moderate"),
  ("Noise pollution from late night parties disturbing residents", "PublicSafety", "Normal"),
  ("Parking disputes between neighbors causing minor arguments", "PublicSafety", "Normal"),
  ("Murder reported near the market area killer still at large", "PublicSafety", "Critical"),
  ("Bomb threat near government building evacuation needed emergency", "PublicSafety", "Critical"),
]

fake_data = [
  ("The sky is falling aliens are invading our smart city right now", "Road", 1),
  ("My neighbor is a secret spy file complaint against him immediately", "Road", 1),
  ("I want my enemy house demolished for personal revenge purposes", "Road", 1),
  ("asdjkh asjkdh random words test abuse system complaint fake gibberish", "Road", 1),
  ("Government is stealing my wifi signal using transformer outside house", "Road", 1),
  ("I dreamt that roads were bad so filing complaint about my dreams", "Road", 1),
  ("Mayor is lizard person controlling weather with secret machines underground", "Road", 1),
  ("Please give me free money and fix everything in entire universe", "Road", 1),
  ("My cat does not like street light color change it immediately", "Road", 1),
  ("File complaint because my friend did not greet me today morning hello", "Road", 1),
  ("xhsjdh jskdhk jskdh completely random meaningless text submission abuse", "Road", 1),
  ("I want priority treatment above all other citizens for myself alone", "Road", 1),
  ("The moon is causing potholes on my street space conspiracy truth", "Road", 1),
  ("My enemy should be arrested because I dislike him personally neighbor", "Road", 1),
]

texts, cats, pris, fakes = [], [], [], []
for t, c, p in data:
    texts.append(t); cats.append(c); pris.append(p); fakes.append(0)
for t, c, f in fake_data:
    texts.append(t); cats.append(c)
    pris.append("Normal"); fakes.append(f)

df = pd.DataFrame({'text': texts, 'category': cats, 'priority': pris, 'is_fake': fakes})
df['clean'] = df['text'].apply(preprocess)

print(f"Training on {len(df)} samples")

os.makedirs('models', exist_ok=True)

# ── MODEL 1: CATEGORY ──────────────────────────────────────────────────────
cat_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000, min_df=1)),
    ('clf', MultinomialNB(alpha=0.3))
])
cat_pipe.fit(df['clean'], df['category'])
with open('models/category_model.pkl','wb') as f: pickle.dump(cat_pipe, f)

# Verify
acc = accuracy_score(df['category'], cat_pipe.predict(df['clean']))
print(f"Category model accuracy: {acc:.2%}")

# ── MODEL 2: PRIORITY ──────────────────────────────────────────────────────
pri_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000, min_df=1)),
    ('clf', LogisticRegression(max_iter=1000, C=2.0, solver='lbfgs'))
])
pri_pipe.fit(df['clean'], df['priority'])
with open('models/priority_model.pkl','wb') as f: pickle.dump(pri_pipe, f)

acc = accuracy_score(df['priority'], pri_pipe.predict(df['clean']))
print(f"Priority model accuracy: {acc:.2%}")

# ── MODEL 3: FAKE DETECTION ────────────────────────────────────────────────
fake_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000, min_df=1)),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))
])
fake_pipe.fit(df['clean'], df['is_fake'])
with open('models/fake_model.pkl','wb') as f: pickle.dump(fake_pipe, f)

acc = accuracy_score(df['is_fake'], fake_pipe.predict(df['clean']))
print(f"Fake detection accuracy: {acc:.2%}")

with open('models/preprocess_config.pkl','wb') as f:
    pickle.dump({'stop_words': STOP_WORDS}, f)

print("\n✅ All 3 models trained and saved to /models/")
print("   Run: python app.py")
