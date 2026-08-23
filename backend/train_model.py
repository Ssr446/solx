"""
SOLX Production Training Script v2
Uses lightweight but high-quality hate speech datasets:
  - Davidson hate speech (25k Twitter tweets) — fast download ~2MB
  - hate_speech18 (10k forum posts) — fast download ~1MB  
  - Our curated Tanglish (weighted 5x for domain specificity)

Avoids large downloads like civil_comments (1.7GB) that time out.
"""
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

print("=" * 60)
print("SOLX Dataset Download & Training v2")
print("=" * 60)

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets -q")
    from datasets import load_dataset

X_combined = []
y_combined = []  # [Toxic, Sarcastic, Cyberbullying]

# ── DATASET 1: Davidson hate speech (25k) ─────────────────────────────────────
print("\n[1/3] Davidson Twitter hate speech (~25k)...")
try:
    davidson = load_dataset("tdavidson/hate_speech_offensive", split="train", trust_remote_code=True)
    counts = [0, 0, 0]
    for row in davidson:
        text = row["tweet"].strip()
        label = row["class"]  # 0=hate, 1=offensive, 2=neither
        if label == 0:
            X_combined.append(text); y_combined.append([1, 0, 1]); counts[0] += 1
        elif label == 1:
            X_combined.append(text); y_combined.append([1, 0, 0]); counts[1] += 1
        else:
            X_combined.append(text); y_combined.append([0, 0, 0]); counts[2] += 1
    print(f"     Loaded {sum(counts)} examples: {counts[0]} hate | {counts[1]} offensive | {counts[2]} clean")
except Exception as e:
    print(f"     ERROR: {e}")

# ── DATASET 2: hate_speech18 (10k Stormfront forum posts) ─────────────────────
print("[2/3] hate_speech18 (~10k)...")
try:
    hs18 = load_dataset("hate_speech18", split="train", trust_remote_code=True)
    hate_c, clean_c = 0, 0
    for row in hs18:
        text = row["text"].strip()
        label = row["label"]  # 0=noHate, 1=hate
        if label == 1:
            X_combined.append(text); y_combined.append([1, 0, 1]); hate_c += 1
        else:
            X_combined.append(text); y_combined.append([0, 0, 0]); clean_c += 1
    print(f"     Loaded {hate_c} hate | {clean_c} clean")
except Exception as e:
    print(f"     ERROR: {e}")

print(f"\nTotal English examples: {len(X_combined)}")

# ── DATASET 2b: Real-world Sarcasm (Internet/Twitter/Reddit style) ────────────
print("[2b/4] Adding real-world internet sarcasm dataset (tweet_eval irony)...")
try:
    sarcasm_ds = load_dataset("tweet_eval", "irony", split="train", trust_remote_code=True)
    sarcasm_c, non_sarcasm_c = 0, 0
    for row in sarcasm_ds:
        text = row["text"].strip()
        label = row["label"]  # 0=non_irony, 1=irony
        if label == 1:
            X_combined.append(text); y_combined.append([0, 1, 0]); sarcasm_c += 1
        else:
            X_combined.append(text); y_combined.append([0, 0, 0]); non_sarcasm_c += 1
    print(f"     Loaded {sarcasm_c} sarcasm | {non_sarcasm_c} clean")
except Exception as e:
    print(f"     ERROR: {e}")

# ── DATASET 2c: Add extra generic English CLEAN examples to fix Davidson bias ──
print("[2c/4] Adding generic clean English examples to fix dataset bias...")
clean_english = [
    "hello how are you", "good morning", "how is life", "what time is it",
    "where are you", "nice to meet you", "see you tomorrow", "have a good day",
    "good night", "how was your day", "I am doing well", "thank you so much",
    "great to hear from you", "hope you are well", "take care", "all the best",
    "have a great weekend", "nice to see you", "catch you later", "sounds good",
    "let me know if you need anything", "happy to help", "glad it worked out",
    "congrats on that", "well done everyone", "looking forward to it",
    "that is a great idea", "I agree with you", "makes sense to me",
    "thanks for letting me know", "appreciate the update", "got it thanks",
    "no problem at all", "you are welcome", "of course", "absolutely",
    "see you later", "talk soon", "have fun", "enjoy your day",
    "best of luck", "stay safe", "hope it goes well", "fingers crossed",
    "sounds like a plan", "count me in", "I will be there", "on my way",
    "just checking in", "how have you been", "long time no see",
    "miss you guys", "thinking of you", "hope you feel better soon",
    "happy birthday", "congratulations", "welcome to the team",
    "nice work on that", "great effort", "impressive work",
    "the food was amazing", "beautiful day outside", "lovely weather",
    "just finished reading it", "watched a good movie", "had a great time",
    "the presentation went well", "meeting was productive",
    "just got home", "heading out now", "on my break",
    "grabbing coffee", "just had lunch", "back at my desk",
    "working from home today", "about to start class",
    "what did you think of it", "did you see that",
    "check this out", "have you heard about this", "interesting article",
    "recommended this to a friend", "shared the link", "bookmarked it",
    "signed up for it", "registered already", "downloaded the app",
    "the event was fun", "really enjoyed it", "would do it again",
    "definitely recommend", "worth it", "pretty cool",
    "not bad at all", "better than expected", "quite impressive",
    "simple and clean", "easy to use", "very intuitive",
]
# Add each 8 times to strongly balance against Davidson noise
for phrase in clean_english:
    for _ in range(8):
        X_combined.append(phrase)
        y_combined.append([0, 0, 0])
print(f"     Added {len(clean_english) * 8} clean English entries")

# ── DATASET 3: Curated Tanglish (weighted 10x) ────────────────────────────────
print("[3/3] Injecting curated Tanglish data (10x weighted)...")

tanglish_data = [
    # CLEAN
    ("super da machaan keep it up", [0, 0, 0]),
    ("romba nalla panre", [0, 0, 0]),
    ("semma work da keep going bro", [0, 0, 0]),
    ("mass da thalaiva", [0, 0, 0]),
    ("verithanamana performance", [0, 0, 0]),
    ("vannakam", [0, 0, 0]),
    ("saptiya", [0, 0, 0]),
    ("enge pora", [0, 0, 0]),
    ("naan veetuku poren", [0, 0, 0]),
    ("thanks bro", [0, 0, 0]),
    ("romba thanks", [0, 0, 0]),
    ("dei un peru enna", [0, 0, 0]),
    ("dei enna achu", [0, 0, 0]),
    ("dei epdi iruka", [0, 0, 0]),
    ("dei vaa da", [0, 0, 0]),
    ("dei solu da", [0, 0, 0]),
    ("dei enna panra", [0, 0, 0]),
    ("dei sapitiya", [0, 0, 0]),
    ("dei school povom", [0, 0, 0]),
    ("dei enna solra", [0, 0, 0]),
    ("dei nee engeya", [0, 0, 0]),
    ("dei namaku late aaguthu", [0, 0, 0]),
    ("dei scene paru da", [0, 0, 0]),
    ("dei un veetuku poren", [0, 0, 0]),
    ("un peru enna", [0, 0, 0]),
    ("un veettu address enna", [0, 0, 0]),
    ("un phone number enna", [0, 0, 0]),
    ("enna solra", [0, 0, 0]),
    ("enna achu", [0, 0, 0]),
    ("enna nadakuthu", [0, 0, 0]),
    ("enna padam pakkalam", [0, 0, 0]),
    ("en peru roshini", [0, 0, 0]),
    ("nee enge pore", [0, 0, 0]),
    ("nee saptiya", [0, 0, 0]),
    ("nee college la irukkiya", [0, 0, 0]),
    ("sollu da", [0, 0, 0]),
    ("yenna da nadakuthu", [0, 0, 0]),
    ("epdi iruka", [0, 0, 0]),
    ("nandri", [0, 0, 0]),
    ("good job", [0, 0, 0]),
    # SARCASM
    ("enna koduma saravanan idhu", [0, 1, 0]),
    ("avaru periya aptakar da enna pannuvanga", [0, 1, 1]),
    ("periya einstein vandhutaar parunga", [0, 1, 1]),
    ("mokka padam da idhu", [0, 1, 0]),
    ("vetti officer vandhutar", [0, 1, 0]),
    ("sothula uppu pottu than saapidriya", [0, 1, 0]),
    ("ivanukku onnum theriyathu aptakar nu nenappu", [0, 1, 1]),
    ("periya boss vandhutar", [0, 1, 0]),
    ("nallavanga parunga ivana", [0, 1, 1]),
    ("ada paavame", [0, 1, 0]),
    ("semma plan potirukan", [0, 1, 0]),
    ("periya logic solran parunga", [0, 1, 0]),
    ("yov periya scientist vandhutu", [0, 1, 0]),
    ("avaru periya intellectuals sir", [0, 1, 0]),
    ("romba intelligent da nee", [0, 1, 0]),
    ("oh wow super genius", [0, 1, 0]),
    # TOXIC TANGLISH
    ("loosu payale unakku arivu illaya", [1, 0, 1]),
    ("kulla naaye unakku enna theriyum", [1, 0, 1]),
    ("nee oru thevai illatha fellow da", [1, 0, 1]),
    ("goyyale adichu pallu udachiduven", [1, 0, 1]),
    ("dubakoor velai panra", [1, 0, 0]),
    ("poramboku", [1, 0, 1]),
    ("di nee enna panna pore useless fella", [1, 0, 1]),
    ("dai arivuketta mudal", [1, 0, 1]),
    ("poda panni", [1, 0, 1]),
    ("ivan oru makku", [1, 0, 1]),
    ("nee oru loosu da", [1, 0, 1]),
    ("poda loosu", [1, 0, 1]),
    ("poda venna", [1, 0, 1]),
    ("dei mokkai", [1, 0, 1]),
    ("nee oru kazhuthai da", [1, 0, 1]),
    ("pottai payale", [1, 0, 1]),
    ("avan oru koothi", [1, 0, 1]),
    ("dai poda koothi", [1, 0, 1]),
    ("poda punda", [1, 0, 1]),
    ("punda payale", [1, 0, 1]),
    ("punda mairu", [1, 0, 1]),
    ("thevidiya", [1, 0, 1]),
    ("otha thevidiya", [1, 0, 1]),
    ("mairu da", [1, 0, 1]),
    ("sunni da", [1, 0, 1]),
    ("naaye poda", [1, 0, 1]),
    ("baadu payale", [1, 0, 1]),
    ("thayoli", [1, 0, 1]),
    ("maairu", [1, 0, 1]),
    # MIXED
    ("ivan aptakar nu avan thaane nenapukiran bitch", [1, 1, 1]),
    ("periya genius vara parunga mokkai payale", [1, 1, 1]),
    ("super idea da loosu", [1, 1, 1]),
]

# Weight Tanglish 10x to compensate for small size vs large English corpus
TANGLISH_WEIGHT = 10
for _ in range(TANGLISH_WEIGHT):
    for text, label in tanglish_data:
        X_combined.append(text)
        y_combined.append(label)

print(f"Total dataset size: {len(X_combined)} examples")

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\nTraining model...")

X_arr = np.array(X_combined)
y_arr = np.array(y_combined)
idx = np.random.RandomState(42).permutation(len(X_arr))
X_arr = X_arr[idx].tolist()
y_arr = y_arr[idx]

features = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        min_df=2, sublinear_tf=True, max_features=100000
    )),
    ('char_tfidf', TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5),
        min_df=2, sublinear_tf=True, max_features=100000
    )),
])

base_clf = CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, C=0.5))
pipeline = Pipeline([
    ('features', features),
    ('clf', MultiOutputClassifier(base_clf, n_jobs=-1))
])

pipeline.fit(X_arr, y_arr)
print("Training complete!")

with open("solx_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

size_mb = os.path.getsize("solx_model.pkl") / (1024 * 1024)
print(f"Model saved ({size_mb:.1f} MB)")

# ─── SANITY TESTS ─────────────────────────────────────────────────────────────
print("\n=== Sanity Tests ===")
tests = [
    ("dei un peru enna",            "CLEAN"),
    ("hello how are you",           "CLEAN"),
    ("romba nalla panre",           "CLEAN"),
    ("hey bitch how are you",       "HIGH RISK"),
    ("poda punda",                  "HIGH RISK"),
    ("you are worthless nobody",    "HIGH RISK"),
    ("go fuck yourself",            "HIGH RISK"),
    ("i hate you so much",          "RISK"),
    ("enna koduma saravanan idhu",  "SARCASM"),
    ("periya einstein vandhutaar",  "SARCASM"),
    ("loosu payale unakku",         "HIGH RISK"),
]

for text, expected in tests:
    probs = pipeline.predict_proba([text])
    t = round(probs[0][0][1] * 100, 1)
    s = round(probs[1][0][1] * 100, 1)
    c = round(probs[2][0][1] * 100, 1)
    verdict = "HIGH RISK" if max(t,c) >= 70 else "RISK" if max(t,c) >= 30 else "SARCASM" if s >= 50 else "CLEAN"
    match = "OK" if verdict == expected else "WARN"
    print(f"  [{match}] '{text[:40]}' => T:{t}% S:{s}% C:{c}%  ({verdict})")
