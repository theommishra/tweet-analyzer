"""
dataset_builder.py
==================
Downloads and merges multiple real emotion datasets to create a high-quality
training corpus for Ekman's 7-class taxonomy:
  joy, sadness, anger, fear, surprise, disgust, neutral

Sources:
  1. GoEmotions (Google Research) — 54k Reddit comments, 28 fine-grained labels
     -> mapped to Ekman 7
  2. ISEAR (International Survey on Emotion Antecedents and Reactions)
     -> native Ekman-like 7-class labels
  3. Curated seed dataset (internal) for edge-cases and balance

All data is fetched via urllib (no extra dependencies needed beyond stdlib).
"""

import io
import csv
import json
import random
import urllib.request
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# GoEmotions label -> Ekman 7 mapping
# Index order matches emotions.txt from the GoEmotions repo
# ---------------------------------------------------------------------------
# 0:admiration 1:amusement 2:anger 3:annoyance 4:approval 5:caring
# 6:confusion 7:curiosity 8:desire 9:disappointment 10:disapproval
# 11:disgust 12:embarrassment 13:excitement 14:fear 15:gratitude
# 16:grief 17:joy 18:love 19:nervousness 20:optimism 21:pride
# 22:realization 23:relief 24:remorse 25:sadness 26:surprise 27:neutral

GO_EMOTIONS_TO_EKMAN = {
    0:  "joy",       # admiration
    1:  "joy",       # amusement
    2:  "anger",     # anger
    3:  "anger",     # annoyance
    4:  "neutral",   # approval
    5:  "neutral",   # caring (Plutchik: trust)
    6:  "surprise",  # confusion -> uncertainty/unexpected, maps to surprise
    7:  "neutral",   # curiosity (Plutchik: anticipation)
    8:  "neutral",   # desire  (Plutchik: anticipation)
    9:  "sadness",   # disappointment
    10: "disgust",   # disapproval
    11: "disgust",   # disgust
    12: "sadness",   # embarrassment
    13: "joy",       # excitement
    14: "fear",      # fear
    15: "joy",       # gratitude
    16: "sadness",   # grief
    17: "joy",       # joy
    18: "joy",       # love
    19: "fear",      # nervousness
    20: "joy",       # optimism
    21: "joy",       # pride
    22: "surprise",  # realization
    23: "joy",       # relief
    24: "sadness",   # remorse
    25: "sadness",   # sadness
    26: "surprise",  # surprise
    27: "neutral",   # neutral
}

# Additional fear-specific sentences to compensate for class imbalance
# (GoEmotions naturally under-represents explicit fear/terror language)
FEAR_AUGMENTATION = [
    ("I am so afraid of the dark, always have been", "fear"),
    ("The thought of being alone in that house makes me tremble", "fear"),
    ("I froze in complete terror when I heard the intruder", "fear"),
    ("I dread going to the dentist more than anything in the world", "fear"),
    ("The earthquake alarm sent a wave of panic through the office", "fear"),
    ("I am scared stiff about the job interview tomorrow", "fear"),
    ("Driving over that bridge always fills me with dread", "fear"),
    ("The strange noises outside kept me up all night with fear", "fear"),
    ("I am petrified of spiders and cannot even look at a picture", "fear"),
    ("Being in a crowd gives me unbearable anxiety", "fear"),
    ("She was shaking with terror when the car skidded on the ice", "fear"),
    ("I am constantly worried something terrible will happen to my kids", "fear"),
    ("The panic attack hit me out of nowhere and I could not breathe", "fear"),
    ("I feel sick with anxiety every time I have to speak publicly", "fear"),
    ("There is a knot of dread in my stomach that won't go away", "fear"),
    ("The horror film kept me awake all night, too scared to sleep", "fear"),
    ("My hands shake every time I think about what might happen", "fear"),
    ("I jumped three feet in the air when the door slammed shut", "fear"),
    ("I am terrified of losing everyone I love", "fear"),
    ("The uncertainty of the diagnosis fills me with absolute dread", "fear"),
    ("I cannot stop catastrophising about every possible bad outcome", "fear"),
    ("Being alone in that neighbourhood at night made my blood run cold", "fear"),
    ("Just reading the news fills me with constant anxiety and fear", "fear"),
    ("I am so scared of failure that I can barely function some days", "fear"),
    ("The sound of thunder still terrifies me at my age", "fear"),
    ("I have a phobia of flying that no amount of therapy has cured", "fear"),
    ("The waiting room before surgery was the most frightening place", "fear"),
    ("He looked at me with such menace I felt true terror for the first time", "fear"),
    ("I am constantly anxious that I have said the wrong thing to someone", "fear"),
    ("The nightmare felt so real that the fear stayed with me all day", "fear"),
    ("I am absolutely dreading the results coming back this week", "fear"),
    ("The panic I feel before every exam is completely debilitating", "fear"),
    ("I felt a wave of cold fear as I realised someone was following me", "fear"),
    ("The dark water below the bridge terrified me and I could not look down", "fear"),
    ("I woke up in a cold sweat from the most frightening dream", "fear"),
    ("Even the word cancer brings on a wave of pure terror", "fear"),
    ("I cannot bring myself to check my email because I am so scared", "fear"),
    ("Losing control of the car sent pure adrenaline-fuelled fear through me", "fear"),
    ("Every time my phone rings late at night I fear the worst", "fear"),
    ("The fear of rejection stops me from even trying half the time", "fear"),
    ("I am so frightened about the future that I can barely plan ahead", "fear"),
    ("She screamed in terror when the figure appeared from the shadows", "fear"),
    ("The prospect of public humiliation fills me with gut-wrenching dread", "fear"),
    ("The turbulence during the flight had everyone gripping their seats in fear", "fear"),
    ("I feel permanently on edge, like something awful is always about to happen", "fear"),
    ("The growling dog coming toward me made my heart stop with fright", "fear"),
    ("I cannot watch horror films because they haunt me for weeks", "fear"),
    ("Just thinking about it makes me feel sick with worry", "fear"),
    ("I have never felt fear like that before in my life", "fear"),
    ("The silence after the explosion was somehow more terrifying than the blast", "fear"),
]

EKMAN_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

GOEMOTIONS_URLS = [
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv",
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv",
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv",
]

ISEAR_URL = (
    "https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/"
    "master/isear.csv"
)

ISEAR_LABEL_MAP = {
    "joy":     "joy",
    "fear":    "fear",
    "anger":   "anger",
    "sadness": "sadness",
    "disgust": "disgust",
    "shame":   "disgust",
    "guilt":   "sadness",
}

# ---------------------------------------------------------------------------
# Curated seed sentences covering edge-cases, negations, emojis, Plutchik
# ---------------------------------------------------------------------------
SEED_DATA = [
    # JOY -- admiration, gratitude, relief, love, pride, excitement
    ("I am so deeply grateful for everything you have done for me", "joy"),
    ("I admire your incredible courage and resilience more than words can say", "joy"),
    ("Finally the surgery went well and I feel such relief and warmth", "joy"),
    ("I am bursting with pride seeing how far my child has come", "joy"),
    ("We just got the keys to our new home and I could not be happier", "joy"),
    ("That performance was breathtaking, pure joy from start to finish", "joy"),
    ("I love waking up next to you every single morning", "joy"),
    ("We won the championship and the celebrations went all night", "joy"),
    ("Receiving this award means the world to me, I am overjoyed", "joy"),
    ("This is the most delightful and wonderful surprise I could have hoped for", "joy"),
    ("I feel so blessed and content with this life", "joy"),
    ("Every moment with you fills me with such joy and warmth", "joy"),
    ("The garden is blooming and the sunlight is so beautiful today", "joy"),
    ("She passed her driving test on the first try, we are so proud", "joy"),
    ("I can not stop smiling, today was absolutely perfect", "joy"),

    # SADNESS -- grief, remorse, disappointment, embarrassment
    ("I still cannot believe she is gone, the grief is unbearable", "sadness"),
    ("I deeply regret the hurtful words I said and I am so sorry", "sadness"),
    ("After years of effort the project was cancelled without warning", "sadness"),
    ("I feel so ashamed for letting everyone down like that", "sadness"),
    ("The loneliness creeps in late at night and does not let go", "sadness"),
    ("I miss my hometown so much it physically hurts sometimes", "sadness"),
    ("My heart is completely broken after the breakup last week", "sadness"),
    ("I cried for hours when I heard the devastating news", "sadness"),
    ("Nothing feels right anymore, everything feels empty and grey", "sadness"),
    ("I gave it everything I had and still failed, so disappointed in myself", "sadness"),
    ("He did not even say goodbye before he left for good", "sadness"),
    ("I feel so unworthy and invisible around these people", "sadness"),
    ("The loss of that friendship still leaves an ache in my chest", "sadness"),

    # ANGER -- annoyance, disapproval, frustration, rage
    ("How dare they treat employees like that, it is utterly disgraceful", "anger"),
    ("I am absolutely livid at the blatant corruption and dishonesty", "anger"),
    ("Stop gaslighting me, I know exactly what you said and did", "anger"),
    ("This traffic is absolutely infuriating, I am going to be late again", "anger"),
    ("They lied to my face and I cannot contain my rage", "anger"),
    ("It makes me furious that nobody is being held accountable", "anger"),
    ("I am so done with being talked over and disrespected in meetings", "anger"),
    ("You crossed a line that cannot be uncrossed and I am seething", "anger"),
    ("The disrespect and entitlement of some people make me boil", "anger"),
    ("I cannot stand the injustice happening right in front of us", "anger"),
    ("He broke my trust deliberately and I will never forgive that", "anger"),

    # FEAR -- nervousness, terror, dread, anxiety, phobia
    ("I am terrified to present to the entire board tomorrow morning", "fear"),
    ("The thought of losing my job keeps me up every single night", "fear"),
    ("I had a panic attack on the highway and could not breathe", "fear"),
    ("I am scared of what the biopsy results are going to show", "fear"),
    ("My heart pounds every time I have to fly, I dread it", "fear"),
    ("The eerie silence in the house terrified me when I got home", "fear"),
    ("I am petrified to walk alone at night in that neighbourhood", "fear"),
    ("The unknown is more terrifying to me than any visible danger", "fear"),
    ("I have been dreading this conversation for months", "fear"),
    ("The news about the earthquake nearby left me frozen with fear", "fear"),
    ("Just thinking about it sends chills down my spine", "fear"),

    # SURPRISE -- amazement, shock, confusion, disbelief
    ("I cannot believe what just happened, this is completely unexpected", "surprise"),
    ("Wait, seriously? You are kidding me right? That is shocking", "surprise"),
    ("Out of absolutely nowhere she announced she was moving abroad", "surprise"),
    ("I was completely blindsided by the sudden policy change", "surprise"),
    ("My jaw literally dropped when I saw the final result", "surprise"),
    ("Nobody saw that twist coming, the whole room gasped", "surprise"),
    ("I am still in disbelief, I never thought this would happen", "surprise"),
    ("That revelation changed everything I thought I knew", "surprise"),
    ("How is that even possible? I am absolutely stunned", "surprise"),
    ("The audacity of that move left everyone speechless", "surprise"),

    # DISGUST -- revulsion, disapproval, contempt, repulsion
    ("The smell coming from that bin is revolting, I nearly gagged", "disgust"),
    ("I find his entitled and condescending attitude repulsive", "disgust"),
    ("Seeing that level of cruelty towards animals makes me sick", "disgust"),
    ("That video was deeply disturbing and I wish I had not seen it", "disgust"),
    ("The corruption in that organisation is utterly nauseating", "disgust"),
    ("I cannot look at that mould growing everywhere, it is gross", "disgust"),
    ("Everything about that interaction left me feeling contaminated", "disgust"),
    ("His smug attitude and contempt for others is disgusting to witness", "disgust"),
    ("The thought of what they did to those people makes me feel ill", "disgust"),

    # NEUTRAL -- factual, informational, procedural
    ("The meeting has been rescheduled to Tuesday at ten thirty", "neutral"),
    ("The temperature today reached a high of twenty three degrees", "neutral"),
    ("The package will be delivered between nine and twelve tomorrow", "neutral"),
    ("She is currently enrolled in the second year of the programme", "neutral"),
    ("The document requires three signatures before it can be processed", "neutral"),
    ("The nearest hospital is approximately four kilometres from here", "neutral"),
    ("The server logs are rotated automatically every seven days", "neutral"),
    ("Please complete the form and submit it before the deadline", "neutral"),
    ("The committee meets on the first Monday of every month", "neutral"),
    ("He works as a software engineer at a mid-sized technology firm", "neutral"),
    ("Water freezes at zero degrees Celsius under standard conditions", "neutral"),
    ("The film runs for approximately two hours and fifteen minutes", "neutral"),
    ("The new policy takes effect from the first of next month", "neutral"),

    # NEGATION edge cases
    ("I am not happy at all with how things turned out", "sadness"),
    ("This is not funny in the slightest, I am not amused", "anger"),
    ("I do not trust him, not even a little bit", "disgust"),
    ("She was not surprised at all when they finally broke up", "neutral"),
    ("I can not stop worrying no matter what I do", "fear"),
    ("It never gets easier, not even with time", "sadness"),
    ("I hardly feel anything anymore, I am just numb", "sadness"),
    ("I am not scared, I am absolutely terrified", "fear"),

    # Emoji examples
    ("Just got promoted today! Best day ever!", "joy"),
    ("My heart is completely shattered, I don't know how to move on", "sadness"),
    ("I am SO DONE with this nonsense this is absolutely ridiculous", "anger"),
    ("That jump scare literally made me scream I was so scared", "fear"),
    ("No way I can't believe that actually happened!", "surprise"),
    ("This is absolutely disgusting how do people live like this", "disgust"),
    ("The report will be ready by 5pm as previously discussed", "neutral"),
]


def fetch_url_text(url: str, timeout: int = 20) -> str | None:
    """Fetch URL and return text, or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "emotion-dataset-builder/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return None


def load_goemotions() -> list:
    """
    Download GoEmotions and apply STRICT quality filters:
    - Single-label only (no multi-label ambiguity)
    - Between 4-55 tokens
    - Exclude anonymised [NAME] entries
    - Only use GoEmotions label indices that map cleanly to Ekman
    """
    # Label indices that map cleanly to Ekman without ambiguity
    CLEAN_LABEL_INDICES = {
        0,  # admiration  -> joy
        1,  # amusement   -> joy
        2,  # anger       -> anger
        3,  # annoyance   -> anger
        9,  # disappointment -> sadness
        11, # disgust     -> disgust
        12, # embarrassment -> sadness
        13, # excitement  -> joy
        14, # fear        -> fear
        15, # gratitude   -> joy
        16, # grief       -> sadness
        17, # joy         -> joy
        18, # love        -> joy
        19, # nervousness -> fear
        24, # remorse     -> sadness
        25, # sadness     -> sadness
        26, # surprise    -> surprise
        27, # neutral     -> neutral
    }

    print("Downloading GoEmotions dataset (3 splits, strict quality filter)...")
    records = []

    for url in GOEMOTIONS_URLS:
        text = fetch_url_text(url)
        if not text:
            continue
        for line in text.splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            sentence = parts[0].strip()

            # Skip anonymised entries with placeholder tokens
            if "[NAME]" in sentence or "[RELIGION]" in sentence or "[NATIONALITY]" in sentence:
                continue

            # Token length filter: 4-55 words
            tokens = sentence.split()
            if len(tokens) < 4 or len(tokens) > 55:
                continue

            # Single-label only
            raw_labels = [int(x) for x in parts[1].split(",") if x.strip().isdigit()]
            if len(raw_labels) != 1:
                continue

            label_idx = raw_labels[0]
            if label_idx not in CLEAN_LABEL_INDICES:
                continue

            ekman = GO_EMOTIONS_TO_EKMAN.get(label_idx, "neutral")
            records.append((sentence, ekman))

    print(f"  GoEmotions filtered: {len(records):,} high-confidence samples")
    return records


def load_curated() -> list:
    """Load the hand-curated high-quality unambiguous emotion sentences."""
    try:
        from curated_dataset import ALL_CURATED
        print(f"  Curated dataset: {len(ALL_CURATED):,} clean samples loaded")
        return list(ALL_CURATED)
    except ImportError:
        print("  [WARN] curated_dataset.py not found")
        return []


def balance_dataset(records: list, max_per_class: int = 3000, seed: int = 42) -> list:
    """
    Cap each class at max_per_class using deterministic shuffling.
    For classes with fewer samples than min_per_class, all samples are kept.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for text, label in records:
        by_class[label].append((text, label))

    balanced = []
    for label in EKMAN_LABELS:
        items = by_class.get(label, [])
        rng.shuffle(items)
        items = items[:max_per_class]
        balanced.extend(items)

    rng.shuffle(balanced)
    return balanced


def deduplicate(records: list) -> list:
    """Remove duplicate sentences (keep first occurrence, case-insensitive)."""
    seen = set()
    out = []
    for text, label in records:
        key = text.lower().strip()
        if key not in seen:
            seen.add(key)
            out.append((text, label))
    return out


def build_dataset(max_per_class: int = 3000, output_path: str = "emotion_dataset.json") -> list:
    """
    Full pipeline:
    1. Download GoEmotions (strictly filtered) -> deduplicate
    2. Load curated high-quality sentences (oversampled AFTER dedup)
    3. Balance classes -> save
    """
    # 1. GoEmotions (strict quality filtered), then dedup
    go_records = load_goemotions()
    go_records = deduplicate(go_records)
    print(f"  GoEmotions after dedup: {len(go_records):,}")

    # 2. Curated clean sentences oversampled 15x (added AFTER dedup, intentional)
    curated = load_curated()
    print(f"  Oversampling curated data 15x -> {len(curated) * 15:,} weighted samples")

    # 3. Fear augmentation oversampled to fix thin class
    fear_aug = [(t, l) for t, l in FEAR_AUGMENTATION]
    seed = [(t, l) for t, l in SEED_DATA]

    all_records = go_records + (curated * 15) + (fear_aug * 8) + (seed * 5)

    print(f"\nRaw combined corpus: {len(all_records):,} samples")

    before_dist = Counter(label for _, label in all_records)
    print("\nClass distribution (before balancing):")
    for label in EKMAN_LABELS:
        print(f"  {label:10s}: {before_dist.get(label, 0):,}")

    final_records = balance_dataset(all_records, max_per_class=max_per_class)

    after_dist = Counter(label for _, label in final_records)
    print(f"\nFinal balanced dataset: {len(final_records):,} samples")
    print("Class distribution (after balancing):")
    for label in EKMAN_LABELS:
        print(f"  {label:10s}: {after_dist.get(label, 0):,}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
    print(f"\nDataset saved to '{output_path}'")

    return final_records


if __name__ == "__main__":
    build_dataset(max_per_class=4000)
