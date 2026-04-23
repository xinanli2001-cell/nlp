# Part A Report


## 1. Data Source and Domain

Our main dataset is stored in `data/reviews_all.csv`. It contains 814 review-aspect pairs drawn from 789 unique review IDs. The reviews are from the e-commerce electronics domain. We chose this domain for three reasons. First, consumer reviews in this area often mention several product attributes in a single review, which makes the domain suitable for aspect-based sentiment analysis. Second, the products have recurring, concrete attributes such as battery, screen, sound, price, and usability, so the aspect space is easier to define consistently. Third, this domain is strongly relevant to real customer feedback analysis because the same review can contain both praise and criticism for different aspects.

The final project dataset uses one row per review-aspect pair. This is important because the same review text may express different sentiments toward different aspects. For example, a user may like the battery life but dislike the screen quality. Representing the data in `id, review, aspect, sentiment` format makes the data directly usable for aspect-conditioned sentiment classification.


## 2. Aspect Taxonomy

We use 11 aspect categories in the main dataset:

- overall
- usability
- performance
- screen
- price
- sound
- battery
- connectivity
- design
- build_quality
- camera


## 3. Annotation Procedure

Each review-aspect pair was assigned one sentiment label: `positive`, `negative`, or `neutral`.

The annotation rule:
- `positive`: the review expresses clear praise or satisfaction toward the given aspect
- `negative`: the review expresses clear criticism or dissatisfaction toward the given aspect
- `neutral`: the aspect is mentioned without a clear sentiment, or the evidence is mixed and no direction is clearly dominant

The label is tied to the aspect, not to the whole review. A review can sound positive overall while still being negative for one specific aspect. This was especially important in sentences with contrast markers such as `but`, `however`, and `only`, where one part of the sentence may praise one feature and criticise another.

In the second annotation pass, ambiguous cases were handled more conservatively, when an aspect was mentioned without a clearly evaluative judgment, or when both positive and negative cues were present without a dominant direction, the second annotation tended to prefer `neutral`.


## 4. Inter-Annotator Agreement

To check whether the labelling guideline was stable enough, a second annotation file was created as `data/iaa_annotator2.csv`. Cohen's kappa was then computed with the existing project script:

```powershell
python src\data\iaa.py data\reviews_all_clean.csv data\iaa_annotator2.csv
```

We used `data/reviews_all_clean.csv` rather than the original `reviews_all.csv` when running this command because the clean copy removes the BOM character from the first column name and matches the expectations of `src/data/iaa.py`.

The result:
- n_samples: 180
- agreement_rate: 0.7778
- cohens_kappa: 0.6391
- interpretation: substantial

It shows that the annotation scheme is reasonably consistent. The score is not unrealistically perfect, which is expected in aspect-level sentiment annotation, especially for `overall` samples and sentences with mixed evidence. At the same time, it is above the common threshold of 0.6 used to indicate substantial agreement, so it supports the reliability of the dataset.


## 5. Main Dataset Statistics

The project dataset contains 814 rows in total. The sentiment distribution is imbalanced:
- positive: 714 (87.7%)
- negative: 58 (7.1%)
- neutral: 42 (5.2%)

This imbalance is typical of e-commerce review data, where users are more likely to leave positive feedback than balanced or purely negative feedback. It also explains why macro-averaged metrics are more informative than raw accuracy in later model evaluation.

The aspect distribution is also uneven:
- overall: 278 (34.2%)
- usability: 229 (28.1%)
- performance: 63 (7.7%)
- screen: 54 (6.6%)
- price: 52 (6.4%)
- sound: 40 (4.9%)
- battery: 31 (3.8%)
- connectivity: 23 (2.8%)
- design: 19 (2.3%)
- build_quality: 17 (2.1%)
- camera: 8 (1.0%)

## 6. Train / Validation / Test Split

The existing split under `data/final/` contains:
- train: 569 rows
- val: 122 rows
- test: 123 rows

Their sentiment distributions are:
- train: positive 499, negative 40, neutral 30
- val: positive 107, negative 8, neutral 7
- test: positive 108, negative 10, neutral 5


## 7. External Dataset: SemEval 2014 ABSA

To satisfy the requirement of using public ABSA data, we added both SemEval 2014 domains requested in the project plan:
- `data/semeval_laptop.csv`
- `data/semeval_restaurant.csv`

The conversion is handled by the new script `src/data/semeval_loader.py`. This script downloads the original XML file, extracts aspect terms, and converts each record into the same format used by the project:

```csv
id,review,aspect,sentiment
```

Once the data is converted, it can be inspected, compared with the project dataset, or reused in later experiments without writing a separate reader for SemEval XML.


## 8. SemEval Distribution Comparison

### 8.1 Laptop

The converted SemEval laptop dataset contains:
- rows: 2313
- unique review ids: 1462
- positive: 987 (42.7%)
- negative: 866 (37.4%)
- neutral: 460 (19.9%)

Its most common aspect terms include `screen`, `price`, `use`, `battery_life`, and `keyboard`.

### 8.2 Restaurant

The converted SemEval restaurant dataset contains:
- rows: 3602
- unique review ids: 1978
- positive: 2164 (60.1%)
- negative: 805 (22.3%)
- neutral: 633 (17.6%)

Its most common aspect terms include `food`, `service`, `prices`, `place`, and `menu`.


### 8.3 Comparison with the Project Dataset

There are two differences between SemEval and the project dataset.

First, the project dataset is much more positively skewed than either SemEval domain. Our data has 87.7% positive labels, while SemEval laptop is far more balanced between positive and negative labels, and SemEval restaurant still has a noticeably larger negative share than our project data. This means SemEval can be used to show that our project dataset has a stronger real-world review bias toward positive opinions.

Second, the aspect representation is different. Our project dataset uses a fixed 11-class taxonomy, which makes annotation and modelling more manageable. SemEval uses open-vocabulary aspect terms such as `keyboard`, `battery_life`, `service`, and `menu`. This gives SemEval richer lexical diversity, but it also makes the aspect space less compact and less directly aligned with our chosen taxonomy.

Because of these differences, SemEval is useful in two ways. It provides evidence that the project uses more than one ABSA dataset, and it gives a clear comparison point for discussing class imbalance and domain shift in the report.


## 9. Files Produced for Part A

The main A-part data files are:
- `data/iaa_annotator2.csv`
- `data/reviews_all_clean.csv`
- `data/semeval_laptop.csv`
- `data/semeval_restaurant.csv`
- `src/data/semeval_loader.py`

These files are enough to support the full Part B data section, the IAA result, and the SemEval comparison required by the project plan.
